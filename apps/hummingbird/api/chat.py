"""
=============================================================================
HUMMINGBIRD-LEA - Chat API
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Chat endpoints for interacting with agents.

Phase 2 Updates:
- Added reasoning visibility option
- Added confidence score and factors to response
- Added clarifying questions to response
- Added reasoning steps endpoint
=============================================================================
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.utils.auth import User, get_current_user, get_optional_user
from core.agents import (
    get_agent_manager,
    AgentResponse,
    ConversationContext,
)
from core.providers.ollama import Message
from .microsoft import _load_token, _graph_get

router = APIRouter()


async def _handle_lea_microsoft_shortcut(message: str):
    m = (message or "").lower()
    wants_email = any(k in m for k in ["email", "inbox", "outlook", "messages"])
    wants_calendar = any(k in m for k in ["calendar", "meeting", "events", "schedule"])

    if not wants_email and not wants_calendar:
        return None

    token = _load_token()
    if not token or "access_token" not in token:
        return ChatResponse(
            agent="lea",
            content="I can access your Microsoft data, but you're not connected right now. Please connect via /auth/microsoft/login and I’ll proceed.",
            confidence="high",
            model="microsoft-graph",
        )

    try:
        if wants_calendar:
            data = await _graph_get("/me/events?$top=10&$select=subject,organizer,start,end,location")
            items = (data or {}).get("value", [])[:5]
            if not items:
                content = "I checked your calendar and found no upcoming events."
            else:
                lines = ["I checked your calendar. Here are the next events:"]
                for e in items:
                    subject = e.get("subject", "(No Subject)")
                    start = (e.get("start") or {}).get("dateTime", "")
                    location = (e.get("location") or {}).get("displayName", "")
                    lines.append(f"- {subject} | {start} | {location}")
                content = "\n".join(lines)
        else:
            data = await _graph_get("/me/mailFolders/inbox/messages?$top=10&$select=id,subject,from,toRecipients,ccRecipients,receivedDateTime,isRead,bodyPreview&$orderby=receivedDateTime desc")
            items = (data or {}).get("value", [])[:5]
            if not items:
                content = "I checked your inbox and found no recent emails."
            else:
                lines = ["I checked your inbox. Here are the latest emails:"]
                for e in items:
                    subject = e.get("subject", "(No Subject)")
                    sender = ((e.get("from") or {}).get("emailAddress") or {}).get("address", "Unknown")
                    received = e.get("receivedDateTime", "")
                    lines.append(f"- {subject} | from {sender} | {received}")
                content = "\n".join(lines)

        return ChatResponse(
            agent="lea",
            content=content,
            confidence="high",
            model="microsoft-graph",
        )
    except Exception as ex:
        return ChatResponse(
            agent="lea",
            content=f"I tried to access Microsoft data but hit an error: {str(ex)}",
            confidence="low",
            model="microsoft-graph",
        )



# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """A single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Request to chat with an agent"""
    message: str = Field(..., description="User's message", min_length=1)
    agent: Optional[str] = Field("lea", description="Agent to chat with: lea, chiquis, or grant")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Conversation history")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    show_reasoning: Optional[bool] = Field(False, description="Whether to show agent reasoning (Phase 2)")
    # Phase 4: Vision support
    image_base64: Optional[str] = Field(None, description="Base64 encoded image for vision analysis")


class ChatResponse(BaseModel):
    """Response from the agent"""
    agent: str = Field(..., description="Agent name")
    content: str = Field(..., description="Agent's response")
    confidence: str = Field(..., description="Confidence level: low, medium, high")
    confidence_score: float = Field(1.0, description="Numeric confidence score (0.0-1.0)")
    model: Optional[str] = Field(None, description="Model used for response")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Phase 2 additions
    needs_clarification: bool = Field(False, description="Whether the agent needs clarification")
    clarifying_questions: Optional[List[str]] = Field(None, description="Questions to ask user")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning process (if shown)")
    reasoning_steps: Optional[List[Dict[str, str]]] = Field(None, description="Structured reasoning steps")
    sources: Optional[List[str]] = Field(None, description="Sources cited in response")

    # Phase 4 additions
    image_analysis: Optional[Dict[str, Any]] = Field(None, description="Image analysis results if image was provided")


class AgentInfo(BaseModel):
    """Information about an agent"""
    name: str
    role: str
    capabilities: List[str]


class ReasoningStepInfo(BaseModel):
    """Information about a reasoning step"""
    step_type: str
    content: str


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """
    List all available agents.

    Returns information about Lea, Chiquis, and Grant.
    """
    manager = get_agent_manager()
    return manager.list_agents()


@router.get("/agents/{agent_name}", response_model=AgentInfo)
async def get_agent_info(agent_name: str):
    """
    Get information about a specific agent.
    """
    manager = get_agent_manager()
    info = manager.get_agent_info(agent_name)

    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found. Available: lea, chiquis, grant"
        )

    return info


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Chat with an agent.

    Send a message and get a response from Lea, Chiquis, or Grant.

    **Authentication:** Optional but recommended.

    **Phase 2 Features:**
    - Set `show_reasoning: true` to see the agent's thinking process
    - Response includes `confidence_score` (0.0-1.0) and confidence factors
    - If ambiguity detected, response includes `clarifying_questions`

    **Example:**
    ```json
    {
        "message": "Hello Lea!",
        "agent": "lea",
        "history": [],
        "show_reasoning": false
    }
    ```
    """
    manager = get_agent_manager()

    # Get the requested agent
    agent = manager.get(request.agent or "lea")
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent: {request.agent}. Available: lea, chiquis, grant"
        )

    # Build conversation context
    context = ConversationContext(
        user_name=user.username if user else "Friend",
    )

    # Add history to context
    for msg in (request.history or []):
        context.messages.append(Message(
            role=msg.role,
            content=msg.content,
        ))

    # Phase 4: Handle image analysis if image provided
    image_analysis = None
    enhanced_message = request.message

    if request.image_base64:
        try:
            from core.services.vision import get_analysis_engine

            engine = get_analysis_engine()
            analysis_result = await engine.analyze(
                image=request.image_base64,
                question=request.message if "?" in request.message else None,
            )

            image_analysis = analysis_result.to_dict()

            # Enhance the message with image context for the agent
            image_context = f"\n\n[Image Analysis]\nDescription: {analysis_result.description}"
            if analysis_result.has_text:
                image_context += f"\nExtracted Text: {analysis_result.extracted_text[:500]}"
            enhanced_message = f"{request.message}{image_context}"

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Image analysis failed: {e}")
            # Continue without image analysis

    # Fast path: Lea Microsoft actions (non-streaming)
    if (request.agent or "lea") == "lea" and not request.stream:
        shortcut = await _handle_lea_microsoft_shortcut(request.message)
        if shortcut is not None:
            return shortcut

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            _stream_response(agent, enhanced_message, context),
            media_type="text/event-stream",
        )

    # Process the message
    try:
        response = await agent.process(
            user_message=enhanced_message,
            context=context,
            show_reasoning=request.show_reasoning or False,
        )

        return ChatResponse(
            agent=response.agent,
            content=response.content,
            confidence=response.confidence.value,
            confidence_score=response.confidence_score,
            model=response.metadata.get("model"),
            timestamp=response.timestamp,
            # Phase 2 additions
            needs_clarification=response.needs_clarification,
            clarifying_questions=response.clarifying_questions,
            reasoning=response.reasoning,
            reasoning_steps=response.reasoning_steps,
            sources=response.sources,
            # Phase 4 additions
            image_analysis=image_analysis,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Chat with an agent using streaming response.

    Returns a Server-Sent Events stream.

    **Note:** Streaming responses may pause to ask clarifying questions
    if ambiguity is detected (Phase 2 feature).

    **Usage with JavaScript:**
    ```javascript
    const eventSource = new EventSource('/api/chat/stream');
    eventSource.onmessage = (event) => {
        console.log(event.data);
    };
    ```
    """
    manager = get_agent_manager()

    agent = manager.get(request.agent or "lea")
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent: {request.agent}"
        )

    context = ConversationContext(
        user_name=user.username if user else "Friend",
    )

    for msg in (request.history or []):
        context.messages.append(Message(
            role=msg.role,
            content=msg.content,
        ))

    return StreamingResponse(
        _stream_response(agent, request.message, context),
        media_type="text/event-stream",
    )


async def _stream_response(agent, message: str, context: ConversationContext):
    """Generator for streaming responses"""
    try:
        async for chunk in agent.process_stream(message, context):
            # Format as Server-Sent Event
            yield "data: " + json.dumps({"t": chunk}) + "\n\n"

        # End of stream
        yield "data: " + json.dumps({"done": True}) + "\n\n"

    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"


@router.get("/greeting/{agent_name}")
async def get_greeting(agent_name: str):
    """
    Get a greeting from a specific agent.

    Useful for initializing a chat interface.
    """
    manager = get_agent_manager()
    agent = manager.get(agent_name)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found"
        )

    return {
        "agent": agent.name,
        "greeting": agent.get_greeting(),
    }


# =============================================================================
# Phase 2: Reasoning Endpoints
# =============================================================================

@router.get("/reasoning/{agent_name}")
async def get_reasoning_summary(agent_name: str):
    """
    Get the reasoning summary from the agent's last response.

    **Phase 2 Feature:** Shows how the agent thought through the last request.
    """
    manager = get_agent_manager()
    agent = manager.get(agent_name)

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found"
        )

    return {
        "agent": agent.name,
        "reasoning_summary": agent.get_reasoning_summary(),
    }


@router.post("/analyze-ambiguity")
async def analyze_ambiguity(request: ChatRequest):
    """
    Analyze a message for ambiguity without generating a full response.

    **Phase 2 Feature:** Useful for checking if a request needs clarification
    before sending it to the agent.

    Returns:
    - `ambiguity_score`: 0.0 (clear) to 1.0 (very ambiguous)
    - `needs_clarification`: Whether clarification is recommended
    - `suggested_questions`: Questions to ask for clarification
    """
    manager = get_agent_manager()

    agent = manager.get(request.agent or "lea")
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown agent: {request.agent}"
        )

    # Convert history to context format
    context_dicts = [
        {"content": msg.content, "role": msg.role}
        for msg in (request.history or [])
    ]

    # Analyze the request
    analysis = agent.reasoning_engine.analyze_request(
        request.message,
        context_dicts,
    )

    # Get suggested questions
    suggested_questions = []
    if analysis.get("clarification"):
        suggested_questions = [
            q.question for q in analysis["clarification"].questions[:3]
        ]

    return {
        "message": request.message,
        "ambiguity_score": analysis.get("ambiguity_score", 0.0),
        "needs_clarification": analysis.get("needs_clarification", False),
        "can_proceed": analysis.get("can_proceed", True),
        "suggested_questions": suggested_questions,
    }

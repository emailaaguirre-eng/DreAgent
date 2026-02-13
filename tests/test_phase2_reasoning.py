"""
=============================================================================
HUMMINGBIRD-LEA - Phase 2 Reasoning Tests
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Tests for Phase 2 components:
- ReAct reasoning loop
- Advanced ambiguity detection
- Confidence scoring
- Clarifying questions engine
- Reasoning transparency
=============================================================================
"""

import pytest
from datetime import datetime

# Import Phase 2 components
from core.reasoning import (
    # ReAct
    ReActLoop,
    ReActPromptBuilder,
    ReasoningStep,
    ReasoningStepType,
    ReasoningStatus,
    ActionType,
    # Ambiguity
    AdvancedAmbiguityDetector,
    AmbiguityAnalysis,
    AmbiguityType,
    AmbiguitySeverity,
    detect_ambiguity,
    # Confidence
    ConfidenceScorer,
    ConfidenceLevel,
    assess_confidence,
    # Clarifying
    ClarifyingQuestionsEngine,
    QuestionPriority,
    IntentCategory,
    generate_clarifying_questions,
    # Transparency
    ReasoningFormatter,
    TransparencyLevel,
    format_for_display,
    # Main engine
    ReasoningEngine,
)


# =============================================================================
# ReAct Loop Tests
# =============================================================================

class TestReActLoop:
    """Tests for the ReAct reasoning loop"""

    def test_basic_reasoning_flow(self):
        """Test basic thought -> action -> complete flow"""
        loop = ReActLoop(max_steps=5)

        # Add thoughts
        loop.add_thought("Analyzing the user's request")
        loop.add_thought("User wants help with something")
        loop.add_reflection("I have enough information to respond")

        # Complete
        result = loop.complete(
            response="Here is my response",
            confidence_score=0.9,
        )

        assert result.status == ReasoningStatus.COMPLETED
        assert len(result.steps) == 3
        assert result.confidence_score == 0.9

    def test_low_confidence_triggers_clarification(self):
        """Test that low confidence suggests clarification"""
        loop = ReActLoop(confidence_threshold=0.85)

        loop.add_thought("I'm not sure what the user means")

        result = loop.complete(
            response="I'm uncertain",
            confidence_score=0.5,
        )

        assert result.status == ReasoningStatus.NEEDS_CLARIFICATION

    def test_max_steps_limit(self):
        """Test that max steps prevents infinite loops"""
        loop = ReActLoop(max_steps=3)

        for i in range(5):
            if loop.should_continue():
                loop.add_thought(f"Thought {i}")

        assert loop.step_count <= 3

    def test_reasoning_summary(self):
        """Test reasoning summary generation"""
        loop = ReActLoop()
        loop.add_thought("First consideration")
        loop.add_thought("Second consideration")
        loop.add_reflection("Final conclusion")

        summary = loop.get_reasoning_summary()
        assert "Considered" in summary or "Concluded" in summary


class TestReActPromptBuilder:
    """Tests for ReAct prompt building and parsing"""

    def test_enhance_prompt(self):
        """Test prompt enhancement adds ReAct instructions"""
        base_prompt = "You are a helpful assistant."
        enhanced = ReActPromptBuilder.enhance_prompt(base_prompt)

        assert "THOUGHT" in enhanced
        assert "REFLECTION" in enhanced
        assert "ACTION" in enhanced

    def test_parse_reasoning_with_tags(self):
        """Test parsing reasoning from tagged response"""
        response = """<reasoning>
THOUGHT: The user wants information
THOUGHT: I should provide a clear answer
REFLECTION: I am confident in my response
ACTION: respond
</reasoning>

Here is my actual response to the user."""

        steps, clean_response = ReActPromptBuilder.parse_reasoning(response)

        assert len(steps) >= 3
        assert "Here is my actual response" in clean_response
        assert "<reasoning>" not in clean_response

    def test_parse_reasoning_without_tags(self):
        """Test parsing when no reasoning tags present"""
        response = "Just a normal response without reasoning tags."

        steps, clean_response = ReActPromptBuilder.parse_reasoning(response)

        assert len(steps) == 0
        assert clean_response == response


# =============================================================================
# Ambiguity Detection Tests
# =============================================================================

class TestAdvancedAmbiguityDetector:
    """Tests for advanced ambiguity detection"""

    def test_detect_vague_references(self):
        """Test detection of vague references like 'it' and 'that'"""
        detector = AdvancedAmbiguityDetector()

        # Short message with vague reference should be flagged
        analysis = detector.analyze("Send it")
        assert analysis.needs_clarification
        assert len(analysis.ambiguities) > 0

    def test_detect_incomplete_actions(self):
        """Test detection of incomplete action commands"""
        detector = AdvancedAmbiguityDetector()

        analysis = detector.analyze("delete")
        assert analysis.needs_clarification

        analysis = detector.analyze("send that")
        assert analysis.needs_clarification

    def test_detect_vague_time(self):
        """Test detection of vague time references"""
        detector = AdvancedAmbiguityDetector()

        analysis = detector.analyze("Schedule a meeting for soon")
        assert any(a.ambiguity_type == AmbiguityType.VAGUE_TIME for a in analysis.ambiguities)

    def test_detect_critical_actions(self):
        """Test detection of critical/destructive actions"""
        detector = AdvancedAmbiguityDetector()

        analysis = detector.analyze("delete everything")
        assert any(a.severity == AmbiguitySeverity.CRITICAL for a in analysis.ambiguities)

    def test_clear_message_no_ambiguity(self):
        """Test that clear messages aren't flagged"""
        detector = AdvancedAmbiguityDetector()

        analysis = detector.analyze(
            "Please schedule a meeting with John Smith for tomorrow at 2pm in the main conference room"
        )
        # Long, specific messages should have low ambiguity
        assert analysis.ambiguity_score < 0.5

    def test_backwards_compatible_function(self):
        """Test the detect_ambiguity() backwards-compatible function"""
        questions = detect_ambiguity("Send it")
        assert isinstance(questions, list)


# =============================================================================
# Confidence Scoring Tests
# =============================================================================

class TestConfidenceScorer:
    """Tests for multi-factor confidence scoring"""

    def test_high_confidence_response(self):
        """Test scoring a confident response"""
        scorer = ConfidenceScorer()

        assessment = scorer.assess_response(
            "The answer is definitely 42. This is certain.",
            question="What is the answer?",
        )

        assert assessment.level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        assert assessment.raw_score > 0.7

    def test_low_confidence_response(self):
        """Test scoring an uncertain response"""
        scorer = ConfidenceScorer()

        assessment = scorer.assess_response(
            "I'm not sure, but I think it might possibly be something. I don't know for certain.",
            question="What is the answer?",
        )

        assert assessment.level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
        assert assessment.should_express_uncertainty

    def test_medium_confidence_response(self):
        """Test scoring a moderately confident response"""
        scorer = ConfidenceScorer()

        assessment = scorer.assess_response(
            "I believe the answer is 42, based on my understanding.",
            question="What is the answer?",
        )

        assert assessment.level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]

    def test_complex_question_lowers_confidence(self):
        """Test that complex questions reduce base confidence"""
        scorer = ConfidenceScorer()

        # Simple question
        simple = scorer.assess_response(
            "The capital is Paris.",
            question="What is the capital of France?",
        )

        # Complex question
        complex_q = scorer.assess_response(
            "The answer depends on multiple factors.",
            question="Why did the economy collapse and how can we prevent it?",
        )

        # Complex questions should tend toward lower confidence
        assert complex_q.raw_score <= simple.raw_score

    def test_backwards_compatible_function(self):
        """Test the assess_confidence() backwards-compatible function"""
        level, score = assess_confidence("I am certain about this.")
        assert isinstance(level, ConfidenceLevel)
        assert 0.0 <= score <= 1.0


# =============================================================================
# Clarifying Questions Tests
# =============================================================================

class TestClarifyingQuestionsEngine:
    """Tests for clarifying questions generation"""

    def test_email_missing_recipient(self):
        """Test detection of email without recipient"""
        engine = ClarifyingQuestionsEngine()

        result = engine.generate_questions("Send an email about the project")
        questions = [q.question for q in result.questions]

        # Should ask about recipient
        assert any("who" in q.lower() or "send" in q.lower() for q in questions)

    def test_meeting_without_time(self):
        """Test detection of meeting without time"""
        engine = ClarifyingQuestionsEngine()

        result = engine.generate_questions("Schedule a meeting with the team")
        questions = [q.question for q in result.questions]

        # Should ask about time
        assert any("when" in q.lower() or "time" in q.lower() or "schedule" in q.lower() for q in questions)

    def test_intent_detection(self):
        """Test intent categorization"""
        engine = ClarifyingQuestionsEngine()

        # Communication intent
        result = engine.generate_questions("Send an email to John")
        assert result.questions[0].intent == IntentCategory.COMMUNICATION_REQUEST if result.questions else True

    def test_max_questions_limit(self):
        """Test that max questions is respected"""
        engine = ClarifyingQuestionsEngine(max_questions=2)

        result = engine.generate_questions("do something with the thing")
        assert len(result.questions) <= 2

    def test_backwards_compatible_function(self):
        """Test the generate_clarifying_questions() function"""
        questions = generate_clarifying_questions("Send it")
        assert isinstance(questions, list)


# =============================================================================
# Transparency Tests
# =============================================================================

class TestReasoningFormatter:
    """Tests for reasoning transparency formatting"""

    def test_format_reasoning_steps(self):
        """Test formatting of reasoning steps"""
        formatter = ReasoningFormatter()

        loop = ReActLoop(show_reasoning=True)
        loop.add_thought("First thought")
        loop.add_thought("Second thought")
        loop.add_reflection("Final conclusion")

        result = loop.complete("Response", confidence_score=0.9)

        transparent = formatter.format_reasoning_result(result, TransparencyLevel.DETAILED)
        assert transparent.full_reasoning is not None
        assert "thought" in transparent.full_reasoning.lower()

    def test_format_for_display_function(self):
        """Test the format_for_display helper function"""
        from core.reasoning.react import ReasoningStep

        steps = [
            ReasoningStep(step_type=ReasoningStepType.THOUGHT, content="Thinking..."),
        ]

        result = format_for_display(
            "Here is the response",
            reasoning_steps=steps,
            level=TransparencyLevel.SUMMARY,
        )

        assert "response" in result


# =============================================================================
# ReasoningEngine Integration Tests
# =============================================================================

class TestReasoningEngine:
    """Tests for the unified ReasoningEngine"""

    def test_analyze_clear_request(self):
        """Test analyzing a clear, unambiguous request"""
        engine = ReasoningEngine(agent_name="TestAgent")

        analysis = engine.analyze_request(
            "What is 2 + 2?"
        )

        assert analysis["ambiguity_score"] < 0.5
        assert not analysis["needs_clarification"]

    def test_analyze_ambiguous_request(self):
        """Test analyzing an ambiguous request"""
        engine = ReasoningEngine(agent_name="TestAgent")

        analysis = engine.analyze_request("Send it")

        assert analysis["needs_clarification"]
        assert analysis["clarification"] is not None

    def test_reasoning_flow(self):
        """Test complete reasoning flow"""
        engine = ReasoningEngine(agent_name="TestAgent")

        # Start reasoning
        engine.start_reasoning()

        # Add thoughts
        engine.add_thought("Analyzing the request")
        engine.add_thought("User wants to know something")
        engine.add_reflection("I can provide a good answer")

        # Complete
        result = engine.complete_reasoning(
            response="Here is the answer: 42",
        )

        assert result.status == ReasoningStatus.COMPLETED
        assert len(result.steps) >= 3

    def test_confidence_assessment(self):
        """Test confidence assessment through engine"""
        engine = ReasoningEngine(agent_name="TestAgent")

        assessment = engine.assess_response(
            response="I'm not entirely sure, but I think it might be correct.",
            question="Is this right?",
        )

        assert assessment.level in [ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM]


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

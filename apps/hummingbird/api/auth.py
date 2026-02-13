"""
=============================================================================
HUMMINGBIRD-LEA - Authentication API
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Login and authentication endpoints.
=============================================================================
"""

from fastapi import APIRouter, HTTPException, status

from core.utils.auth import (
    LoginRequest,
    Token,
    authenticate_user,
    create_access_token,
)

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(request: LoginRequest):
    """
    Authenticate and get an access token.
    
    Use this token in the Authorization header for protected routes:
    `Authorization: Bearer <token>`
    """
    user = authenticate_user(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = create_access_token(user.username)
    
    return token


@router.post("/logout")
async def logout():
    """
    Logout endpoint.
    
    Note: With JWT, logout is handled client-side by discarding the token.
    This endpoint exists for completeness.
    """
    return {"message": "Logged out successfully. Please discard your token."}

"""
=============================================================================
HUMMINGBIRD-LEA - Authentication
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Simple authentication system with JWT tokens.
For a private self-hosted app, this is sufficient.
=============================================================================
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import get_settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token security
security = HTTPBearer(auto_error=False)


# =============================================================================
# Models
# =============================================================================

class Token(BaseModel):
    """JWT Token response"""
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class TokenData(BaseModel):
    """Data extracted from token"""
    username: str
    exp: datetime


class User(BaseModel):
    """User model"""
    username: str
    is_active: bool = True


class LoginRequest(BaseModel):
    """Login request body"""
    username: str
    password: str


# =============================================================================
# Password Functions
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


# =============================================================================
# JWT Functions
# =============================================================================

def create_access_token(username: str) -> Token:
    """
    Create a JWT access token for a user.
    """
    settings = get_settings()
    
    expires_at = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
    
    payload = {
        "sub": username,
        "exp": expires_at,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    token = jwt.encode(
        payload, 
        settings.jwt_secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return Token(
        access_token=token,
        expires_at=expires_at
    )


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.
    Returns None if invalid.
    """
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token, 
            settings.jwt_secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        
        username = payload.get("sub")
        exp = payload.get("exp")
        
        if username is None or exp is None:
            return None
        
        return TokenData(
            username=username,
            exp=datetime.fromtimestamp(exp)
        )
        
    except JWTError:
        return None


# =============================================================================
# Authentication Functions
# =============================================================================

def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password.
    
    For Phase 1, we use simple config-based auth.
    Future: Database-backed user management.
    """
    settings = get_settings()
    
    # Simple single-user auth from config
    if username == settings.admin_username:
        # Check if password matches (plain text in config for simplicity)
        # In production, store hashed password
        if password == settings.admin_password:
            return User(username=username)
    
    return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    Dependency to get the current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"message": f"Hello {user.username}"}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if credentials is None:
        raise credentials_exception
    
    token_data = decode_token(credentials.credentials)
    
    if token_data is None:
        raise credentials_exception
    
    # Check if token is expired
    if datetime.utcnow() > token_data.exp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return User(username=token_data.username)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Dependency that returns user if authenticated, None otherwise.
    Useful for routes that work with or without auth.
    """
    if credentials is None:
        return None
    
    token_data = decode_token(credentials.credentials)
    
    if token_data is None:
        return None
    
    if datetime.utcnow() > token_data.exp:
        return None
    
    return User(username=token_data.username)

"""
Authentication endpoints.
User registration, login, token refresh.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from legal_assistant.core import get_logger
from legal_assistant.api.models.requests import LoginRequest, TokenRefreshRequest
from legal_assistant.api.models.responses import TokenResponse, UserInfoResponse
from legal_assistant.api.middleware.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    User,
    UserRepository,
)
from legal_assistant.core.exceptions import AuthenticationError

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Login with email and password.
    Returns access and refresh tokens.
    
    Args:
        request: Login credentials
    
    Returns:
        JWT tokens
    
    Raises:
        HTTPException: If authentication fails
    """
    logger.info("login_attempt", email=request.email)
    
    # Authenticate user
    user = authenticate_user(request.email, request.password)
    
    if not user:
        logger.warning("login_failed", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token({"sub": user.user_id})
    refresh_token = create_refresh_token({"sub": user.user_id})
    
    logger.info("login_successful", user_id=user.user_id, email=user.email)
    
    return TokenResponse(
        success=True,
        message="Login successful",
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=30 * 60  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: TokenRefreshRequest) -> TokenResponse:
    """
    Refresh access token using refresh token.
    
    Args:
        request: Refresh token
    
    Returns:
        New access token
    
    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        # Decode refresh token
        payload = decode_token(request.refresh_token)
        
        # Verify it's a refresh token
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type")
        
        # Get user
        user_id = payload.get("sub")
        user = UserRepository.get_by_id(user_id)
        
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Create new access token
        access_token = create_access_token({"sub": user_id})
        
        logger.info("token_refreshed", user_id=user_id)
        
        return TokenResponse(
            success=True,
            message="Token refreshed",
            access_token=access_token,
            refresh_token=request.refresh_token,  # Keep same refresh token
            token_type="bearer",
            expires_in=30 * 60
        )
    
    except AuthenticationError:
        logger.warning("token_refresh_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserInfoResponse)
async def get_current_user_info(
    user: User = Depends(get_current_user)
) -> UserInfoResponse:
    """
    Get current authenticated user information.
    
    Args:
        user: Current user from token
    
    Returns:
        User information
    """
    return UserInfoResponse(
        success=True,
        message="User information retrieved",
        user_id=user.user_id,
        email=user.email,
        created_at=user.created_at,
        is_active=user.is_active
    )

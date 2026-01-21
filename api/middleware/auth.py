"""
JWT Authentication middleware.
Handles token creation, validation, and user authentication.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from legal_assistant.core import get_settings, get_logger
from legal_assistant.core.exceptions import AuthenticationError, AuthorizationError

logger = get_logger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Bearer scheme
security = HTTPBearer()


# ============================================================================
# PASSWORD UTILITIES
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
    
    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


# ============================================================================
# JWT TOKEN UTILITIES
# ============================================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
    
    Returns:
        Encoded JWT token
    
    Example:
        >>> token = create_access_token({"sub": "user123"})
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.api.access_token_expire_minutes
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.api.secret_key.get_secret_value(),
        algorithm="HS256"
    )
    
    logger.info(
        "access_token_created",
        user_id=data.get("sub"),
        expires_at=expire.isoformat()
    )
    
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in token
    
    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    
    # Refresh tokens last longer (7 days)
    expire = datetime.utcnow() + timedelta(days=7)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.api.secret_key.get_secret_value(),
        algorithm="HS256"
    )
    
    logger.info(
        "refresh_token_created",
        user_id=data.get("sub")
    )
    
    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload
    
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.api.secret_key.get_secret_value(),
            algorithms=["HS256"]
        )
        return payload
    
    except JWTError as e:
        logger.warning("token_decode_failed", error=str(e))
        raise AuthenticationError(
            message="Could not validate credentials"
        )


# ============================================================================
# USER AUTHENTICATION
# ============================================================================

class User:
    """User model for authentication"""
    
    def __init__(
        self,
        user_id: str,
        email: str,
        hashed_password: str,
        is_active: bool = True,
        created_at: Optional[datetime] = None
    ):
        self.user_id = user_id
        self.email = email
        self.hashed_password = hashed_password
        self.is_active = is_active
        self.created_at = created_at or datetime.utcnow()
    
    def verify_password(self, password: str) -> bool:
        """Verify user password"""
        return verify_password(password, self.hashed_password)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }


# TODO: Replace with actual database queries
class UserRepository:
    """
    User repository for database operations.
    This is a mock - replace with real database implementation.
    """
    
    # Mock user storage (replace with database)
    _users: dict[str, User] = {}
    
    @classmethod
    def get_by_email(cls, email: str) -> Optional[User]:
        """Get user by email"""
        for user in cls._users.values():
            if user.email == email:
                return user
        return None
    
    @classmethod
    def get_by_id(cls, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return cls._users.get(user_id)
    
    @classmethod
    def create_user(
        cls,
        email: str,
        password: str,
        user_id: Optional[str] = None
    ) -> User:
        """Create a new user"""
        import uuid
        
        if cls.get_by_email(email):
            raise ValueError("User already exists")
        
        user_id = user_id or str(uuid.uuid4())
        hashed_password = get_password_hash(password)
        
        user = User(
            user_id=user_id,
            email=email,
            hashed_password=hashed_password
        )
        
        cls._users[user_id] = user
        logger.info("user_created", user_id=user_id, email=email)
        
        return user


def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password.
    
    Args:
        email: User email
        password: User password
    
    Returns:
        User object if authenticated, None otherwise
    """
    user = UserRepository.get_by_email(email)
    
    if not user:
        logger.warning("authentication_failed", email=email, reason="user_not_found")
        return None
    
    if not user.verify_password(password):
        logger.warning("authentication_failed", email=email, reason="invalid_password")
        return None
    
    if not user.is_active:
        logger.warning("authentication_failed", email=email, reason="inactive_user")
        return None
    
    logger.info("user_authenticated", user_id=user.user_id, email=email)
    return user


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Dependency to get current authenticated user.
    
    Args:
        credentials: HTTP Bearer credentials
    
    Returns:
        Authenticated user
    
    Raises:
        AuthenticationError: If authentication fails
    
    Example:
        >>> @app.get("/me")
        >>> async def get_me(user: User = Depends(get_current_user)):
        ...     return user.to_dict()
    """
    token = credentials.credentials
    
    try:
        payload = decode_token(token)
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check token type
    if payload.get("type") != "access":
        logger.warning("invalid_token_type", token_type=payload.get("type"))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = UserRepository.get_by_id(user_id)
    if not user:
        logger.warning("user_not_found", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active user.
    
    Args:
        current_user: Current user from get_current_user
    
    Returns:
        Active user
    
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


# Optional authentication (doesn't fail if no token)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Optional authentication dependency.
    Returns user if authenticated, None otherwise.
    
    Args:
        credentials: Optional HTTP Bearer credentials
    
    Returns:
        User if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a test user
    user = UserRepository.create_user(
        email="test@example.com",
        password="SecurePassword123!",
        user_id="test_user_123"
    )
    
    # Authenticate
    authenticated = authenticate_user("test@example.com", "SecurePassword123!")
    print(f"Authenticated: {authenticated is not None}")
    
    # Create tokens
    access_token = create_access_token({"sub": user.user_id})
    refresh_token = create_refresh_token({"sub": user.user_id})
    
    print(f"Access Token: {access_token[:50]}...")
    print(f"Refresh Token: {refresh_token[:50]}...")
    
    # Decode token
    payload = decode_token(access_token)
    print(f"Token Payload: {payload}")

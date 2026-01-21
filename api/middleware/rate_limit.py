"""
Redis-based rate limiting middleware.
Implements sliding window rate limiting.
"""

import time
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from legal_assistant.core import get_settings, get_logger
from legal_assistant.core.exceptions import RateLimitError

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis sliding window.
    
    Example:
        >>> app.add_middleware(RateLimitMiddleware)
    """
    
    def __init__(self, app, redis_client=None):
        super().__init__(app)
        self.redis_client = redis_client
        self.window = settings.redis.rate_limit_window
        self.max_requests = settings.redis.rate_limit_max_requests
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with rate limiting"""
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        # Get user identifier (IP or user_id)
        identifier = self._get_identifier(request)
        
        # Check rate limit
        try:
            await self._check_rate_limit(identifier)
        except RateLimitError as e:
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Window"] = str(self.window)
        
        return response
    
    def _get_identifier(self, request: Request) -> str:
        """Get user identifier for rate limiting"""
        # Try to get user_id from auth (if available)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fallback to IP address
        return f"ip:{request.client.host}"
    
    async def _check_rate_limit(self, identifier: str) -> None:
        """
        Check if request exceeds rate limit.
        Uses Redis sliding window algorithm.
        """
        if not self.redis_client:
            # Rate limiting disabled if no Redis
            return
        
        key = f"rate_limit:{identifier}"
        now = int(time.time())
        window_start = now - self.window
        
        try:
            # Remove old entries
            await self.redis_client.zremrangebyscore(
                key,
                0,
                window_start
            )
            
            # Count requests in window
            count = await self.redis_client.zcard(key)
            
            if count >= self.max_requests:
                # Calculate retry_after
                oldest = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1]) + self.window - now
                else:
                    retry_after = self.window
                
                logger.warning(
                    "rate_limit_exceeded",
                    identifier=identifier,
                    count=count,
                    limit=self.max_requests
                )
                
                raise RateLimitError(
                    limit=self.max_requests,
                    window_seconds=self.window,
                    retry_after=retry_after
                )
            
            # Add current request
            await self.redis_client.zadd(key, {str(now): now})
            
            # Set expiry on key
            await self.redis_client.expire(key, self.window)
            
        except Exception as e:
            logger.error(
                "rate_limit_check_failed",
                identifier=identifier,
                error=str(e)
            )
            # Fail open - allow request if rate limiting fails
            return

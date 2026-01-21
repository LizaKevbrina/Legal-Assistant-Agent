"""
Global error handling middleware.
Converts exceptions to structured JSON responses.
"""

from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from legal_assistant.core import get_logger, track_error
from legal_assistant.core.exceptions import LegalAssistantException
from legal_assistant.api.models.responses import ErrorResponse

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handler middleware.
    Catches all exceptions and converts them to structured responses.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with error handling"""
        
        try:
            response = await call_next(request)
            return response
        
        except LegalAssistantException as e:
            # Our custom exceptions - already structured
            track_error("api", e)
            
            logger.error(
                "request_failed",
                path=request.url.path,
                method=request.method,
                error_code=e.error_code,
                status_code=e.status_code,
                recoverable=e.recoverable
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
        
        except ValueError as e:
            # Validation errors
            logger.warning(
                "validation_error",
                path=request.url.path,
                error=str(e)
            )
            
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(e),
                        "code": "VALIDATION_ERROR",
                        "details": {},
                        "recoverable": False
                    }
                }
            )
        
        except Exception as e:
            # Unexpected errors - log full traceback
            track_error("api", e)
            
            logger.exception(
                "unexpected_error",
                path=request.url.path,
                method=request.method,
                error_type=type(e).__name__
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "code": "INTERNAL_ERROR",
                        "details": {},
                        "recoverable": False
                    }
                }
            )

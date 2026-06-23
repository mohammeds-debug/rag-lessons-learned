"""
Retry utility with exponential backoff for handling transient failures.
"""

import time
import logging
from functools import wraps
from typing import Callable, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 10.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: all exceptions)

    Returns:
        Decorated function that implements retry logic

    Example:
        @retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
        async def my_function():
            # Function that might fail
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries += 1

                    if retries > max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Last error: {str(e)}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** (retries - 1)), max_delay)

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {retries}/{max_retries}). "
                        f"Retrying in {delay:.2f} seconds. Error: {str(e)}"
                    )

                    time.sleep(delay)

            # Should never reach here
            raise RuntimeError(f"Unexpected retry loop exit for {func.__name__}")

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1

                    if retries > max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Last error: {str(e)}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (exponential_base ** (retries - 1)), max_delay)

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {retries}/{max_retries}). "
                        f"Retrying in {delay:.2f} seconds. Error: {str(e)}"
                    )

                    time.sleep(delay)

            # Should never reach here
            raise RuntimeError(f"Unexpected retry loop exit for {func.__name__}")

        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

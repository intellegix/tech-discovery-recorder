"""
Result pattern implementation for error handling.
Following Austin's patterns from CLAUDE.md for robust error handling.
"""

from typing import TypeVar, Generic, Union, Callable, Any, Optional
from dataclasses import dataclass


T = TypeVar('T')
E = TypeVar('E', bound=Exception)


@dataclass
class Ok(Generic[T]):
    """Success result containing a value."""
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        """Extract the success value."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return self.value

    def map(self, func: Callable[[T], Any]) -> 'Result[Any, Exception]':
        """Transform the success value."""
        try:
            return Ok(func(self.value))
        except Exception as e:
            return Err(e)

    def map_err(self, func: Callable[[Exception], Exception]) -> 'Result[T, Exception]':
        """Transform error (no-op for Ok)."""
        return self

    def and_then(self, func: Callable[[T], 'Result[Any, Exception]']) -> 'Result[Any, Exception]':
        """Chain operations that return Results."""
        try:
            return func(self.value)
        except Exception as e:
            return Err(e)


@dataclass
class Err(Generic[E]):
    """Error result containing an exception."""
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> Any:
        """Raise the contained error."""
        raise self.error

    def unwrap_or(self, default: T) -> T:
        """Return default value instead of raising."""
        return default

    def map(self, func: Callable[[Any], Any]) -> 'Result[Any, E]':
        """Transform success value (no-op for Err)."""
        return self

    def map_err(self, func: Callable[[E], Exception]) -> 'Result[Any, Exception]':
        """Transform the error."""
        return Err(func(self.error))

    def and_then(self, func: Callable[[Any], 'Result[Any, Exception]']) -> 'Result[Any, E]':
        """Chain operations (no-op for Err)."""
        return self


# Type alias for convenience
Result = Union[Ok[T], Err[E]]


def wrap_result(func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """
    Decorator to wrap functions that might raise exceptions into Result pattern.

    Usage:
        @wrap_result
        def divide(a: int, b: int) -> float:
            return a / b

        result = divide(10, 2)  # Returns Ok(5.0)
        result = divide(10, 0)  # Returns Err(ZeroDivisionError)
    """
    def wrapper(*args, **kwargs) -> Result[T, Exception]:
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            return Err(e)
    return wrapper


def from_optional(value: Optional[T], error_msg: str = "Value is None") -> Result[T, ValueError]:
    """Convert Optional to Result."""
    if value is None:
        return Err(ValueError(error_msg))
    return Ok(value)


def collect_results(results: list[Result[T, Exception]]) -> Result[list[T], Exception]:
    """
    Collect multiple Results into a single Result containing a list.
    Returns the first error encountered, or Ok with all values.
    """
    values = []
    for result in results:
        if result.is_err():
            return result  # Return first error
        values.append(result.unwrap())
    return Ok(values)


# Common error types for the application
class RecordingError(Exception):
    """Base exception for recording-related errors."""
    pass


class ConsentError(RecordingError):
    """Error related to consent validation."""
    pass


class ProcessingError(RecordingError):
    """Error during AI processing."""
    pass


class StorageError(RecordingError):
    """Error in file storage operations."""
    pass


class AuthenticationError(Exception):
    """Authentication/authorization errors."""
    pass


class ValidationError(Exception):
    """Data validation errors."""
    pass


class ExternalServiceError(Exception):
    """Error from external services (Claude API, S3, etc.)."""
    pass


# Utility functions for common Result patterns
def ensure(condition: bool, error: Exception) -> Result[None, Exception]:
    """Ensure a condition is true, return error otherwise."""
    if condition:
        return Ok(None)
    return Err(error)


def try_parse_int(value: str) -> Result[int, ValueError]:
    """Safely parse integer from string."""
    try:
        return Ok(int(value))
    except ValueError as e:
        return Err(e)


def try_parse_float(value: str) -> Result[float, ValueError]:
    """Safely parse float from string."""
    try:
        return Ok(float(value))
    except ValueError as e:
        return Err(e)
from abc import ABC, abstractmethod
from typing import Any


class TaskProcessor(ABC):
    """Base class for task-specific circuit result processing"""

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Process input data for a specific task

        Args:
            *args: Positional arguments (implementation-specific)
            **kwargs: Keyword arguments (implementation-specific)

        Returns:
            Task-specific result
        """

    @abstractmethod
    def validate_result(self, result: Any) -> bool:
        """
        Validate if the result meets task requirements

        Args:
            result: The processed result

        Returns:
            True if valid, False otherwise
        """

from abc import ABC, abstractmethod
from typing import Any, Dict


class TaskProcessor(ABC):
    """Base class for task-specific circuit result processing"""

    @abstractmethod
    def process(self, counts: Dict[str, int], **kwargs) -> Any:
        """
        Process measurement results for a specific task

        Args:
            counts: Dictionary mapping bitstrings to measurement counts
            **kwargs: Task-specific parameters

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

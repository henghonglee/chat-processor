"""
Abstract Base Class for Query Steps

This module provides the base class that all query processing steps inherit from.
It standardizes the interface and provides common functionality like logging.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseQueryStep(ABC):
    """Abstract base class for all query processing steps."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the query step.

        Args:
            config: Configuration dictionary for the step
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup()

    def setup(self):
        """
        Setup method called after initialization.
        Override this method to perform step-specific initialization.
        """
        # Default implementation - no setup required

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Main processing method that each step must implement.

        Args:
            input_data: The input data for this step

        Returns:
            The processed output data
        """
        pass

    def execute(self, input_data: Any) -> Any:
        """
        Execute the step with timing and error handling.

        Args:
            input_data: The input data for this step

        Returns:
            The processed output data
        """
        step_name = self.__class__.__name__
        self.logger.info(f"Starting {step_name}")

        start_time = time.time()

        try:
            result = self.process(input_data)
            execution_time = time.time() - start_time

            self.logger.info(
                f"{step_name} completed successfully in {execution_time:.2f}s"
            )

            # Log intermediate results if verbose logging is enabled
            if self.logger.isEnabledFor(logging.DEBUG):
                self._log_step_result(result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"{step_name} failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

    def _log_step_result(self, result: Any):
        """
        Log the result of the step processing.
        Override this method to provide step-specific logging.

        Args:
            result: The result to log
        """
        self.logger.debug(f"Step result type: {type(result).__name__}")

    def close(self):
        """
        Cleanup method called when the step is no longer needed.
        Override this method to perform step-specific cleanup.
        """
        # Default implementation - no cleanup required

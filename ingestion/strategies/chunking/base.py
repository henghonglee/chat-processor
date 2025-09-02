"""
Base chunking strategy interface.

Defines the abstract base class for all chunking strategies that determine
when to start new chunks based on different criteria.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

# WhatsApp timestamp pattern: [DD/MM/YY, HH:MM:SS AM/PM]
WHATSAPP_TIMESTAMP_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{1,2}:\d{1,2}) (AM|PM)\]"
)

# Unix timestamp pattern: [1234567890]
UNIX_TIMESTAMP_RE = re.compile(r"^\[(\d{10,})\]")


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def should_start_new_chunk(
        self, current_line: str, current_index: int, chunk_context: dict
    ) -> bool:
        """
        Determine if a new chunk should be started at this line.

        Args:
            current_line: The current line being processed
            current_index: Index of the current line
            chunk_context: Dictionary containing context about the current chunk
                          (e.g., start_time, message_count, etc.)

        Returns:
            True if a new chunk should be started
        """
        pass

    def initialize_chunk_context(self, first_line: str, first_index: int) -> dict:
        """
        Initialize context for a new chunk.

        Args:
            first_line: First line of the new chunk
            first_index: Index of the first line

        Returns:
            Dictionary with initial chunk context
        """
        timestamp = self._parse_timestamp(first_line)
        return {
            "start_index": first_index,
            "start_time": timestamp,
            "message_count": (
                1 if first_line.strip() and first_line.startswith("[") else 0
            ),
            "last_timestamp": timestamp,
        }

    def update_chunk_context(self, context: dict, line: str, line_index: int) -> None:
        """
        Update chunk context with information from the current line.

        Args:
            context: Current chunk context to update
            line: Current line being processed
            line_index: Index of current line
        """
        if line.strip() and line.startswith("["):
            context["message_count"] += 1
            timestamp = self._parse_timestamp(line)
            if timestamp:
                context["last_timestamp"] = timestamp

    def _parse_timestamp(self, line: str) -> Optional[datetime]:
        """Parse timestamp from a message line."""
        # Try Unix timestamp first (most common in processed data)
        unix_result = self._parse_unix_timestamp(line)
        if unix_result:
            return unix_result

        # Try WhatsApp format (legacy)
        whatsapp_result = self._parse_whatsapp_timestamp(line)
        if whatsapp_result:
            return whatsapp_result

        return None

    def _parse_unix_timestamp(self, line: str) -> Optional[datetime]:
        """Parse Unix timestamp from a message line."""
        clean_line = line.strip()
        match = UNIX_TIMESTAMP_RE.match(clean_line)
        if not match:
            return None

        try:
            unix_timestamp = int(match.group(1))
            return datetime.fromtimestamp(unix_timestamp)
        except (ValueError, OverflowError, OSError):
            return None

    def _parse_whatsapp_timestamp(self, line: str) -> Optional[datetime]:
        """Parse WhatsApp timestamp from a message line (legacy format)."""
        # Clean invisible characters like right-to-left marks and narrow no-break spaces
        clean_line = (
            line.strip().lstrip("\u200e\u200f\u202d\u202e").replace("\u202f", " ")
        )
        match = WHATSAPP_TIMESTAMP_RE.match(clean_line)
        if not match:
            return None

        try:
            date_str, time_str, ampm = match.groups()

            # Parse date
            day, month, year = map(int, date_str.split("/"))
            year += 2000  # Convert YY to YYYY

            # Parse time
            hour, minute, second = map(int, time_str.split(":"))

            # Convert to 24-hour format
            if ampm.upper() == "PM" and hour != 12:
                hour += 12
            elif ampm.upper() == "AM" and hour == 12:
                hour = 0

            return datetime(year, month, day, hour, minute, second)
        except (ValueError, IndexError):
            return None

"""
Base parser interface for all chat format parsers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class Message:
    """Standardized message representation."""

    def __init__(
        self,
        timestamp: datetime,
        sender: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict] = None,
    ):
        self.timestamp = timestamp
        self.sender = sender
        self.content = content
        self.message_type = (
            message_type  # text, image, video, audio, file, reaction, etc.
        )
        self.metadata = metadata or {}

    def to_standardized_format(self) -> str:
        """Convert to standardized text format using Unix timestamps."""
        unix_timestamp = int(self.timestamp.timestamp())

        if self.message_type == "text":
            return f"[{unix_timestamp}] {self.sender}: {self.content}"
        elif self.message_type == "image":
            return f"[{unix_timestamp}] {self.sender}: <image: {self.content}>"
        elif self.message_type == "video":
            return f"[{unix_timestamp}] {self.sender}: <video: {self.content}>"
        elif self.message_type == "audio":
            return f"[{unix_timestamp}] {self.sender}: <audio: {self.content}>"
        elif self.message_type == "file":
            return f"[{unix_timestamp}] {self.sender}: <file: {self.content}>"
        elif self.message_type == "reaction":
            return f"[{unix_timestamp}] {self.content}"
        else:
            return f"[{unix_timestamp}] {self.sender}: <{self.message_type}: {self.content}>"

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "content": self.content,
            "message_type": self.message_type,
        }


class ChatContext:
    """Container for chat context and metadata."""

    def __init__(self, chat_name: str, context_text: str = ""):
        self.chat_name = chat_name
        self.context_text = context_text
        self.participants: List[str] = []
        self.message_count: int = 0
        self.date_range: Optional[Tuple[datetime, datetime]] = None
        self.source_format: str = "unknown"

    def to_dict(self):
        return {
            "chat_name": self.chat_name,
            "context_text": self.context_text,
            "participants": self.participants,
            "message_count": self.message_count,
            "date_range": (
                (self.date_range[0].isoformat(), self.date_range[1].isoformat())
                if self.date_range
                else None
            ),
            "source_format": self.source_format,
        }


class BaseParser(ABC):
    """Abstract base class for all chat parsers."""

    @abstractmethod
    def parse_file(
        self, file_path: str, context_path: Optional[str] = None
    ) -> Tuple[List[Message], ChatContext]:
        """
        Parse a chat export file and return a list of Message objects and ChatContext.
        """
        pass

    def get_supported_extensions(self) -> List[str]:
        """
        Get the file extensions supported by this parser.

        Returns:
            List of supported file extensions (e.g., ['.txt', '.json'])
        """
        return []

    def get_parser_name(self) -> str:
        """
        Get the name of this parser.

        Returns:
            String name of the parser
        """
        return self.__class__.__name__

"""
WhatsApp chat parser.

Parses WhatsApp chat exports (.txt format) into standardized intermediate representation.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .base import BaseParser


class WhatsAppParser(BaseParser):
    """Parser for WhatsApp chat exports."""

    def parse_file(self, file_path: str, context_path: Optional[str] = None):
        """
        Parse WhatsApp chat file and return messages and context.

        Args:
            file_path: Path to WhatsApp chat export file
            context_path: Optional path to context file (not used for WhatsApp)

        Returns:
            Tuple of (messages, context)
        """
        # Import required classes and functions from the parent module
        import importlib.util
        import sys

        # Import from the steps module
        steps_path = Path(__file__).parent.parent.parent / "steps" / "a_parsing.py"
        spec = importlib.util.spec_from_file_location("parsing_steps", steps_path)
        parsing_module = importlib.util.module_from_spec(spec)
        sys.modules["parsing_steps"] = parsing_module
        spec.loader.exec_module(parsing_module)

        # Import chunking for timestamp parsing
        chunking_path = Path(__file__).parent.parent.parent / "steps" / "b_chunking.py"
        spec = importlib.util.spec_from_file_location("chunking", chunking_path)
        chunking_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunking_module)
        parse_timestamp = chunking_module.parse_timestamp

        # Use the original parsing logic
        messages, context = self._parse_whatsapp_file(file_path, parse_timestamp)

        # Return the parsed data
        return messages, context

    def _parse_whatsapp_file(self, file_path: str, parse_timestamp_func):
        """Parse WhatsApp chat file using the original logic."""
        # Import required classes
        import importlib.util

        steps_path = Path(__file__).parent.parent.parent / "steps" / "a_parsing.py"
        spec = importlib.util.spec_from_file_location("parsing_steps", steps_path)
        parsing_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parsing_module)

        Message = parsing_module.Message
        ChatContext = parsing_module.ChatContext

        messages = []
        chat_name = Path(file_path).parent.name

        context = ChatContext(chat_name, "")
        context.source_format = "WhatsApp"

        participants = set()

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse timestamp format (supports both Unix and WhatsApp formats)
                timestamp = parse_timestamp_func(line)
                if timestamp:
                    # Extract sender and content
                    match = re.match(r"^\[.*?\] ([^:]+): (.*)$", line)
                    if match:
                        sender, content = match.groups()
                        participants.add(sender)

                        # Determine message type
                        message_type = "text"
                        if "image omitted" in content.lower():
                            message_type = "image"
                            content = "image omitted"
                        elif "video omitted" in content.lower():
                            message_type = "video"
                            content = "video omitted"
                        elif "audio omitted" in content.lower():
                            message_type = "audio"
                            content = "audio omitted"
                        elif "document omitted" in content.lower():
                            message_type = "file"
                            content = "document omitted"

                        messages.append(
                            Message(timestamp, sender, content, message_type)
                        )

        context.participants = list(participants)
        context.message_count = len(messages)

        if messages:
            context.date_range = (messages[0].timestamp, messages[-1].timestamp)

        return messages, context

    def _save_as_jsonl(self, messages, context, output_path: Path):
        """Save messages and context as JSONL."""
        import json

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Write context as first line
            context_dict = {
                "type": "context",
                "chat_name": context.chat_name,
                "source_format": context.source_format,
                "participants": context.participants,
                "message_count": context.message_count,
                "date_range": (
                    [
                        (
                            context.date_range[0].isoformat()
                            if context.date_range
                            else None
                        ),
                        (
                            context.date_range[1].isoformat()
                            if context.date_range
                            else None
                        ),
                    ]
                    if context.date_range
                    else None
                ),
            }
            f.write(json.dumps(context_dict, ensure_ascii=False) + "\n")

            # Write messages
            for message in messages:
                message_dict = {
                    "type": "message",
                    "timestamp": message.timestamp.isoformat(),
                    "sender": message.sender,
                    "content": message.content,
                    "message_type": message.message_type,
                }
                f.write(json.dumps(message_dict, ensure_ascii=False) + "\n")

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions for WhatsApp."""
        return [".txt"]

    def get_parser_name(self) -> str:
        """Get parser name."""
        return "WhatsApp"

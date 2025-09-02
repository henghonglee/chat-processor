"""
Telegram chat parser.

Parses Telegram chat exports (.json format) into standardized intermediate representation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .base import BaseParser, Message, ChatContext


class TelegramParser(BaseParser):
    """Parser for Telegram chat exports."""

    def parse_file(self, file_path: str, context_path: Optional[str] = None) -> Tuple[List[Message], ChatContext]:
        """
        Parse Telegram chat file and return messages and context.

        Args:
            file_path: Path to Telegram JSON export file
            context_path: Optional path to context file (not used for Telegram)

        Returns:
            Tuple of (messages, context)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract chat context
        chat_name = data.get("name", Path(file_path).parent.name)
        chat_type = data.get("type", "unknown")
        chat_id = data.get("id")

        # Extract participants from messages
        participants = set()
        messages = []

        for msg_data in data.get("messages", []):
            # Skip non-message types we don't want to process
            if msg_data.get("type") not in ["message", "service"]:
                continue

            # Parse timestamp
            timestamp_str = msg_data.get("date")
            if not timestamp_str:
                continue

            try:
                # Handle both ISO format and unix timestamp
                if timestamp_str.isdigit():
                    timestamp = datetime.fromtimestamp(int(timestamp_str))
                else:
                    # Handle ISO format: 2025-05-06T18:58:54
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            # Extract sender information
            sender = self._extract_sender(msg_data)
            if sender:
                participants.add(sender)

            # Parse message content based on type
            if msg_data.get("type") == "message":
                message = self._parse_regular_message(msg_data, timestamp, sender)
            elif msg_data.get("type") == "service":
                message = self._parse_service_message(msg_data, timestamp, sender)
            else:
                continue

            if message:
                messages.append(message)

        # Build context
        context = ChatContext(chat_name=chat_name)
        context.participants = list(participants)
        context.message_count = len(messages)
        context.source_format = "telegram"
        
        # Add telegram-specific fields directly to context
        context.chat_type = chat_type
        context.chat_id = chat_id

        if messages:
            # Convert first and last message timestamps to datetime objects
            start_time = datetime.fromisoformat(messages[0]["timestamp"])
            end_time = datetime.fromisoformat(messages[-1]["timestamp"])
            context.date_range = (start_time, end_time)

        # Convert dictionary messages to Message objects
        message_objects = []
        for msg_dict in messages:
            timestamp = datetime.fromisoformat(msg_dict["timestamp"])
            message_obj = Message(
                timestamp=timestamp,
                sender=msg_dict["sender"],
                content=msg_dict["content"],
                message_type=msg_dict["message_type"],
                metadata={
                    "id": msg_dict.get("id"),
                    "raw_data": msg_dict.get("raw_data", {})
                }
            )
            message_objects.append(message_obj)

        return message_objects, context

    def _extract_sender(self, msg_data: Dict[str, Any]) -> Optional[str]:
        """Extract sender name from message data."""
        # For service messages, use actor
        if msg_data.get("type") == "service" and msg_data.get("actor"):
            return msg_data["actor"]

        # For regular messages, use from
        sender_info = msg_data.get("from")
        if isinstance(sender_info, str):
            return sender_info
        elif isinstance(sender_info, dict):
            # Handle user ID format
            if "user" in sender_info:
                return sender_info["user"]
            return sender_info.get("name", "Unknown")
        elif msg_data.get("from_id"):
            # Handle from_id format like "user687804705"
            from_id = msg_data["from_id"]
            if isinstance(from_id, str) and from_id.startswith("user"):
                # Try to get actual name from 'from' field
                return msg_data.get("from", from_id)
            return str(from_id)

        return None

    def _parse_regular_message(self, msg_data: Dict[str, Any], timestamp: datetime, sender: str) -> Optional[Dict[str, Any]]:
        """Parse a regular message."""
        # Extract text content
        text_content = msg_data.get("text", "")
        content = self._extract_text_content(text_content)

        # Determine message type and enhance content
        message_type = "text"

        # Handle media attachments
        if msg_data.get("photo"):
            message_type = "image"
            content = f"[photo] {content}".strip()
        elif msg_data.get("video_file"):
            message_type = "video"
            content = f"[video] {content}".strip()
        elif msg_data.get("voice_message"):
            message_type = "audio"
            content = f"[voice] {content}".strip()
        elif msg_data.get("audio_file"):
            message_type = "audio"
            content = f"[audio] {content}".strip()
        elif msg_data.get("file"):
            message_type = "file"
            content = f"[file] {content}".strip()
        elif msg_data.get("sticker_emoji"):
            message_type = "sticker"
            content = f"[sticker: {msg_data['sticker_emoji']}]"
        elif msg_data.get("poll"):
            message_type = "poll"
            poll = msg_data["poll"]
            question = poll.get("question", "poll")
            content = f"[poll: {question}]"

        # Handle forwarded messages
        if msg_data.get("forwarded_from"):
            content = f"[forwarded from {msg_data['forwarded_from']}] {content}"

        # Handle replies
        if msg_data.get("reply_to_message_id"):
            content = f"[reply to {msg_data['reply_to_message_id']}] {content}"

        # Skip empty messages
        if not content.strip():
            return None

        return {
            "id": msg_data.get("id"),
            "timestamp": timestamp.isoformat(),
            "sender": sender,
            "content": content.strip(),
            "message_type": message_type,
            "raw_data": msg_data
        }

    def _parse_service_message(self, msg_data: Dict[str, Any], timestamp: datetime, sender: str) -> Optional[Dict[str, Any]]:
        """Parse a service message."""
        action = msg_data.get("action", "")
        actor = msg_data.get("actor", sender)

        # Build service message content
        content_parts = []

        if action == "create_group":
            title = msg_data.get("title", "group")
            members = msg_data.get("members", [])
            content_parts.append(f"{actor} created group '{title}'")
            if members:
                content_parts.append(f"with members: {', '.join(members)}")
        elif action == "join_group":
            content_parts.append(f"{actor} joined the group")
        elif action == "leave_group":
            content_parts.append(f"{actor} left the group")
        elif action == "add_members":
            members = msg_data.get("members", [])
            content_parts.append(f"{actor} added {', '.join(members)}")
        elif action == "remove_members":
            members = msg_data.get("members", [])
            content_parts.append(f"{actor} removed {', '.join(members)}")
        else:
            content_parts.append(f"{actor} {action}")

        content = " ".join(content_parts)

        return {
            "id": msg_data.get("id"),
            "timestamp": timestamp.isoformat(),
            "sender": sender,
            "content": content,
            "message_type": "service",
            "raw_data": msg_data
        }

    def _extract_text_content(self, text_content: Any) -> str:
        """Extract text content from various text formats."""
        if isinstance(text_content, str):
            return text_content
        elif isinstance(text_content, list):
            # Handle text entities
            content_parts = []
            for part in text_content:
                if isinstance(part, dict):
                    if part.get("type") == "link":
                        content_parts.append(part.get("text", ""))
                    else:
                        content_parts.append(part.get("text", ""))
                else:
                    content_parts.append(str(part))
            return "".join(content_parts)
        elif text_content is None:
            return ""
        else:
            return str(text_content)

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions for Telegram."""
        return [".json"]

    def get_parser_name(self) -> str:
        """Get parser name."""
        return "Telegram"

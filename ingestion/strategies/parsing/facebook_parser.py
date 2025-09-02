"""
Facebook Messenger chat parser.

Parses Facebook Messenger chat exports (.json format) into standardized intermediate representation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import BaseParser

# Facebook reaction emoji mapping
FACEBOOK_REACTION_MAP = {
    "👍": "[like]",
    "👎": "[dislike]",
    "😍": "[love]",
    "😢": "[sad]",
    "😮": "[wow]",
    "😠": "[angry]",
    "😆": "[laugh]",
    "❤️": "[heart]",
    "💯": "[100]",
    "🔥": "[fire]",
    "👏": "[clap]",
    "💔": "[broken_heart]",
    "😭": "[tear]",
    "😊": "[smile]",
    "🤣": "[rofl]",
    "🙏": "[pray]",
    "💪": "[strong]",
    "🤔": "[thinking]",
    "🎉": "[party]",
    "💰": "[money]",
    # Common Facebook-specific encodings
    "ð\x9f\x91\x8d": "[like]",  # 👍
    "ð\x9f\x91\x8e": "[dislike]",  # 👎
    "ð\x9f\x98\x8d": "[love]",  # 😍
    "ð\x9f\x98\xa2": "[sad]",  # 😢
    "ð\x9f\x98\xae": "[wow]",  # 😮
    "ð\x9f\x98\xa0": "[angry]",  # 😠
    "ð\x9f\x98\x86": "[laugh]",  # 😆
}


class FacebookParser(BaseParser):
    """Parser for Facebook Messenger chat exports."""

    def parse_file(self, file_path: str, context_path: Optional[str] = None):
        """
        Parse Facebook Messenger chat file and return messages and context.

        Args:
            file_path: Path to Facebook Messenger JSON export file
            context_path: Optional path to context file (not used for Facebook)

        Returns:
            Tuple of (messages, context)
        """
        # Parse the Facebook file
        messages, context = self._parse_facebook_file(file_path)

        # Return the parsed data
        return messages, context

    @staticmethod
    def _normalize_reaction(reaction_unicode: str) -> str:
        """Convert Unicode reaction to readable format."""
        # First try direct mapping
        if reaction_unicode in FACEBOOK_REACTION_MAP:
            return FACEBOOK_REACTION_MAP[reaction_unicode]

        # Try to decode if it appears to be UTF-8 encoded
        try:
            # If the string contains escape sequences, try to decode them
            if "\\x" in repr(reaction_unicode):
                # Convert to bytes and decode as UTF-8
                decoded = reaction_unicode.encode("latin1").decode("utf-8")
                if decoded in FACEBOOK_REACTION_MAP:
                    return FACEBOOK_REACTION_MAP[decoded]
                return f"[{decoded}]"
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        # Fallback to original with brackets
        return f"[{reaction_unicode}]"

    def _parse_facebook_file(self, file_path: str):
        """Parse Facebook Messenger JSON file using the original logic."""
        # Import required classes
        import importlib.util

        steps_path = Path(__file__).parent.parent.parent / "steps" / "a_parsing.py"
        spec = importlib.util.spec_from_file_location("parsing_steps", steps_path)
        parsing_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parsing_module)

        Message = parsing_module.Message
        ChatContext = parsing_module.ChatContext

        messages = []
        message_timestamp_map = (
            {}
        )  # Store unix_timestamp -> Message for reaction linking
        recent_messages = []  # Store recent messages for "previous message" lookup

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chat_name = Path(file_path).parent.name

        context = ChatContext(chat_name, "")
        context.source_format = "Facebook Messenger"

        # Extract participants
        participants = [p.get("name", "Unknown") for p in data.get("participants", [])]
        context.participants = participants

        # Parse messages (Facebook stores them in reverse chronological order)
        fb_messages = data.get("messages", [])
        fb_messages.reverse()  # Convert to chronological order

        # First pass: Process all main messages and build timestamp map
        for msg in fb_messages:
            timestamp = datetime.fromtimestamp(msg["timestamp_ms"] / 1000)
            unix_timestamp = int(timestamp.timestamp())
            sender = msg.get("sender_name", "Unknown")

            # Handle different message types
            if "content" in msg:
                content = msg["content"]
                message_type = "text"

                # Skip standalone reaction messages since we handle reactions via the reactions array
                if "reacted" in content and "to your message" in content:
                    continue

            elif "photos" in msg:
                message_type = "image"
                content = f"{len(msg['photos'])} photo(s)"

            elif "videos" in msg:
                message_type = "video"
                content = f"{len(msg['videos'])} video(s)"

            elif "audio_files" in msg:
                message_type = "audio"
                content = f"{len(msg['audio_files'])} audio file(s)"

            elif "files" in msg:
                message_type = "file"
                content = f"{len(msg['files'])} file(s)"

            elif "gifs" in msg:
                message_type = "image"
                content = "GIF"

            elif "sticker" in msg:
                message_type = "text"
                content = "sticker"

            else:
                # Skip messages without recognizable content
                continue

            # Create the main message
            main_message = Message(timestamp, sender, content, message_type)
            messages.append(main_message)
            recent_messages.append(main_message)
            message_timestamp_map[unix_timestamp] = main_message

            # Keep only recent messages (last 20) to avoid memory issues
            if len(recent_messages) > 20:
                recent_messages = recent_messages[-20:]

            # Handle reactions attached to this message
            if "reactions" in msg:
                for reaction in msg["reactions"]:
                    reaction_unicode = reaction.get("reaction", "")
                    reactor = reaction.get("actor", "Unknown")
                    reaction_readable = self._normalize_reaction(reaction_unicode)

                    # Create reaction message with timestamp +1 second from original message
                    reaction_timestamp = datetime.fromtimestamp(
                        timestamp.timestamp() + 1
                    )
                    reaction_msg = Message(
                        reaction_timestamp,
                        reactor,
                        f"{reactor} reacted {reaction_readable} to {unix_timestamp}",
                        "reaction",
                    )
                    messages.append(reaction_msg)
                    recent_messages.append(reaction_msg)

        context.message_count = len(messages)

        if messages:
            context.date_range = (messages[0].timestamp, messages[-1].timestamp)

        return messages, context

    def _save_as_jsonl(self, messages, context, output_path: Path):
        """Save messages and context as JSONL."""
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
        """Get supported file extensions for Facebook."""
        return [".json"]

    def get_parser_name(self) -> str:
        """Get parser name."""
        return "Facebook Messenger"

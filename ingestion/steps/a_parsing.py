#!/usr/bin/env python3
"""
Parsing Module for Intermediate Representation Creation

Handles multiple chat formats:
- WhatsApp exports (.txt format)
- Facebook Messenger exports (.json format)
- Telegram exports (.json format)

Converts them into a standardized intermediate representation and saves as JSONL.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

# Import parsing strategies
from ingestion.strategies.parsing import (
    ChatContext,
    FacebookParser,
    Message,
    TelegramParser,
    WhatsAppParser,
)


class ChatParser:
    """Lightweight chat parser that delegates parsing to strategy classes."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.chats_dir = Path("./chats")
        self.output_dir = Path("./chats-parsed")

        # Initialize parsers
        self.parsers = {
            "whatsapp": WhatsAppParser(),
            "facebook": FacebookParser(),
            "telegram": TelegramParser(),
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def process_single_chat(self, chat_name: str) -> None:
        """Process a single chat by determining the parser type and delegating to it."""
        self.logger.info(f"🎯 Processing chat: {chat_name}")

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Determine parser type and chat folder
        parser_type, folder_name = self._determine_parser_and_folder(chat_name)

        if not parser_type:
            self.logger.error(f"❌ Could not determine parser for: {chat_name}")
            return

        # Get the appropriate parser
        parser = self.parsers[parser_type]

        # Find the chat files
        chat_folder = self.chats_dir / parser_type / folder_name
        input_file, context_file = self._find_chat_files(chat_folder, parser_type)

        if not input_file:
            self.logger.error(f"❌ No valid chat file found in: {chat_folder}")
            return

        self.logger.info(f"📄 Processing {parser_type} chat: {folder_name}")

        try:
            print(f"Input file: {input_file}")
            print(f"Context file: {context_file}")
            print(parser)
            # Parse using the appropriate strategy
            messages, context = parser.parse_file(
                str(input_file),
                str(context_file) if context_file and context_file.exists() else None,
            )

            # Save as JSONL
            output_path = self.output_dir / f"{chat_name}.jsonl"

            self._save_to_jsonl(messages, context, output_path)

            self.logger.info(f"✅ Completed: {chat_name}")

        except Exception as e:
            self.logger.error(f"❌ Error processing {chat_name}: {e}")

    def _determine_parser_and_folder(self, chat_name: str) -> Tuple[Optional[str], str]:
        """Determine which parser to use and the folder name based on chat_name."""
        # Check if chat_name has a prefix
        if chat_name.startswith("facebook-"):
            return "facebook", chat_name[9:]
        elif chat_name.startswith("whatsapp-"):
            return "whatsapp", chat_name[9:]
        elif chat_name.startswith("telegram-"):
            return "telegram", chat_name[9:]

        # Try to infer from folder structure
        for parser_type in ["facebook", "whatsapp", "telegram"]:
            chat_folder = self.chats_dir / parser_type / chat_name
            if chat_folder.exists():
                return parser_type, chat_name

        return None, chat_name

    def _find_chat_files(
        self, chat_folder: Path, parser_type: str
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Find the main chat file and context file for the given parser type."""
        if not chat_folder.exists():
            return None, None

        context_file = chat_folder / "context.txt"

        if parser_type == "whatsapp":
            chat_file = chat_folder / "chat.txt"
            return chat_file if chat_file.exists() else None, context_file

        elif parser_type == "facebook":
            # Look for any JSON file (prefer message_*.json)
            json_files = list(chat_folder.glob("message_*.json"))
            if not json_files:
                json_files = list(chat_folder.glob("*.json"))
            return json_files[0] if json_files else None, context_file

        elif parser_type == "telegram":
            # Look for result.json or any JSON file
            result_file = chat_folder / "result.json"
            if result_file.exists():
                return result_file, context_file
            json_files = list(chat_folder.glob("*.json"))
            return json_files[0] if json_files else None, context_file

        return None, None

    def _save_to_jsonl(
        self, messages: List[Message], context: ChatContext, output_path: Path
    ) -> None:
        """Save messages and context in JSONL format."""

        with open(output_path, "w", encoding="utf-8") as f:
            # Write context as first line
            context_record = {"type": "context", "data": context.to_dict()}
            f.write(json.dumps(context_record, ensure_ascii=False) + "\n")

            # Write each message as a separate line
            for msg in messages:
                message_record = {
                    "type": "message",
                    "timestamp": int(msg.timestamp.timestamp()),
                    "sender": msg.sender,
                    "content": msg.content,
                    "message_type": msg.message_type,
                    "metadata": msg.metadata,
                }
                f.write(json.dumps(message_record, ensure_ascii=False) + "\n")

        self.logger.info(f"💾 Saved JSONL: {output_path}")
        self.logger.info(f"📊 {len(messages)} messages converted to JSONL format")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Chat Parser - Creates Intermediate Representation in JSONL format"
    )
    parser.add_argument(
        "chat_name",
        help="Specific chat to process (e.g., facebook-soldegen, whatsapp-jiawei, telegram-group)",
    )

    args = parser.parse_args()
    chat_parser = ChatParser()

    chat_parser.process_single_chat(args.chat_name)


if __name__ == "__main__":
    main()

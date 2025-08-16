#!/usr/bin/env python3
"""
Universal Chat Parser and Processing Engine

Handles multiple chat formats:
- WhatsApp exports (.txt format)
- Facebook Messenger exports (.json format)

Converts them into a standardized intermediate representation and processes them
with time-based chunking and context preservation.
"""

import json
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from chunk import create_time_based_chunks, parse_timestamp

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
    "ð\x9f\x91\x8d": "[like]",        # 👍
    "ð\x9f\x91\x8e": "[dislike]",     # 👎
    "ð\x9f\x98\x8d": "[love]",        # 😍
    "ð\x9f\x98\xa2": "[sad]",         # 😢
    "ð\x9f\x98\xae": "[wow]",         # 😮
    "ð\x9f\x98\xa0": "[angry]",       # 😠
    "ð\x9f\x98\x86": "[laugh]",       # 😆
    "â\x9d¤": "[heart]",              # ❤️
    "â\x9d¤ï¸\x8f": "[heart]",         # ❤️ (with variant selector)
    "ð\x9f\x92¯": "[100]",           # 💯
    "ð\x9f\x94¥": "[fire]",          # 🔥
    "ð\x9f\x91\x8f": "[clap]",        # 👏
    "ð\x9f\x92\x94": "[broken_heart]", # 💔
    "ð\x9f\x98\xad": "[tear]",        # 😭
    "ð\x9f\x98\x8a": "[smile]",       # 😊
    "ð\x9f¤£": "[rofl]",              # 🤣
    "ð\x9f\x99\x8f": "[pray]",        # 🙏
    "ð\x9f\x92ª": "[strong]",         # 💪
    "ð\x9f¤\x94": "[thinking]",       # 🤔
    "ð\x9f\x8e\x89": "[party]",       # 🎉
    "ð\x9f\x92°": "[money]",          # 💰
    # Additional Facebook-encoded reactions found in data
    "ð\x9f\x98\x82": "[joy]",         # 😂
    "ð\x9f\x9a\x80": "[rocket]",      # 🚀
    "â\x98\xa0ï¸\x8f": "[skull]",      # ☠️
    "ð\x9f\x98\x85": "[sweat_smile]", # 😅
    "ð\x9f\x92\x80": "[skull]",       # 💀
    "ð\x9f¤¦â\x80\x8dâ\x99\x82ï¸\x8f": "[facepalm]", # 🤦‍♂️
    "ð\x9f\x98¶": "[neutral]",        # 😶
    "ð\x9f\x98\x81": "[grin]",        # 😁
    "ð\x9f\x98\x84": "[smile_big]",   # 😄
    "ð\x9f\x98¡": "[rage]",           # 😡
    "ð\x9f¥²": "[melting]",          # 🥲 (melting face)
    # Raw emojis that might appear directly
    "😅": "[sweat_smile]",
    "🥲": "[melting]",
    "🚀": "[rocket]",
    "😂": "[joy]",
    "💀": "[skull]",
    "☠️": "[skull]",
    "🤦‍♂️": "[facepalm]",
    "🤦": "[facepalm]",
    "😶": "[neutral]",
    "😁": "[grin]",
    "😄": "[smile_big]",
    "😡": "[rage]"
}


class Message:
    """Standardized message representation."""
    
    def __init__(
        self,
        timestamp: datetime,
        sender: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict] = None
    ):
        self.timestamp = timestamp
        self.sender = sender
        self.content = content
        self.message_type = message_type  # text, image, video, audio, file, reaction, etc.
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


class ChatContext:
    """Container for chat context and metadata."""
    
    def __init__(self, chat_name: str, context_text: str = "", participants: List[str] = None):
        self.chat_name = chat_name
        self.context_text = context_text.strip()
        self.participants = participants or []
        self.message_count = 0
        self.date_range = None
        self.source_format = None
    
    def to_context_file(self) -> str:
        """Generate context file content."""
        lines = [f"Chat: {self.chat_name}"]
        
        if self.context_text:
            lines.append(f"Description: {self.context_text}")
        
        if self.participants:
            lines.append(f"Participants: {', '.join(self.participants)}")
        
        if self.message_count:
            lines.append(f"Message Count: {self.message_count}")
        
        if self.date_range:
            start, end = self.date_range
            start_unix = int(start.timestamp())
            end_unix = int(end.timestamp())
            lines.append(f"Date Range: {start_unix} - {end_unix} (Unix timestamps)")
        
        if self.source_format:
            lines.append(f"Source Format: {self.source_format}")
        
        return "\n".join(lines)


class WhatsAppParser:
    """Parser for WhatsApp chat exports."""
    
    @staticmethod
    def parse_file(file_path: str, context_path: str = None) -> Tuple[List[Message], ChatContext]:
        """Parse WhatsApp chat file."""
        messages = []
        chat_name = Path(file_path).parent.name
        
        # Read context if available
        context_text = ""
        if context_path and os.path.exists(context_path):
            with open(context_path, 'r', encoding='utf-8') as f:
                context_text = f.read().strip()
        
        context = ChatContext(chat_name, context_text)
        context.source_format = "WhatsApp"
        
        participants = set()
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            line_count = 0
            parsed_count = 0
            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    continue
                
                # Parse timestamp format (supports both Unix and WhatsApp formats)
                timestamp = parse_timestamp(line)
                if timestamp:
                    parsed_count += 1
                    # Extract sender and content
                    match = re.match(r'^\[.*?\] ([^:]+): (.*)$', line)
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
                        
                        messages.append(Message(timestamp, sender, content, message_type))
        
        context.participants = list(participants)
        context.message_count = len(messages)
        
        if messages:
            context.date_range = (messages[0].timestamp, messages[-1].timestamp)
        
        return messages, context


class FacebookParser:
    """Parser for Facebook Messenger chat exports."""
    
    @staticmethod
    def _normalize_reaction(reaction_unicode: str) -> str:
        """Convert Unicode reaction to readable format."""
        # First try direct mapping
        if reaction_unicode in FACEBOOK_REACTION_MAP:
            return FACEBOOK_REACTION_MAP[reaction_unicode]
        
        # Try to decode if it appears to be UTF-8 encoded
        try:
            # If the string contains escape sequences, try to decode them
            if '\\x' in repr(reaction_unicode):
                # Convert to bytes and decode as UTF-8
                decoded = reaction_unicode.encode('latin1').decode('utf-8')
                if decoded in FACEBOOK_REACTION_MAP:
                    return FACEBOOK_REACTION_MAP[decoded]
                return f"[{decoded}]"
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        
        # Fallback to original with brackets
        return f"[{reaction_unicode}]"
    
    @staticmethod
    def parse_file(file_path: str, context_path: str = None) -> Tuple[List[Message], ChatContext]:
        """Parse Facebook Messenger JSON file."""
        messages = []
        message_timestamp_map = {}  # Store unix_timestamp -> Message for reaction linking
        recent_messages = []  # Store recent messages for "previous message" lookup
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chat_name = Path(file_path).parent.name
        
        # Read context if available
        context_text = ""
        if context_path and os.path.exists(context_path):
            with open(context_path, 'r', encoding='utf-8') as f:
                context_text = f.read().strip()
        
        context = ChatContext(chat_name, context_text)
        context.source_format = "Facebook Messenger"
        
        # Extract participants
        participants = [p.get('name', 'Unknown') for p in data.get('participants', [])]
        context.participants = participants
        
        # Parse messages (Facebook stores them in reverse chronological order)
        fb_messages = data.get('messages', [])
        fb_messages.reverse()  # Convert to chronological order
        
        # First pass: Process all main messages and build timestamp map
        for msg in fb_messages:
            timestamp = datetime.fromtimestamp(msg['timestamp_ms'] / 1000)
            unix_timestamp = int(timestamp.timestamp())
            sender = msg.get('sender_name', 'Unknown')
            
            # Handle different message types
            if 'content' in msg:
                content = msg['content']
                message_type = "text"
                
                # Skip standalone reaction messages since we handle reactions via the reactions array
                if "reacted" in content and "to your message" in content:
                    continue
                
            elif 'photos' in msg:
                message_type = "image"
                content = f"{len(msg['photos'])} photo(s)"
                
            elif 'videos' in msg:
                message_type = "video" 
                content = f"{len(msg['videos'])} video(s)"
                
            elif 'audio_files' in msg:
                message_type = "audio"
                content = f"{len(msg['audio_files'])} audio file(s)"
                
            elif 'files' in msg:
                message_type = "file"
                content = f"{len(msg['files'])} file(s)"
                
            elif 'gifs' in msg:
                message_type = "image"
                content = "GIF"
                
            elif 'sticker' in msg:
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
            if 'reactions' in msg:
                for reaction in msg['reactions']:
                    reaction_unicode = reaction.get('reaction', '')
                    reactor = reaction.get('actor', 'Unknown')
                    reaction_readable = FacebookParser._normalize_reaction(reaction_unicode)
                    
                    # Create reaction message with timestamp +1 second from original message
                    reaction_timestamp = datetime.fromtimestamp(timestamp.timestamp() + 1)
                    reaction_msg = Message(
                        reaction_timestamp,
                        reactor,
                        f"{reactor} reacted {reaction_readable} to {unix_timestamp}",
                        "reaction"
                    )
                    messages.append(reaction_msg)
                    recent_messages.append(reaction_msg)
        
        context.message_count = len(messages)
        
        if messages:
            context.date_range = (messages[0].timestamp, messages[-1].timestamp)
        
        return messages, context


class ChatProcessor:
    """Main processing engine for parsing chats and creating standardized intermediate representation."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.chats_dir = self.base_dir / "chats"
        self.whatsapp_dir = self.chats_dir / "whatsapp"
        self.fb_dir = self.chats_dir / "facebook"
        self.output_dir = self.base_dir / "chats-ir"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_all_chats(self) -> None:
        """Process all chats from whatsapp-chats and fb-chats folders."""
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        total_processed = 0
        
        # Process WhatsApp chats
        if self.whatsapp_dir.exists():
            self.logger.info(f"📱 Processing WhatsApp chats from: {self.whatsapp_dir}")
            total_processed += self._process_whatsapp_chats()
        else:
            self.logger.info(f"📱 No WhatsApp chats found at: {self.whatsapp_dir}")
        
        # Process Facebook chats
        if self.fb_dir.exists():
            self.logger.info(f"💬 Processing Facebook chats from: {self.fb_dir}")
            total_processed += self._process_facebook_chats()
        else:
            self.logger.info(f"💬 No Facebook chats found at: {self.fb_dir}")
        
        self.logger.info(f"🎉 Processing complete! {total_processed} chat(s) processed")
    
    def process_single_chat(self, chat_name: str) -> None:
        """Process a single chat by name (e.g., 'facebook-soldegen' or 'whatsapp-jiawei')."""
        self.logger.info(f"🎯 Processing single chat: {chat_name}")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Parse chat type and name from the chat_name
        if chat_name.startswith('facebook-'):
            chat_type = 'facebook'
            actual_name = chat_name[9:]  # Remove 'facebook-' prefix
            self._process_single_facebook_chat(actual_name)
        elif chat_name.startswith('whatsapp-'):
            chat_type = 'whatsapp'
            actual_name = chat_name[9:]  # Remove 'whatsapp-' prefix
            self._process_single_whatsapp_chat(actual_name)
        else:
            # Try to infer from directories
            facebook_path = self.fb_dir / chat_name
            whatsapp_path = self.whatsapp_dir / chat_name
            
            if facebook_path.exists():
                self._process_single_facebook_chat(chat_name)
            elif whatsapp_path.exists():
                self._process_single_whatsapp_chat(chat_name)
            else:
                self.logger.error(f"❌ Chat not found: {chat_name}")
                self.logger.info(f"💡 Try: 'facebook-{chat_name}' or 'whatsapp-{chat_name}'")
                return
        
        self.logger.info(f"✅ Completed processing: {chat_name}")
    
    def _process_single_whatsapp_chat(self, chat_name: str) -> None:
        """Process a single WhatsApp chat."""
        chat_folder = self.whatsapp_dir / chat_name
        
        if not chat_folder.exists():
            self.logger.error(f"❌ WhatsApp chat folder not found: {chat_folder}")
            return
        
        chat_file = chat_folder / "chat.txt"
        context_file = chat_folder / "context.txt"
        
        if not chat_file.exists():
            self.logger.error(f"❌ No chat.txt found in {chat_folder}")
            return
        
        self.logger.info(f"📱 Processing WhatsApp chat: {chat_name}")
        
        try:
            # Parse the chat
            messages, context = WhatsAppParser.parse_file(
                str(chat_file), 
                str(context_file) if context_file.exists() else None
            )
            
            # Convert to standardized format and save
            output_path = self.output_dir / f"whatsapp-{chat_name}.txt"
            
            self._save_standardized_chat(messages, context, output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {chat_name}: {e}")
    
    def _process_single_facebook_chat(self, chat_name: str) -> None:
        """Process a single Facebook chat."""
        chat_folder = self.fb_dir / chat_name
        
        if not chat_folder.exists():
            self.logger.error(f"❌ Facebook chat folder not found: {chat_folder}")
            return
        
        # Find JSON files in the folder
        json_files = list(chat_folder.glob("*.json"))
        if not json_files:
            self.logger.error(f"❌ No JSON files found in {chat_folder}")
            return
        
        self.logger.info(f"💬 Processing Facebook chat: {chat_name}")
        
        try:
            # Process the first JSON file (or combine multiple if needed)
            json_file = json_files[0]  # For now, process the first file
            context_file = chat_folder / "context.txt"
            
            # Parse the chat
            messages, context = FacebookParser.parse_file(
                str(json_file), 
                str(context_file) if context_file.exists() else None
            )
            
            # Convert to standardized format and save
            output_path = self.output_dir / f"facebook-{chat_name}.txt"
            
            self._save_standardized_chat(messages, context, output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {chat_name}: {e}")
    
    def _process_whatsapp_chats(self) -> int:
        """Process all WhatsApp chats."""
        processed = 0
        
        for chat_folder in self.whatsapp_dir.iterdir():
            if not chat_folder.is_dir():
                continue
            
            chat_file = chat_folder / "chat.txt"
            context_file = chat_folder / "context.txt"
            
            if not chat_file.exists():
                self.logger.warning(f"⚠️ No chat.txt found in {chat_folder}")
                continue
            
            self.logger.info(f"📱 Processing WhatsApp chat: {chat_folder.name}")
            
            try:
                # Parse the chat
                messages, context = WhatsAppParser.parse_file(
                    str(chat_file), 
                    str(context_file) if context_file.exists() else None
                )
                
                # Convert to standardized format and save
                output_path = self.output_dir / f"whatsapp-{chat_folder.name}.txt"
                
                self._save_standardized_chat(messages, context, output_path)
                processed += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error processing {chat_folder.name}: {e}")
        
        return processed
    
    def _process_facebook_chats(self) -> int:
        """Process all Facebook chats."""
        processed = 0
        
        for chat_folder in self.fb_dir.iterdir():
            if not chat_folder.is_dir():
                continue
            
            # Look for message JSON files
            json_files = list(chat_folder.glob("message_*.json"))
            context_file = chat_folder / "context.txt"
            
            if not json_files:
                self.logger.warning(f"⚠️ No message_*.json found in {chat_folder}")
                continue
            
            self.logger.info(f"💬 Processing Facebook chat: {chat_folder.name}")
            
            try:
                all_messages = []
                context = None
                
                # Process all message files (Facebook sometimes splits large chats)
                for json_file in sorted(json_files):
                    messages, file_context = FacebookParser.parse_file(
                        str(json_file),
                        str(context_file) if context_file.exists() else None
                    )
                    all_messages.extend(messages)
                    if context is None:
                        context = file_context
                
                # Sort all messages by timestamp
                all_messages.sort(key=lambda m: m.timestamp)
                
                # Update context with combined stats
                if context:
                    context.message_count = len(all_messages)
                    if all_messages:
                        context.date_range = (all_messages[0].timestamp, all_messages[-1].timestamp)
                
                # Convert to standardized format and save
                output_path = self.output_dir / f"facebook-{chat_folder.name}.txt"
                
                self._save_standardized_chat(all_messages, context, output_path)
                processed += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error processing {chat_folder.name}: {e}")
        
        return processed
    
    def _save_standardized_chat(
        self, 
        messages: List[Message], 
        context: ChatContext, 
        output_path: Path
    ) -> None:
        """Save messages in standardized format."""
        
        # Convert messages to standardized text format
        standardized_lines = [msg.to_standardized_format() + "\n" for msg in messages]
        
        # Save complete standardized chat
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(standardized_lines)
        
        self.logger.info(f"💾 Saved standardized chat: {output_path}")
        self.logger.info(f"📊 {len(messages)} messages converted to standardized format")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Chat Parser and Processor - Creates Intermediate Representation")
    parser.add_argument('chat_name', nargs='?', help='Specific chat to process (e.g., facebook-soldegen, whatsapp-jiawei)')
    parser.add_argument('--base-dir', default='.', help='Base directory containing chats/ folder with whatsapp/ and fb/ subdirectories')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = ChatProcessor(args.base_dir)
    
    if args.chat_name:
        processor.process_single_chat(args.chat_name)
    else:
        processor.process_all_chats()


if __name__ == "__main__":
    main()

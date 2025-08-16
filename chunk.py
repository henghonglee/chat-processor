#!/usr/bin/env python3
"""
Time-based chunking for WhatsApp chat messages.
Splits messages when time gap between consecutive messages exceeds a threshold.
"""

import re
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


# WhatsApp timestamp pattern: [DD/MM/YY, HH:MM:SS AM/PM]
WHATSAPP_TIMESTAMP_RE = re.compile(
    r'^\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{1,2}:\d{1,2}) (AM|PM)\]'
)

# Unix timestamp pattern: [1234567890]
UNIX_TIMESTAMP_RE = re.compile(r'^\[(\d{10,})\]')


def parse_unix_timestamp(line: str) -> Optional[datetime]:
    """
    Parse Unix timestamp from a message line.
    
    Args:
        line: Message line starting with [1234567890]
        
    Returns:
        datetime object or None if parsing fails
    """
    clean_line = line.strip()
    match = UNIX_TIMESTAMP_RE.match(clean_line)
    if not match:
        return None
    
    try:
        unix_timestamp = int(match.group(1))
        return datetime.fromtimestamp(unix_timestamp)
    except (ValueError, OverflowError, OSError):
        return None


def parse_whatsapp_timestamp(line: str) -> Optional[datetime]:
    """
    Parse WhatsApp timestamp from a message line (legacy format).
    
    Args:
        line: Message line starting with [DD/MM/YY, HH:MM:SS AM/PM]
        
    Returns:
        datetime object or None if parsing fails
    """
    # Clean invisible characters like right-to-left marks and narrow no-break spaces
    clean_line = line.strip().lstrip('\u200e\u200f\u202d\u202e').replace('\u202f', ' ')
    match = WHATSAPP_TIMESTAMP_RE.match(clean_line)
    if not match:
        return None
    
    date_str, time_str, period = match.groups()
    
    try:
        # Parse date (DD/MM/YY or D/M/YY)
        day, month, year = map(int, date_str.split('/'))
        # Convert 2-digit year to 4-digit (assume 2000s)
        year = 2000 + year
        
        # Parse time (HH:MM:SS AM/PM)
        hour, minute, second = map(int, time_str.split(':'))
        
        # Convert to 24-hour format
        if period == 'PM' and hour != 12:
            hour += 12
        elif period == 'AM' and hour == 12:
            hour = 0
            
        return datetime(year, month, day, hour, minute, second)
    
    except (ValueError, IndexError):
        return None


def parse_timestamp(line: str) -> Optional[datetime]:
    """
    Parse timestamp from a message line, supporting both Unix and WhatsApp formats.
    
    Args:
        line: Message line starting with timestamp in brackets
        
    Returns:
        datetime object or None if parsing fails
    """
    # Try Unix timestamp first (new format)
    timestamp = parse_unix_timestamp(line)
    if timestamp:
        return timestamp
    
    # Fall back to WhatsApp format (legacy format)
    return parse_whatsapp_timestamp(line)


def create_time_based_chunks(lines: List[str], gap_minutes: int = 40) -> List[Tuple[int, List[str]]]:
    """
    Create chunks based on time gaps between messages.
    
    Args:
        lines: List of message lines
        gap_minutes: Minimum gap in minutes to create a new chunk
        
    Returns:
        List of (start_index, chunk_lines) tuples
    """
    if not lines:
        return []
    
    chunks = []
    current_chunk = []
    current_start_idx = 0
    last_timestamp = None
    gap_threshold = timedelta(minutes=gap_minutes)
    
    for line_idx, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            continue
            
        # Parse timestamp (supports both Unix and WhatsApp formats)
        timestamp = parse_timestamp(line)
        
        # If we can't parse timestamp, add to current chunk
        if timestamp is None:
            current_chunk.append(line)
            continue
        
        # Check if we need to start a new chunk
        if (last_timestamp is not None and 
            timestamp - last_timestamp > gap_threshold and
            current_chunk):  # Don't create empty chunks
            
            # Finalize current chunk
            chunks.append((current_start_idx, current_chunk))
            
            # Start new chunk
            current_chunk = [line]
            current_start_idx = line_idx
        else:
            # Add to current chunk
            current_chunk.append(line)
        
        last_timestamp = timestamp
    
    # Add final chunk if it has content
    if current_chunk:
        chunks.append((current_start_idx, current_chunk))
    
    return chunks


def chunk_chat_file(file_path: str, gap_minutes: int = 40) -> List[Tuple[int, List[str]]]:
    """
    Read a chat file and create time-based chunks.
    
    Args:
        file_path: Path to the chat file
        gap_minutes: Minimum gap in minutes to create a new chunk
        
    Returns:
        List of (start_index, chunk_lines) tuples
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [line.rstrip() + '\n' for line in f if line.strip()]
    
    return create_time_based_chunks(lines, gap_minutes)


def get_chunk_info(chunks: List[Tuple[int, List[str]]]) -> str:
    """
    Get human-readable information about the chunks.
    
    Args:
        chunks: List of (start_index, chunk_lines) tuples
        
    Returns:
        Formatted string with chunk information
    """
    if not chunks:
        return "No chunks created"
    
    info_lines = [f"Created {len(chunks)} time-based chunks:"]
    
    for i, (start_idx, chunk_lines) in enumerate(chunks, 1):
        # Get first and last timestamps from chunk
        first_timestamp = None
        last_timestamp = None
        
        for line in chunk_lines:
            timestamp = parse_timestamp(line)
            if timestamp:
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
        
        if first_timestamp and last_timestamp:
            duration = last_timestamp - first_timestamp
            info_lines.append(
                f"  Chunk {i}: {len(chunk_lines)} lines, "
                f"{first_timestamp.strftime('%d/%m/%y %H:%M')} → "
                f"{last_timestamp.strftime('%d/%m/%y %H:%M')} "
                f"(duration: {duration})"
            )
        else:
            info_lines.append(f"  Chunk {i}: {len(chunk_lines)} lines (no timestamps)")
    
    return "\n".join(info_lines)


def process_all_chats(base_dir: str = ".", gap_minutes: int = 180) -> None:
    """Process all chats from chats-ir/ directory and create chunks."""
    import os
    from pathlib import Path
    
    base_path = Path(base_dir)
    input_dir = base_path / "chats-ir"
    output_dir = base_path / "chats-ir-chunked"
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find all .txt files in chats-ir/
    ir_files = list(input_dir.glob("*.txt"))
    
    if not ir_files:
        print(f"❌ No .txt files found in {input_dir}")
        return
    
    print(f"📂 Found {len(ir_files)} chat files to chunk")
    total_processed = 0
    
    for ir_file in ir_files:
        chat_name = ir_file.stem  # Remove .txt extension
        print(f"🔄 Processing: {chat_name}")
        
        try:
            # Create output folder for this chat
            chat_output_dir = output_dir / chat_name
            chat_output_dir.mkdir(exist_ok=True)
            
            # Read and chunk the file
            with open(ir_file, 'r', encoding='utf-8') as f:
                lines = [line.rstrip() + '\n' for line in f if line.strip()]
            
            chunks = create_time_based_chunks(lines, gap_minutes)
            
            if chunks:
                # Save individual chunks
                for i, (start_idx, chunk_lines) in enumerate(chunks, 1):
                    chunk_path = chat_output_dir / f"chunk_{i:03d}.txt"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.writelines(chunk_lines)
                
                print(f"✅ {chat_name}: Created {len(chunks)} chunks")
                total_processed += 1
            else:
                print(f"⚠️ {chat_name}: No chunks created")
                
        except Exception as e:
            print(f"❌ Error processing {chat_name}: {e}")
    
    print(f"🎉 Processing complete! {total_processed}/{len(ir_files)} chats processed")


def process_single_chat(chat_name: str, base_dir: str = ".", gap_minutes: int = 180) -> None:
    """Process a single chat by name."""
    from pathlib import Path
    
    base_path = Path(base_dir)
    input_dir = base_path / "chats-ir"
    output_dir = base_path / "chats-ir-chunked"
    
    # Find the chat file
    ir_file = input_dir / f"{chat_name}.txt"
    
    if not ir_file.exists():
        print(f"❌ Chat file not found: {ir_file}")
        print(f"💡 Available chats:")
        available = list(input_dir.glob("*.txt"))
        for f in available:
            print(f"  - {f.stem}")
        return
    
    print(f"🎯 Processing single chat: {chat_name}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    chat_output_dir = output_dir / chat_name
    chat_output_dir.mkdir(exist_ok=True)
    
    try:
        # Read and chunk the file
        with open(ir_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip() + '\n' for line in f if line.strip()]
        
        chunks = create_time_based_chunks(lines, gap_minutes)
        
        if chunks:
            # Save individual chunks
            for i, (start_idx, chunk_lines) in enumerate(chunks, 1):
                chunk_path = chat_output_dir / f"chunk_{i:03d}.txt"
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.writelines(chunk_lines)
            
            print(f"✅ Created {len(chunks)} chunks in: {chat_output_dir}")
            
            # Show chunk info
            print(f"\n📊 Chunk Analysis:")
            print(get_chunk_info(chunks))
            
        else:
            print(f"⚠️ No chunks created for {chat_name}")
            
    except Exception as e:
        print(f"❌ Error processing {chat_name}: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Time-based Chat Chunker - Creates chunks from standardized IR")
    parser.add_argument('chat_name', nargs='?', help='Specific chat to chunk (e.g., facebook-soldegen, whatsapp-jiawei)')
    parser.add_argument('--base-dir', default='.', help='Base directory containing chats-ir/ folder')
    parser.add_argument('--gap-minutes', type=int, default=180, help='Minutes gap for chunking conversations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.chat_name:
        process_single_chat(args.chat_name, args.base_dir, args.gap_minutes)
    else:
        process_all_chats(args.base_dir, args.gap_minutes)


if __name__ == "__main__":
    main()

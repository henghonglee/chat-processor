"""
Text Cleaner Preprocessor

Handles text cleaning and normalization including:
- URL removal
- Whitespace normalization
- Special character handling
- Text encoding fixes
"""

import re
import time
from typing import Any, Dict, Optional

from .base import BasePreprocessor, PreprocessorResult


class TextCleaner(BasePreprocessor):
    """Preprocessor for cleaning and normalizing text."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text cleaner.

        Args:
            config: Configuration dictionary with options:
                - remove_urls: Remove URLs (default: True)
                - remove_mentions: Remove @mentions (default: False)
                - remove_hashtags: Remove #hashtags (default: False)
                - normalize_whitespace: Normalize whitespace (default: True)
                - strip_text: Strip leading/trailing whitespace (default: True)
                - preserve_line_breaks: Keep line breaks (default: False)
                - replace_personal_pronouns: Replace 'i'/'me'/'my'/'mine' with 'HengHong' (default: True)
        """
        super().__init__(config)
        self.setup_patterns()

    def setup_patterns(self):
        """Setup regex patterns for text cleaning."""
        # URL patterns
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # Social media patterns
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#\w+")

        # Whitespace patterns
        self.extra_whitespace = re.compile(r"\s+")
        self.line_break_pattern = re.compile(r"\n+")

        # Special character patterns
        self.special_chars = re.compile(r"[^\w\s\-.,!?;:()\[\]{}\"']")

    def get_name(self) -> str:
        """Get the name of this preprocessor."""
        return "TextCleaner"

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub("", text)

    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return self.mention_pattern.sub("", text)

    def remove_hashtags(self, text: str) -> str:
        """Remove #hashtags from text."""
        return self.hashtag_pattern.sub("", text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        if self.get_config_value("preserve_line_breaks", False):
            # Preserve line breaks but normalize other whitespace
            lines = text.split("\n")
            normalized_lines = [
                self.extra_whitespace.sub(" ", line.strip()) for line in lines
            ]
            return "\n".join(normalized_lines)
        else:
            # Normalize all whitespace to single spaces
            return self.extra_whitespace.sub(" ", text)

    def remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving important punctuation."""
        return self.special_chars.sub("", text)

    def replace_personal_pronouns(self, text: str) -> str:
        """Replace personal pronouns 'i', 'me', 'my', 'mine' with 'HengHong'."""
        # Use word boundaries to ensure we only replace standalone words
        # Handle contractions properly
        
        # Replace "I" (uppercase) with "HengHong", including contractions
        text = re.sub(r'\bI\'', 'HengHong\'', text)  # I'm, I'll, I've, etc.
        text = re.sub(r'\bI\b', 'HengHong', text)   # Standalone I
        
        # Replace "i" (lowercase) with "HengHong"  
        text = re.sub(r'\bi\b', 'HengHong', text)
        
        # Replace "me" (any case) with "HengHong"
        text = re.sub(r'\bme\b', 'HengHong', text, flags=re.IGNORECASE)
        
        # Replace "my" (any case) with "HengHong's"
        text = re.sub(r'\bmy\b', "HengHong's", text, flags=re.IGNORECASE)
        
        # Replace "mine" (any case) with "HengHong's"
        text = re.sub(r'\bmine\b', "HengHong's", text, flags=re.IGNORECASE)
        
        return text

    def clean_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Fix common Unicode issues
        replacements = {
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u2026": "...",  # Horizontal ellipsis
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def process(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> PreprocessorResult:
        """
        Clean and normalize the input text.

        Args:
            text: Input text to clean
            context: Optional context from previous processors

        Returns:
            PreprocessorResult with cleaned text
        """
        start_time = time.time()
        self.log_processing_start(text)

        try:
            # Validate input
            if not self.validate_input(text):
                return PreprocessorResult(
                    success=False,
                    error="Invalid input text",
                    processing_time=time.time() - start_time,
                )

            cleaned_text = text

            # Apply cleaning steps based on configuration
            if self.get_config_value("remove_urls", True):
                cleaned_text = self.remove_urls(cleaned_text)

            if self.get_config_value("remove_mentions", False):
                cleaned_text = self.remove_mentions(cleaned_text)

            if self.get_config_value("remove_hashtags", False):
                cleaned_text = self.remove_hashtags(cleaned_text)

            # Fix encoding issues
            cleaned_text = self.clean_encoding(cleaned_text)

            # Replace personal pronouns if configured
            if self.get_config_value("replace_personal_pronouns", True):
                cleaned_text = self.replace_personal_pronouns(cleaned_text)

            # Remove special characters if configured
            if self.get_config_value("remove_special_chars", False):
                cleaned_text = self.remove_special_characters(cleaned_text)

            # Normalize whitespace
            if self.get_config_value("normalize_whitespace", True):
                cleaned_text = self.normalize_whitespace(cleaned_text)

            # Strip leading/trailing whitespace
            if self.get_config_value("strip_text", True):
                cleaned_text = cleaned_text.strip()

            # Validate result
            if not cleaned_text:
                return PreprocessorResult(
                    success=False,
                    error="Text cleaning resulted in empty string",
                    processing_time=time.time() - start_time,
                )

            processing_time = time.time() - start_time
            result = PreprocessorResult(
                success=True,
                data={
                    "cleaned_text": cleaned_text,
                    "original_length": len(text),
                    "cleaned_length": len(cleaned_text),
                    "reduction_ratio": (
                        1 - (len(cleaned_text) / len(text)) if len(text) > 0 else 0
                    ),
                },
                processing_time=processing_time,
                metadata={
                    "applied_operations": self._get_applied_operations(),
                    "character_count": len(cleaned_text),
                    "word_count": len(cleaned_text.split()),
                },
            )

            self.log_processing_end(result)
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Text cleaning failed: {str(e)}"
            self.logger.error(error_msg)

            return PreprocessorResult(
                success=False,
                error=error_msg,
                processing_time=processing_time,
            )

    def _get_applied_operations(self) -> list:
        """Get list of operations that were applied based on config."""
        operations = []
        if self.get_config_value("remove_urls", True):
            operations.append("url_removal")
        if self.get_config_value("remove_mentions", False):
            operations.append("mention_removal")
        if self.get_config_value("remove_hashtags", False):
            operations.append("hashtag_removal")
        if self.get_config_value("replace_personal_pronouns", True):
            operations.append("personal_pronoun_replacement")
        if self.get_config_value("normalize_whitespace", True):
            operations.append("whitespace_normalization")
        if self.get_config_value("strip_text", True):
            operations.append("text_stripping")
        if self.get_config_value("remove_special_chars", False):
            operations.append("special_char_removal")
        return operations

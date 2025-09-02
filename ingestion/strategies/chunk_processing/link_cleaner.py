"""
Link cleaning and normalization processor.
"""

import re

from .base import BaseProcessor


class LinkCleanerProcessor(BaseProcessor):
    """Processor that cleans and normalizes URLs."""

    def __init__(self):
        super().__init__()

    def _process_content(self, content: str, context: dict) -> str:
        """Clean and normalize URLs in content."""
        # Remove tracking parameters from URLs
        tracking_params = [
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid",
            "msclkid",
            "ref_src",
            "ref_url",
        ]

        def clean_url(match):
            url = match.group(0)

            # Remove tracking parameters
            for param in tracking_params:
                url = re.sub(rf"[?&]{param}=[^&]*", "", url)

            # Clean up dangling ? or & at the end
            url = re.sub(r"[?&]$", "", url)

            return url

        # Apply URL cleaning
        processed_content = re.sub(
            r'https?://[^\s<>"\']+', clean_url, content, flags=re.IGNORECASE
        )

        return processed_content

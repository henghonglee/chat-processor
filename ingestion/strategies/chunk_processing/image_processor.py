"""
Image processing processor.
"""

import re

from .base import BaseProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


class ImageProcessor(BaseProcessor):
    """Processor that handles image URLs and attachments."""

    def __init__(self):
        super().__init__()

    def _process_content(self, content: str, context: dict) -> str:
        """Process image references in content."""
        # Find image URLs
        image_urls = []
        for match in re.finditer(r'https?://[^\s<>"\']+', content, re.IGNORECASE):
            url = match.group(0).rstrip(".,)")
            if any(url.lower().endswith(ext) for ext in IMAGE_EXTS):
                image_urls.append(url)

        if not image_urls:
            return content

        processed_content = content

        for url in image_urls:
            processed_content += f"\n--- Image: {url} ---\n"

        return processed_content

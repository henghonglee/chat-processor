"""
URL expansion processor.
"""

import re
from typing import Optional

from .base import BaseProcessor


# Optional imports handled gracefully
def _try_import(modname: str):
    try:
        return __import__(modname)
    except ImportError:
        return None


requests = _try_import("requests")
bs4 = _try_import("bs4")
trafilatura = _try_import("trafilatura")

URL_RE = re.compile(r"""(?P<url>https?://[^\s<>"'\)\]]+)""", re.IGNORECASE)
USER_AGENT = "chat-processor/1.0"


class URLExpanderProcessor(BaseProcessor):
    """Processor that expands URLs to their content."""

    def __init__(self, timeout: int = 15, cache: Optional[dict] = None):
        super().__init__()
        self.timeout = timeout
        self.cache = cache or {}

    def _process_content(self, content: str, context: dict) -> str:
        """Expand URLs found in content."""
        if requests is None:
            return content

        # Find URLs in content
        urls = [match.group("url").rstrip(").,") for match in URL_RE.finditer(content)]

        if not urls:
            return content

        processed_content = content

        for url in urls:
            # Skip social media URLs (handled by specialized processors)
            if any(
                domain in url.lower()
                for domain in ["twitter.com", "x.com", "facebook.com", "instagram.com"]
            ):
                continue

            try:
                expanded_content = self._fetch_url_content(url)
                if expanded_content:
                    processed_content += f" --- URL Content from {url} --- {expanded_content} --- End URL Content --- "
            except Exception as e:
                processed_content += f" --- URL {url} (error: {e}) --- "

        return processed_content

    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch and extract readable text from a URL."""
        # Check cache first
        if url in self.cache:
            return self.cache[url]

        try:
            headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
            resp = requests.get(url, headers=headers, timeout=self.timeout)

            if resp.status_code >= 400:
                return None

            # Check content type to avoid processing binary files
            content_type = resp.headers.get("content-type", "").lower()
            if any(
                binary_type in content_type
                for binary_type in [
                    "image/",
                    "video/",
                    "audio/",
                    "application/octet-stream",
                ]
            ):
                result = f"[Binary content: {content_type}]"
                self.cache[url] = result
                return result

            html = resp.content.decode("utf-8", errors="replace")

            # Try trafilatura first (better extraction)
            if trafilatura is not None:
                extracted = trafilatura.extract(
                    html, include_comments=False, include_tables=False
                )
                if extracted:
                    meta = trafilatura.extract_metadata(html)
                    title = meta.title if meta else None
                    content = extracted.strip()
                    if title:
                        result = f"Title: {title}\n\n{content}"
                    else:
                        result = content
                    self.cache[url] = result
                    return result

            # Fallback to BeautifulSoup
            if bs4 is None:
                return None

            soup = bs4.BeautifulSoup(html, "html.parser")

            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract title
            title = soup.find("title")
            title_text = title.get_text().strip() if title else "No title"

            # Extract main content
            content_tags = soup.find_all(["p", "div", "article", "main"])
            content_texts = [
                tag.get_text().strip() for tag in content_tags if tag.get_text().strip()
            ]

            if content_texts:
                content = "\n".join(content_texts[:10])  # Limit to first 10 paragraphs
                result = f"Title: {title_text} - {content}"
                self.cache[url] = result
                return result
            else:
                result = f"Title: {title_text} [No readable content found]"
                self.cache[url] = result
                return result

        except Exception:
            return None

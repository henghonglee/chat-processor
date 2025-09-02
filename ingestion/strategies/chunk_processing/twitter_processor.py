"""Twitter/X specific processor."""

import os
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

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import selenium components
webdriver = _try_import("selenium")
if webdriver:
    try:
        from selenium import webdriver as wd
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        webdriver = None


class TwitterProcessor(BaseProcessor):
    """Processor specifically for Twitter/X URLs."""

    def __init__(self, timeout: int = 15, cache: Optional[dict] = None):
        super().__init__()
        self.timeout = timeout
        self.cache = cache or {}

    def _process_content(self, content: str, context: dict) -> str:
        """Process Twitter/X URLs in content."""
        # Skip requests check if we have cache entries
        if requests is None and not self.cache:
            return content

        # Find Twitter URLs
        twitter_urls = []
        for match in re.finditer(
            r"https?://(?:twitter\.com|x\.com)/\S+", content, re.IGNORECASE
        ):
            twitter_urls.append(match.group(0).rstrip(".,)"))

        if not twitter_urls:
            return content

        processed_content = content

        for url in twitter_urls:
            try:
                twitter_content = self._fetch_twitter_content(url)
                if twitter_content:
                    # Extract just the content part (remove "Twitter/X Content: " prefix)
                    content_text = twitter_content
                    if content_text.startswith("Twitter/X Content: "):
                        content_text = content_text[
                            19:
                        ]  # Remove "Twitter/X Content: " prefix

                    # Replace the URL with URL + XML expansion inline
                    expansion = f"{url} <xcontent>{content_text}</xcontent>"
                    processed_content = processed_content.replace(url, expansion)
                else:
                    # If no content fetched, cache as "Post removed" and expand inline
                    normalized_url = self._normalize_twitter_url(url)
                    if normalized_url not in self.cache:
                        # Cache the failed URL so we don't try again
                        failure_content = "Post removed"
                        self.cache[normalized_url] = failure_content
                        self._save_to_cache_file(normalized_url, failure_content)
                        print(f"📝 Cached failed URL as 'Post removed': {url[:60]}...")

                    # Expand with "Post removed" inline
                    expansion = f"{url} <xcontent>Post removed</xcontent>"
                    processed_content = processed_content.replace(url, expansion)

            except Exception:
                # On error, also cache as "Post removed"
                normalized_url = self._normalize_twitter_url(url)
                if normalized_url not in self.cache:
                    failure_content = "Post removed"
                    self.cache[normalized_url] = failure_content
                    self._save_to_cache_file(normalized_url, failure_content)
                    print(f"📝 Cached error URL as 'Post removed': {url[:60]}...")

                # Expand with "Post removed" inline
                expansion = f"{url} <xcontent>Post removed</xcontent>"
                processed_content = processed_content.replace(url, expansion)

        return processed_content

    def _normalize_twitter_url(self, url: str) -> str:
        """Normalize Twitter URL by removing query parameters and tracking info."""
        import re
        from urllib.parse import urlparse

        # Parse the URL and remove query parameters
        parsed = urlparse(url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Ensure it follows the standard pattern
        match = re.search(r"(https://x\.com/[^/]+/status/\d+)", clean_url)
        if match:
            return match.group(1)
        return clean_url

    def _fetch_twitter_content(self, url: str) -> Optional[str]:
        """Fetch Twitter content using cache first, then fetch and cache if missing."""
        # Normalize the URL for cache lookup
        normalized_url = self._normalize_twitter_url(url)

        # Check cache first with normalized URL
        if normalized_url in self.cache:
            content = self.cache[normalized_url]
            if content:
                # Replace both actual newlines and escaped newlines with spaces
                content = content.replace("\n", " ").replace("\r", " ")
                content = content.replace("\\n", " ").replace("\\r", " ")
                return content
            return content

        # If not in cache, try to fetch and cache it
        fetched_content = self._fetch_and_cache_url(normalized_url)
        if fetched_content:
            return fetched_content

        # If fetching failed, return None (don't cache failures)
        return None

    def _fetch_and_cache_url(self, url: str) -> Optional[str]:
        """Fetch a URL and cache it if successful."""
        if requests is None:
            return None

        try:
            # Try Twitter syndication API first (fastest if it works)
            print(f"🔄 Trying Twitter syndication for: {url[:60]}...")
            syndication_content = self._fetch_twitter_via_syndication(url)
            if syndication_content and len(syndication_content) > 10:
                # Clean the content before caching
                syndication_content = syndication_content.replace("\n", " ").replace(
                    "\r", " "
                )
                syndication_content = syndication_content.replace("\\n", " ").replace(
                    "\\r", " "
                )

                # Cache the successful fetch
                self.cache[url] = syndication_content
                self._save_to_cache_file(url, syndication_content)
                print(f"✅ Cached via syndication: {url[:60]}...")
                return syndication_content

            # Try Twitter APIs second (lightweight but more reliable)
            print(f"🔄 Trying Twitter APIs for: {url[:60]}...")
            content = self._fetch_twitter_via_api(url)
            if content and len(content) > 10:  # Only cache if we got meaningful content
                # Clean the content before caching
                content = content.replace("\n", " ").replace("\r", " ")
                content = content.replace("\\n", " ").replace("\\r", " ")

                # Cache the successful fetch
                self.cache[url] = content
                self._save_to_cache_file(url, content)
                print(f"✅ Cached to url_cache.csv: {url[:60]}...")
                return content

            # Browser automation is disabled
            print(f"❌ All API methods failed for: {url[:60]}...")
            print("ℹ️ Browser automation is disabled - URL will remain unexpanded")
            return None

        except Exception as e:
            print(f"❌ Error fetching {url[:60]}...: {e}")
            return None

    def _save_to_cache_file(self, url: str, content: str):
        """Append a new entry to the cache file."""
        import csv
        from pathlib import Path

        cache_file = Path("url_cache.csv")

        # Append to existing file
        with open(cache_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([url, content])

    def _fetch_twitter_via_syndication(self, url: str) -> Optional[str]:
        """Try to fetch tweet content via Twitter's syndication API endpoints."""
        try:
            # Extract tweet ID from URL
            tweet_info = self._extract_twitter_info(url)
            if not tweet_info:
                return None

            username, tweet_id = tweet_info

            # Try Twitter's syndication endpoints
            syndication_endpoints = [
                f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=en",
                f"https://cdn.syndication.twimg.com/widgets/timelines/embedded_timeline?id={tweet_id}",
            ]

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://platform.twitter.com/",
                "Origin": "https://platform.twitter.com",
            }

            for endpoint in syndication_endpoints:
                try:
                    response = requests.get(endpoint, headers=headers, timeout=8)
                    if response.status_code == 200:
                        # Try to parse JSON response
                        try:
                            data = response.json()
                            if "text" in data:
                                text = data["text"]
                                if text and len(text) > 10:
                                    return f"Twitter/X Content: {text}"
                        except ValueError:
                            # Not JSON, try HTML parsing
                            if response.text and len(response.text) > 100:
                                return "Twitter/X Content: [Syndication data available]"

                except Exception:
                    continue

            return None

        except Exception:
            return None

    def _fetch_twitter_via_api(self, url: str) -> Optional[str]:
        """Extract Twitter/X content using API-based approaches."""
        # Extract tweet info once
        tweet_info = self._extract_twitter_info(url)
        if not tweet_info:
            return None

        username, tweet_id = tweet_info

        # Strategy 1: Try Twitter's oEmbed API (public, most reliable)
        print("  🔄 Trying oEmbed API...")
        oembed_result = self._fetch_via_oembed(url, tweet_id)
        if oembed_result:
            print("  ✅ Success with oEmbed API...")
            return oembed_result

        # Strategy 2: Try Twitter's guest token API
        print("  🔄 Trying guest token API...")
        guest_token_result = self._fetch_via_guest_token(tweet_id, username)
        if guest_token_result:
            print("  ✅ Success with guest token API")
            return guest_token_result

        # Strategy 3: Try Twitter's GraphQL API (mobile endpoints)
        print("  🔄 Trying GraphQL API...")
        graphql_result = self._fetch_via_graphql(tweet_id, username)
        if graphql_result:
            print("  ✅ Success with GraphQL API")
            return graphql_result

        # Strategy 4: Try Twitter's legacy syndication endpoints
        print("  🔄 Trying legacy syndication...")
        legacy_result = self._fetch_via_legacy_syndication(tweet_id, username)
        if legacy_result:
            print("  ✅ Success with legacy syndication")
            return legacy_result

        return None

    def _fetch_via_oembed(self, url: str, tweet_id: str) -> Optional[str]:
        """Try Twitter's oEmbed API (public endpoint)."""
        try:
            oembed_url = (
                f"https://publish.twitter.com/oembed?url={url}&omit_script=true"
            )
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }

            response = requests.get(oembed_url, headers=headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                if "html" in data:
                    # Parse HTML to extract tweet content
                    if bs4:
                        soup = bs4.BeautifulSoup(data["html"], "html.parser")
                        # Extract text content from oEmbed HTML
                        text_content = soup.get_text().strip()
                        # Clean up the text
                        lines = [
                            line.strip()
                            for line in text_content.split("\n")
                            if line.strip()
                        ]
                        cleaned_text = " ".join(lines)
                        if cleaned_text and len(cleaned_text) > 10:
                            return f"Twitter/X Content: {cleaned_text}"

        except Exception:
            pass
        return None

    def _fetch_via_guest_token(self, tweet_id: str, username: str) -> Optional[str]:
        """Try to fetch content using Twitter API v2 with bearer token."""
        try:
            # Get bearer token from environment
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
            if not bearer_token:
                return None

            # Use Twitter API v2 to fetch tweet
            api_url = f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=text,author_id,created_at,public_metrics"
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "TwitterAPIExpansion/1.0",
            }

            response = requests.get(api_url, headers=headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and "text" in data["data"]:
                    return f"Twitter/X Content: {data['data']['text']}"

        except Exception:
            pass
        return None

    def _get_guest_token(self) -> Optional[str]:
        """Get API credentials from environment."""
        # This method is kept for compatibility but now returns API key
        return os.getenv("TWITTER_API_KEY")

    def _fetch_via_graphql(self, tweet_id: str, username: str) -> Optional[str]:
        """Try Twitter's GraphQL endpoints (mobile API)."""
        try:
            # Twitter's mobile GraphQL endpoint for tweet details
            graphql_url = (
                "https://twitter.com/i/api/graphql/VWxGj2thymNgPKcFeONSTQ/TweetDetail"
            )

            headers = {
                "User-Agent": "TwitterAndroid/10.10.0",
                "X-Twitter-Client-Language": "en",
                "X-Twitter-Active-User": "yes",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            # GraphQL variables for the query
            variables = {
                "focalTweetId": tweet_id,
                "with_rux_injections": False,
                "includePromotedContent": True,
                "withCommunity": True,
                "withQuickPromoteEligibilityTweetFields": True,
                "withBirdwatchNotes": True,
                "withVoice": True,
                "withV2Timeline": True,
            }

            import json

            params = {"variables": json.dumps(variables)}

            response = requests.get(
                graphql_url, headers=headers, params=params, timeout=8
            )
            if response.status_code == 200:
                data = response.json()
                # Parse GraphQL response for tweet text
                if (
                    "data" in data
                    and "threaded_conversation_with_injections_v2" in data["data"]
                ):
                    instructions = data["data"][
                        "threaded_conversation_with_injections_v2"
                    ]["instructions"]
                    for instruction in instructions:
                        if "entries" in instruction:
                            for entry in instruction["entries"]:
                                if (
                                    "content" in entry
                                    and "itemContent" in entry["content"]
                                ):
                                    tweet_data = (
                                        entry["content"]["itemContent"]
                                        .get("tweet_results", {})
                                        .get("result", {})
                                    )
                                    if (
                                        "legacy" in tweet_data
                                        and "full_text" in tweet_data["legacy"]
                                    ):
                                        return f"Twitter/X Content: {tweet_data['legacy']['full_text']}"

        except Exception:
            pass
        return None

    def _fetch_via_legacy_syndication(
        self, tweet_id: str, username: str
    ) -> Optional[str]:
        """Try legacy Twitter syndication endpoints with authentication."""
        try:
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

            # Try different endpoints (some with auth, some without)
            endpoints_configs = [
                {
                    "url": f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=text,author_id,created_at,public_metrics",
                    "headers": (
                        {
                            "Authorization": f"Bearer {bearer_token}",
                            "User-Agent": "TwitterAPIExpansion/1.0",
                        }
                        if bearer_token
                        else {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        }
                    ),
                },
                {
                    "url": f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=en",
                    "headers": {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                        "Referer": "https://platform.twitter.com/",
                    },
                },
            ]

            for config in endpoints_configs:
                try:
                    response = requests.get(
                        config["url"], headers=config["headers"], timeout=8
                    )
                    if response.status_code == 200:
                        data = response.json()

                        # Handle different response formats
                        if "data" in data and "text" in data["data"]:
                            return f"Twitter/X Content: {data['data']['text']}"
                        elif "text" in data:
                            return f"Twitter/X Content: {data['text']}"

                except Exception:
                    continue

        except Exception:
            pass
        return None

    def _fetch_js_content_selenium(self, url: str) -> Optional[str]:
        """Fetch content using Selenium browser automation."""
        if webdriver is None:
            return None

        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--user-agent=chat-processor/1.0")

            service = Service(ChromeDriverManager().install())
            driver = wd.Chrome(service=service, options=options)

            try:
                driver.set_page_load_timeout(self.timeout)
                driver.get(url)

                return self._extract_twitter_content_selenium(driver, url)

            finally:
                driver.quit()

        except Exception:
            return None

    def _extract_twitter_content_selenium(self, driver, url: str) -> Optional[str]:
        """Extract content from Twitter/X using Selenium."""
        try:
            import time

            time.sleep(2)  # Wait for content to load

            # Try to find tweet content
            tweet_selectors = [
                '[data-testid="tweetText"]',
                '[data-testid="tweet"]',
                ".tweet-text",
                ".css-901oao",
            ]

            content_parts = []

            # Extract tweet text
            for selector in tweet_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        if text and text not in content_parts:
                            content_parts.append(text)
                    break
                except Exception:
                    continue

            # Extract images and media
            try:
                img_elements = driver.find_elements(
                    By.CSS_SELECTOR, 'img[src*="pbs.twimg.com"]'
                )
                for img in img_elements[:3]:  # Limit to first 3 images
                    src = img.get_attribute("src")
                    if src and self._is_tweet_content_image(src):
                        content_parts.append(f"<image: {src}>")
            except Exception:
                pass

            if content_parts:
                return "Twitter/X Content:\n\n" + "\n\n".join(content_parts)
            else:
                return None

        except Exception:
            return None

    def _extract_twitter_info(self, url: str) -> Optional[tuple]:
        """Extract username and tweet ID from Twitter/X URL."""
        try:
            # Handle various Twitter URL formats
            patterns = [
                r"(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)",
                r"(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)\?",
                r"(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)/",
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    username = match.group(1)
                    tweet_id = match.group(2)
                    return (username, tweet_id)

            return None

        except Exception:
            return None

    def _is_tweet_content_image(self, img_url: str) -> bool:
        """Check if image is tweet content (not profile pic)."""
        return "profile_images" not in img_url.lower()

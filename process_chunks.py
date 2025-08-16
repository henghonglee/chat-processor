#!/usr/bin/env python3
"""
Chunk Content Processor

Processes all chunks in chats-ir-chunked/ and expands URLs, processes images,
and extracts content, saving results to chats-processed/.
"""

import argparse
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# We'll use our own URL processing functions based on the chat.py file
import re
import requests
from urllib.parse import urlparse

# Optional imports handled gracefully
def _try_import(modname: str):
    try:
        return __import__(modname)
    except ImportError:
        return None

# Import optional dependencies
bs4 = _try_import("bs4")
trafilatura = _try_import("trafilatura")
requests = _try_import("requests")

# Import selenium components
webdriver = _try_import("selenium")
if webdriver:
    try:
        from selenium import webdriver as wd
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        webdriver = None

URL_RE = re.compile(r"""(?P<url>https?://[^\s<>"'\)\]]+)""", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
USER_AGENT = "chat-processor/1.0"


class URLCache:
    """CSV-based file system cache for all URL content."""
    
    def __init__(self, cache_file: str = "url_cache.csv"):
        self.cache_file = Path(cache_file)
        self.cache = {}
        self.load_cache()
    
    def load_cache(self):
        """Load cache from CSV file."""
        import csv
        
        if not self.cache_file.exists():
            logging.info(f"📁 Creating new URL cache: {self.cache_file}")
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                # Skip header if present
                try:
                    first_row = next(reader)
                    if first_row[0] != 'url':  # Not a header row
                        self.cache[first_row[0]] = first_row[1]
                except StopIteration:
                    pass
                
                # Load rest of the data
                for row in reader:
                    if len(row) >= 2:
                        url, content = row[0], row[1]
                        self.cache[url] = content
            
            logging.info(f"✅ Loaded {len(self.cache)} entries from URL cache")
            
        except Exception as e:
            logging.warning(f"⚠️ Error loading cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save cache to CSV file."""
        import csv
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(['url', 'content'])  # Header
                
                for url, content in self.cache.items():
                    # Escape newlines in content
                    escaped_content = content.replace('\n', '\\n').replace('\r', '\\r')
                    writer.writerow([url, escaped_content])
            
            logging.debug(f"💾 Saved {len(self.cache)} entries to URL cache")
            
        except Exception as e:
            logging.warning(f"⚠️ Error saving cache: {e}")
    
    def get(self, url: str) -> str:
        """Get cached content for URL."""
        content = self.cache.get(url)
        if content:
            # Unescape newlines
            content = content.replace('\\n', '\n').replace('\\r', '\r')
            logging.debug(f"🎯 Cache HIT for {url}")
            return content
        else:
            logging.debug(f"❌ Cache MISS for {url}")
            return None
    
    def set(self, url: str, content: str) -> None:
        """Cache content for URL."""
        if not content:
            return
        
        # Only cache successful extractions (not error messages)
        if not content.startswith("[") or content.startswith("Twitter/X Content:") or content.startswith("Title:"):
            self.cache[url] = content
            logging.debug(f"💾 Cached content for {url}")
            # Save immediately to persist changes
            self.save_cache()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        file_size = 0
        if self.cache_file.exists():
            file_size = self.cache_file.stat().st_size
        
        return {
            "enabled": True,
            "cached_urls": len(self.cache),
            "cache_file": str(self.cache_file),
            "file_size_bytes": file_size,
            "file_size_human": f"{file_size / 1024:.1f} KB" if file_size > 0 else "0 B"
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logging.info("🗑️ URL cache cleared")


# Global cache instance
url_cache = URLCache()


def fetch_js_content(url: str, timeout: int) -> str:
    """Fetch content from JavaScript-heavy sites using Nitter first, then Selenium fallback."""
    # For Twitter/X URLs, try Nitter first, then fallback to Selenium
    if 'twitter.com' in url.lower() or 'x.com' in url.lower():
        return fetch_twitter_hybrid(url, timeout)
    
    # For other social media sites, try Selenium if available
    if webdriver is not None:
        return fetch_js_content_selenium(url, timeout)
    
    # Final fallback
    return f"[Social media URL - limited extraction available: {url}]"


def fetch_twitter_via_nitter(url: str, timeout: int) -> str:
    """Extract Twitter/X content using Nitter instances with fallback."""
    if requests is None:
        return "[Requests library not available]"
    
    # List of Nitter instances to try
    nitter_instances = [
        "nitter.net",
        "nitter.it", 
        "nitter.privacydev.net",
        "nitter.fdn.fr",
        "nitter.unixfox.eu"
    ]
    
    # Extract tweet info once
    tweet_info = extract_twitter_info(url)
    if not tweet_info:
        return "[Could not parse Twitter URL format]"
    
    username, tweet_id = tweet_info
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }
    
    last_error = None
    
    # Try each Nitter instance
    for instance in nitter_instances:
        try:
            nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
            response = requests.get(nitter_url, headers=headers, timeout=min(timeout, 8))
            
            if response.status_code == 200:
                content = parse_nitter_content(response.text, url)
                # If we got actual content (not an error message), return it
                if content and not content.startswith("["):
                    return content
                
        except Exception as e:
            last_error = str(e)
            continue
    
    # If all instances failed, return None to trigger Selenium fallback
    return None


def fetch_twitter_hybrid(url: str, timeout: int) -> str:
    """Try cache first, then Nitter, then Selenium fallback."""
    # First check cache
    cached_content = url_cache.get(url)
    if cached_content:
        return cached_content
    
    # Try Nitter (fast and lightweight)
    nitter_result = fetch_twitter_via_nitter(url, timeout)
    if nitter_result is not None:
        # Cache successful result
        url_cache.set(url, nitter_result)
        return nitter_result
    
    # If Nitter failed, try Selenium fallback
    if webdriver is not None:
        selenium_result = fetch_js_content_selenium(url, timeout)
        if selenium_result and not selenium_result.startswith("[Error") and not selenium_result.startswith("[Browser"):
            # Cache successful result
            url_cache.set(url, selenium_result)
            return selenium_result
    
    # Final fallback with clean format
    tweet_info = extract_twitter_info(url)
    if tweet_info:
        username, tweet_id = tweet_info
        fallback_result = f"[Twitter/X link: {username} - tweet {tweet_id}]"
        # Don't cache fallback results as they're not real content
        return fallback_result
    else:
        return f"[Twitter/X link - could not extract details]"


def fetch_js_content_selenium(url: str, timeout: int) -> str:
    """Fetch content using Selenium browser automation."""
    if webdriver is None:
        return "[Browser automation not available - install selenium and webdriver-manager]"
    
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-agent={USER_AGENT}")
        
        service = Service(ChromeDriverManager().install())
        driver = wd.Chrome(service=service, options=options)
        
        try:
            driver.set_page_load_timeout(timeout)
            driver.get(url)
            
            # Extract content based on site type
            if 'twitter.com' in url or 'x.com' in url:
                return extract_twitter_content_selenium(driver, url)
            else:
                # Generic extraction
                title_elem = driver.find_elements(By.TAG_NAME, "title")
                title = title_elem[0].text if title_elem else "No title"
                
                # Try to get main content
                content_selectors = ['[data-testid="tweetText"]', 'article', 'main', '.content', '#content']
                content = ""
                
                for selector in content_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            content = elements[0].text
                            break
                    except:
                        continue
                
                if not content:
                    content = driver.find_element(By.TAG_NAME, "body").text[:500]
                
                return f"Title: {title}\n\nContent: {content}"
                
        finally:
            driver.quit()
            
    except Exception as e:
        return f"[Error fetching content via Selenium: {str(e)}]"


def extract_twitter_content_selenium(driver, url: str) -> str:
    """Extract content from Twitter/X using Selenium."""
    try:
        import time
        time.sleep(2)  # Wait for content to load
        
        # Try to find tweet content
        tweet_selectors = [
            '[data-testid="tweetText"]',
            '[data-testid="tweet"]',
            '.tweet-text',
            '.css-901oao'
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
            except:
                continue
        
        # Extract images and media
        try:
            img_elements = driver.find_elements(By.CSS_SELECTOR, 'img[src*="pbs.twimg.com"]')
            for img in img_elements[:3]:  # Limit to first 3 images
                src = img.get_attribute('src')
                if src and is_tweet_content_image(src):
                    content_parts.append(f"<image: {src}>")
        except:
            pass
        
        if content_parts:
            return "Twitter/X Content:\n\n" + "\n\n".join(content_parts)
        else:
            return "[Twitter content could not be extracted via Selenium]"
            
    except Exception as e:
        return f"[Error extracting Twitter content via Selenium: {str(e)}]"


def extract_twitter_info(url: str) -> tuple:
    """Extract username and tweet ID from Twitter/X URL."""
    try:
        import re
        
        # Handle various Twitter URL formats
        patterns = [
            r'(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)',
            r'(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)\?',
            r'(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)/',
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


def parse_nitter_content(html: str, original_url: str) -> str:
    """Parse tweet content from Nitter HTML."""
    try:
        if bs4 is None:
            return "[BeautifulSoup not available for parsing]"
        
        soup = bs4.BeautifulSoup(html, 'html.parser')
        
        # Find tweet content in Nitter's HTML structure
        tweet_content = []
        
        # Look for tweet text
        tweet_text_elem = soup.find('div', class_='tweet-content')
        if tweet_text_elem:
            # Extract text content, removing unnecessary elements
            for elem in tweet_text_elem.find_all(['a', 'span']):
                if elem.get('class') and 'invisible' in elem.get('class'):
                    elem.decompose()
            
            text = tweet_text_elem.get_text(strip=True)
            if text:
                tweet_content.append(f"Tweet Text:\n{text}")
        
        # Look for media/images
        media_elements = soup.find_all('img', class_='attachment')
        if media_elements:
            for i, img in enumerate(media_elements[:3], 1):  # Limit to 3 images
                alt_text = img.get('alt', '')
                if alt_text and 'attachment' not in alt_text.lower():
                    tweet_content.append(f"Image {i}: {alt_text}")
                else:
                    tweet_content.append(f"<image: attachment {i}>")
        
        # Look for quoted tweets or replies
        quote_elem = soup.find('div', class_='quote')
        if quote_elem:
            quote_text = quote_elem.get_text(strip=True)
            if quote_text:
                tweet_content.append(f"Quoted Tweet:\n{quote_text}")
        
        if tweet_content:
            result = "Twitter/X Content:\n\n" + "\n\n".join(tweet_content)
            return result
        else:
            return "[No tweet content found in Nitter response]"
            
    except Exception as e:
        return f"[Error parsing Nitter content: {str(e)}]"


def is_tweet_content_image(img_url: str) -> bool:
    """Check if image is tweet content (not profile pic)."""
    return 'profile_images' not in img_url.lower()


def fetch_url_content(url: str, timeout: int) -> str:
    """Fetch and extract readable text from a URL with caching."""
    if requests is None:
        return "[HTTP requests not available]"
    
    # Check cache first
    cached_content = url_cache.get(url)
    if cached_content:
        return cached_content
    
    try:
        headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        
        if resp.status_code >= 400:
            error_result = f"[HTTP {resp.status_code} error]"
            # Don't cache HTTP errors
            return error_result
        
        # Check content type to avoid processing binary files
        content_type = resp.headers.get('content-type', '').lower()
        if any(binary_type in content_type for binary_type in ['image/', 'video/', 'audio/', 'application/octet-stream']):
            binary_result = f"[Binary content: {content_type}]"
            url_cache.set(url, binary_result)
            return binary_result
        
        html = resp.content.decode("utf-8", errors="replace")
        
        # Try trafilatura first (better extraction)
        if trafilatura is not None:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
            if extracted:
                meta = trafilatura.extract_metadata(html)
                title = meta.title if meta else None
                content = extracted.strip()
                if title:
                    return f"Title: {title}\n\n{content}"
                return content
        
        # Fallback to BeautifulSoup
        if bs4 is None:
            return "[Content extraction libraries not available]"
        
        soup = bs4.BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract title
        title = soup.find("title")
        title_text = title.get_text().strip() if title else "No title"
        
        # Extract main content
        content_tags = soup.find_all(["p", "div", "article", "main"])
        content_texts = [tag.get_text().strip() for tag in content_tags if tag.get_text().strip()]
        
        if content_texts:
            content = "\n".join(content_texts[:10])  # Limit to first 10 paragraphs
            result = f"Title: {title_text}\n\n{content}"
            url_cache.set(url, result)
            return result
        else:
            result = f"Title: {title_text}\n\n[No readable content found]"
            url_cache.set(url, result)
            return result
    
    except Exception as e:
        error_result = f"[Error fetching content: {str(e)}]"
        # Don't cache errors
        return error_result


class ChunkProcessor:
    """Processes chunks with URL expansion and image processing."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "chats-ir-chunked"
        self.output_dir = self.base_dir / "chats-processed"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [Thread-%(thread)d] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_all_chunks(self, timeout: int = 15, max_workers: int = 4) -> None:
        """Process all chunks from chats-ir-chunked and save to chats-processed."""
        
        if not self.input_dir.exists():
            self.logger.error(f"❌ Input directory not found: {self.input_dir}")
            return
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Find all chat directories
        chat_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not chat_dirs:
            self.logger.warning(f"⚠️ No chat directories found in {self.input_dir}")
            return
        
        self.logger.info(f"🚀 Processing {len(chat_dirs)} chat directories with {max_workers} workers")
        
        total_processed = 0
        
        # Process each chat directory
        for chat_dir in chat_dirs:
            chat_name = chat_dir.name
            self.logger.info(f"📁 Processing chat: {chat_name}")
            
            output_chat_dir = self.output_dir / chat_name
            output_chat_dir.mkdir(exist_ok=True)
            
            # Copy context file
            context_file = chat_dir / "context.txt"
            if context_file.exists():
                shutil.copy2(context_file, output_chat_dir / "context.txt")
                self.logger.info(f"📄 Copied context file for {chat_name}")
            
            # Find all chunk files
            chunk_files = sorted([f for f in chat_dir.glob("chunk_*.txt")])
            
            if not chunk_files:
                self.logger.warning(f"⚠️ No chunk files found in {chat_dir}")
                continue
            
            self.logger.info(f"📊 Found {len(chunk_files)} chunks in {chat_name}")
            
            # Process chunks in parallel
            processed_chunks = self._process_chunks_parallel(
                chunk_files, output_chat_dir, timeout, max_workers, chat_name
            )
            
            total_processed += processed_chunks
            self.logger.info(f"✅ Completed {chat_name}: {processed_chunks} chunks processed")
        
        self.logger.info(f"🎉 Processing complete! {total_processed} total chunks processed")
    
    def process_single_chat(self, chat_name: str, timeout: int, max_workers: int):
        """Process chunks for a single chat directory."""
        self.logger.info(f"🎯 Processing single chat: {chat_name}")
        
        input_chat_dir = self.input_dir / chat_name
        output_chat_dir = self.output_dir / chat_name
        
        if not input_chat_dir.exists():
            self.logger.error(f"❌ Chat directory not found: {input_chat_dir}")
            return
        
        # Create output directory
        output_chat_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy context file if it exists
        context_src = input_chat_dir / "context.txt"
        context_dst = output_chat_dir / "context.txt"
        if context_src.exists():
            shutil.copy2(context_src, context_dst)
            self.logger.info(f"📄 Copied context file for {chat_name}")
        
        # Get chunk files
        chunk_files = sorted([f for f in input_chat_dir.glob("chunk_*.txt")])
        
        if not chunk_files:
            self.logger.warning(f"⚠️ No chunk files found in {input_chat_dir}")
            return
        
        self.logger.info(f"📊 Found {len(chunk_files)} chunks in {chat_name}")
        
        # Process chunks in parallel
        processed_chunks = self._process_chunks_parallel(
            chunk_files, output_chat_dir, timeout, max_workers, chat_name
        )
        
        self.logger.info(f"✅ Completed {chat_name}: {processed_chunks} chunks processed")
        
        # Show cache statistics
        cache_stats = url_cache.stats()
        if cache_stats["enabled"]:
            self.logger.info(f"📊 Cache stats: {cache_stats['cached_urls']} URLs cached ({cache_stats['file_size_human']})")
    
    def _process_chunks_parallel(
        self, 
        chunk_files: list, 
        output_dir: Path, 
        timeout: int, 
        max_workers: int,
        chat_name: str
    ) -> int:
        """Process chunk files in parallel."""
        
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._process_single_chunk, 
                    chunk_file, 
                    output_dir, 
                    timeout,
                    chat_name
                ): chunk_file 
                for chunk_file in chunk_files
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_file = future_to_chunk[future]
                chunk_name = chunk_file.name
                
                try:
                    success = future.result()
                    if success:
                        processed_count += 1
                        if processed_count % 50 == 0:  # Progress every 50 chunks
                            self.logger.info(f"📈 {chat_name}: {processed_count}/{len(chunk_files)} chunks processed")
                except Exception as e:
                    self.logger.error(f"❌ Error processing {chunk_name}: {e}")
        
        return processed_count
    
    def _process_single_chunk(
        self, 
        chunk_file: Path, 
        output_dir: Path, 
        timeout: int,
        chat_name: str
    ) -> bool:
        """Process a single chunk file."""
        
        try:
            # Read chunk content
            with open(chunk_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = [line.rstrip() + '\n' for line in f if line.strip()]
            
            if not lines:
                # Empty chunk, just copy as-is
                output_file = output_dir / chunk_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("")
                return True
            
            # Process the chunk content using our existing process_chunk function
            # We need to adapt it since it expects different parameters
            processed_lines = self._process_chunk_content(lines, timeout)
            
            # Write processed content
            output_file = output_dir / chunk_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(processed_lines)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process {chunk_file.name}: {e}")
            return False
    
    def _process_chunk_content(self, lines: list, timeout: int) -> list:
        """Process chunk content - expand URLs and process images."""
        
        processed_lines = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Check for URLs in the line
            urls = [match.group("url").rstrip(").,") for match in URL_RE.finditer(line)]
            
            if urls:
                processed_lines.append(line)
                
                for url in urls:
                    try:
                        # Determine how to process the URL
                        if any(domain in url.lower() for domain in ['twitter.com', 'x.com', 'facebook.com', 'instagram.com']):
                            content = fetch_js_content(url, timeout)
                        else:
                            content = fetch_url_content(url, timeout)
                        
                        if content:
                            processed_lines.append(f"--- URL Content from {url} ---\n{content}\n--- End URL Content ---\n")
                        else:
                            processed_lines.append(f"--- URL {url} (failed to fetch) ---\n")
                            
                    except Exception as e:
                        processed_lines.append(f"--- URL {url} (error: {e}) ---\n")
            else:
                processed_lines.append(line)
        
        return processed_lines


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process chunks with URL expansion and image processing")
    parser.add_argument('chat_folder', nargs='?', help='Specific chat folder to process (e.g., facebook-soldegen)')
    parser.add_argument('--base-dir', default='.', help='Base directory containing chats-ir-chunked/')
    parser.add_argument('--timeout', type=int, default=15, help='HTTP timeout in seconds')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verbose logging is already setup in ChunkProcessor.__init__
    
    processor = ChunkProcessor(args.base_dir)
    
    if args.chat_folder:
        processor.process_single_chat(args.chat_folder, args.timeout, args.workers)
    else:
        processor.process_all_chunks(args.timeout, args.workers)


if __name__ == "__main__":
    main()

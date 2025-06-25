import logging
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from playwright_stealth import stealth_sync
from urllib.parse import urlparse
from deep_translator import GoogleTranslator
from .azureAITranslate import AzureTranslation
import re
import time

class CompanyInfoScraper:
    TARGET_KEYWORDS = {"about", "who we are"}
    MAX_URL_DEPTH = 3
    MAX_TRANSLATION_CHAR_LIMIT = 500

    def __init__(self, headless=True, log_file='translation_errors.log'):
        logging.basicConfig(filename=log_file, level=logging.ERROR)
        self.visited_urls = set()
        self.headless = headless

    def normalize_url(self, url):
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        except:
            return url

    def get_url_depth(self, url):
        return urlparse(url).path.count("/")

    def close_cookie_popup(self, page):
        selectors = [
            "button:has-text('Accept')",
            "button[aria-label*='Accept']",
            "button[class*='cookie']"
        ]
        for sel in selectors:
            try:
                button = page.query_selector(sel)
                if button and button.is_visible():
                    button.click(timeout=1000)
                    break
            except:
                continue

    def remove_unwanted_elements(self, page):
        script = """
            ['script', 'style', 'img', 'input'].forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove());
            });
        """
        try:
            page.evaluate(script)
        except:
            pass

    def safe_goto(self, page, url):
        try:
            page.goto(url, timeout=8000, wait_until="domcontentloaded")
            return True
        except PlaywrightTimeout:
            return False

    def scrape_page_text(self, page, url):
        info = {"url": url}
        try:
            startTime = time.time()
            if not self.safe_goto(page, url):
                endTime = time.time()
                info["error"] = "Page load failed"
                print(f"safe_goto fn Elapsed time: {(endTime-startTime):.4f} seconds")
                return info
            startTime = time.time()
            self.close_cookie_popup(page)
            endTime = time.time()
            print(f"close_cookie_popup fn Elapsed time: {(endTime-startTime):.4f} seconds")
            startTime = time.time()
            self.remove_unwanted_elements(page)
            endTime = time.time()
            print(f"remove_unwanted_elements fn Elapsed time: {(endTime-startTime):.4f} seconds")
            body_text = page.inner_text("body")
            info["page_content"] = body_text
        except Exception as e:
            info["error"] = str(e)
        return info

    def is_target_link(self, href):
        try:
            path = urlparse(href).path.lower()
            return any(kw in path for kw in self.TARGET_KEYWORDS)
        except:
            return False

    def summarize_paragraphs(self, text, max_sentences=3):
        paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 80]
        summaries = []
        for para in paras:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            summaries.append(' '.join(sentences[:max_sentences]))
        return summaries

    def detect_and_translate(self, text):
        output = []
        chunks = [text[i:i + self.MAX_TRANSLATION_CHAR_LIMIT] for i in range(0, len(text), self.MAX_TRANSLATION_CHAR_LIMIT)]
        for chunk in chunks:
            try:
                output.append(AzureTranslation(chunk).toEnglish())
            except:
                try:
                    output.append(GoogleTranslator(source='auto', target='en').translate(chunk))
                except Exception as e:
                    logging.error(f"Translation failed: {e}")
                    output.append(chunk)
        return "\n".join(output)

    def scrape_company_info(self, start_url, max_depth=1):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            stealth_sync(page)

            # Block heavy resources
            page.route("**/*.{mp4,webm,ogg,avi,mov,gif,jpg,jpeg,png,bmp,tiff,woff,woff2}", lambda route: route.abort())

            def crawl(url, depth):
                if depth > max_depth:
                    return []
                startTime = time.time()
                norm_url = self.normalize_url(url)
                endTime = time.time()
                print(f"normalize_url fn Elapsed time: {(endTime-startTime):.4f} seconds")
                if norm_url in self.visited_urls or self.get_url_depth(norm_url) > self.MAX_URL_DEPTH:
                    return []

                self.visited_urls.add(norm_url)
                
                startTime = time.time()
                info = self.scrape_page_text(page, norm_url)
                endTime = time.time()
                print(f"scrape_page_text fn Elapsed time: {(endTime-startTime):.4f} seconds")
                results = [info]

                if "page_content" in info:
                    summarized = [info["page_content"]]
                    info["summarized_content"] = summarized
                    startTime = time.time()
                    info["translated_content"] = summarized
                    # [self.detect_and_translate(p) for p in summarized]
                    endTime = time.time()
                    print(f"detect_and_translate fn Elapsed time: {(endTime-startTime):.4f} seconds")

                    try:
                        links = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
                        for link in links:
                            if self.is_target_link(link):
                                results.extend(crawl(link, depth + 1))
                    except Exception as e:
                        logging.error(f"Link extraction error at {url}: {e}")

                return results

            all_info = crawl(start_url, 0)
            browser.close()
            return all_info

    def __del__(self):
        print("CompanyInfoScraper cleaned up.")
 
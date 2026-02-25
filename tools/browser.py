"""Playwright browser interaction tool.

Provides the agent with the ability to interact with web pages:
navigate, extract content, click elements, type text, and take screenshots.
"""

import asyncio
import base64
import os
from typing import Optional

from langchain_core.tools import tool

from agent import config

# ── Playwright lifecycle management ───────────────────────────────────────────

_pw = None
_browser = None
_page = None


async def _ensure_browser():
    """Launch or reuse a persistent browser instance."""
    global _pw, _browser, _page
    if _browser is None or not _browser.is_connected():
        from playwright.async_api import async_playwright

        # Stop any previously leaked Playwright engine before creating a new one
        if _pw is not None:
            await _pw.stop()

        _pw = await async_playwright().start()
        _browser = await _pw.chromium.launch(headless=True)
        context = await _browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        _page = await context.new_page()
    return _page


async def _shutdown_browser():
    """Gracefully tear down the page, browser, and Playwright engine."""
    global _pw, _browser, _page
    if _page is not None:
        await _page.close()
        _page = None
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _pw is not None:
        await _pw.stop()
        _pw = None


def close_browser():
    """Sync wrapper to shut down all Playwright resources. Call on agent exit."""
    _run_async(_shutdown_browser())


def _run_async(coro):
    """Run an async coroutine from a sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop — use a thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ── Tool definitions ──────────────────────────────────────────────────────────


@tool
def browser_navigate(url: str) -> str:
    """Navigate the browser to a URL and return the page title.

    Use this to visit a website. After navigating, you can use other browser
    tools to interact with the page content.

    Args:
        url: The full URL to navigate to (e.g. 'https://example.com')
    """

    async def _navigate():
        page = await _ensure_browser()
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = await page.title()
        status = response.status if response else "unknown"
        return f"Navigated to: {url}\nTitle: {title}\nStatus: {status}"

    return _run_async(_navigate())


@tool
def browser_get_content(selector: Optional[str] = None) -> str:
    """Extract visible text content from the current page.

    Use this after navigating to a page to read its content.

    Args:
        selector: Optional CSS selector to extract text from a specific element.
                  If not provided, extracts the full page body text.
    """

    async def _get_content():
        page = await _ensure_browser()
        if selector:
            element = await page.query_selector(selector)
            if element:
                text = await element.inner_text()
            else:
                text = f"No element found matching selector: {selector}"
        else:
            text = await page.inner_text("body")
        # Truncate very long pages
        if len(text) > 5000:
            text = text[:5000] + "\n\n... [content truncated — use a CSS selector for specific sections]"
        return text

    return _run_async(_get_content())


@tool
def browser_click(selector: str) -> str:
    """Click an element on the current page.

    Args:
        selector: CSS selector for the element to click (e.g. 'button#submit', 'a.nav-link')
    """

    async def _click():
        page = await _ensure_browser()
        try:
            await page.click(selector, timeout=5000)
            await page.wait_for_load_state("domcontentloaded")
            title = await page.title()
            return f"Clicked element '{selector}'. Page title is now: {title}"
        except Exception as e:
            return f"Error clicking '{selector}': {e}"

    return _run_async(_click())


@tool
def browser_type_text(selector: str, text: str) -> str:
    """Type text into an input field on the current page.

    Args:
        selector: CSS selector for the input element (e.g. 'input#search', 'textarea.comment')
        text: The text to type into the element
    """

    async def _type():
        page = await _ensure_browser()
        try:
            await page.fill(selector, text, timeout=5000)
            return f"Typed '{text}' into element '{selector}'"
        except Exception as e:
            return f"Error typing into '{selector}': {e}"

    return _run_async(_type())


@tool
def browser_screenshot() -> str:
    """Take a screenshot of the current page and return its base64 data.

    Use this when you want to visually inspect a page. The screenshot
    is saved inside the sandbox and returned as a base64-encoded PNG string.
    """

    async def _screenshot():
        page = await _ensure_browser()
        screenshots_dir = os.path.join(config.SANDBOX_ROOT, ".screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        path = os.path.join(screenshots_dir, "latest.png")
        await page.screenshot(path=path, full_page=False)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        current_url = page.url
        return (
            f"Screenshot saved to {path}\n"
            f"Current URL: {current_url}\n"
            f"Base64 PNG ({len(b64)} chars):\n{b64[:200]}..."
            if len(b64) > 200
            else f"Screenshot saved to {path}\nCurrent URL: {current_url}\nBase64 PNG:\n{b64}"
        )

    return _run_async(_screenshot())

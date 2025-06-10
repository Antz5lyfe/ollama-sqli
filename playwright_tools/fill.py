from __future__ import annotations

from typing import Optional, Type

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field


class FillToolInput(BaseModel):
    """Input for FillTool."""

    selector: str = Field(..., description="CSS selector for the element to fill")
    value: str = Field(
        "",
        description="Value to fill for the `<input>`, `<textarea>` or `[contenteditable]` element. Note that you can pass an empty string to clear the input field.",
    )


class FillTool(BaseBrowserTool):
    """Tool for Filling on an element with the given CSS selector."""

    name: str = "fill_element"
    description: str = "Fill on an element with the given CSS selector"
    args_schema: Type[BaseModel] = FillToolInput  # type: ignore
    visible_only: bool = False
    """Whether to consider only visible elements."""
    playwright_strict: bool = False
    """Whether to employ Playwright's strict mode when Filling on elements. When true, the call requires selector to resolve to a single element. If given selector resolves to more than one element, the call throws an exception."""
    playwright_timeout: float = 1_000
    """Timeout (in ms) for Playwright to wait for element to be ready."""

    def _selector_effective(self, selector: str) -> str:
        if not self.visible_only:
            return selector
        return f"{selector} >> visible=1"

    def _value_effective(self, value: str) -> str:
        if not self.visible_only:
            return value
        return f"{value}"

    def _run(
        self,
        selector: str,
        value: str = "test",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        value_effective = self._value_effective(value=value)
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            page.fill(
                selector_effective,
                value_effective,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to Fill on element '{selector}'"
        return f"Filled element '{selector}'"

    async def _arun(
        self,
        selector: str,
        value: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        value_effective = self._value_effective(value=value)
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            await page.fill(
                selector_effective,
                value_effective,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to Fill on element '{selector}'.\nselector_effective: {selector_effective}\nvalue_effective: {value_effective}"
        return f"Filled element '{selector}'"

from __future__ import annotations

from playwright.sync_api import Page


class AmanajePage:
    def __init__(self, page: Page):
        self.page = page

    def goto(self, path: str) -> None:
        self.page.goto(path, wait_until="networkidle")

    def heading_text(self, selector: str = "h3") -> str:
        return self.page.locator(selector).first.inner_text()


class UploadPage(AmanajePage):
    def expand_support(self) -> None:
        self.page.locator("#supportSummary").click()


class CreatePage(AmanajePage):
    def expand_support(self) -> None:
        self.page.locator("#supportSummary").click()


class FeaturePage(AmanajePage):
    pass


class TrainingPage(AmanajePage):
    pass


class ProductionPage(AmanajePage):
    pass

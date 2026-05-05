from __future__ import annotations

from playwright.sync_api import Page


class AmanajePage:
    def __init__(self, page: Page):
        self.page = page

    def goto(self, path: str) -> None:
        self.page.goto(path, wait_until="domcontentloaded")
        self.page.locator("main.container").wait_for(state="visible")

    def heading_text(self, selector: str = "h3") -> str:
        return self.page.locator(selector).first.inner_text()


class UploadPage(AmanajePage):
    def expand_support(self) -> None:
        self.page.locator("#supportSummary").click()
        self.page.locator("#supportMatrix .support-card").first.wait_for(state="visible")


class CreatePage(AmanajePage):
    def expand_support(self) -> None:
        self.page.locator("#supportSummary").click()
        self.page.locator("#supportMatrix .support-card").first.wait_for(state="visible")


class FeaturePage(AmanajePage):
    pass


class TrainingPage(AmanajePage):
    pass


class ProductionPage(AmanajePage):
    pass


class AssistantPage(AmanajePage):
    pass


class SettingsPage(AmanajePage):
    pass

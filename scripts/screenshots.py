"""Снимки экрана работающего дашборда для отчёта и презентации (Playwright + установленный Chrome).

Запуск: python scripts/screenshots.py [url] [каталог]
Дашборд должен быть запущен: python -m cryptocast dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "docs" / "screenshots")
TABS = [
    ("01_decision", "Решение на завтра"),
    ("02_series", "Факт и прогноз"),
    ("03_quality", "Качество моделей"),
    ("04_explain", "Объяснение"),
    ("05_monitor", "Мониторинг"),
    ("06_experiments", "Эксперименты"),
    ("07_data", "Данные"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    errors = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 2300}, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=cryptocast", timeout=60000)
        page.wait_for_timeout(4000)
        for name, label in TABS:
            page.get_by_role("tab", name=label).click()
            page.wait_for_timeout(3500)
            exceptions = page.locator('[data-testid="stException"]').count()
            if exceptions:
                errors += exceptions
                print(f"[{label}] ошибок на странице: {exceptions}")
                print(page.locator('[data-testid="stException"]').first.inner_text()[:1500])
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print("снимок", name)
        browser.close()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export the panels of the W&B report as PNG images for the PDF.

The report is rendered by W&B itself in a headless Chromium; each panel is
saved as report/figures/wandb/<panel>.png.

    pip install playwright && python -m playwright install chromium
    python scripts/wandb_screenshots.py --report_url <W&B report URL>

Authentication uses the API key that `wandb login` stored in ~/.netrc (it is
sent only to the W&B server and never printed).
"""
import argparse
import base64
import netrc
import os
import re
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

# Panel title in the W&B report -> output file name used by report/report.tex
NAMES = {"MobileNet-v2 compression sweep": "parallel_coordinates",
         "Accuracy vs. compression ratio": "acc_vs_ratio",
         "Accuracy vs. activation compression ratio": "acc_vs_act_ratio",
         "test_acc during quantization-aware fine-tuning": "qat_test_acc"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report_url", required=True)
    p.add_argument("--api_host", default="api.forge.coreweave.com",
                   help="host of the W&B API entry in ~/.netrc (api.wandb.ai for wandb.ai)")
    p.add_argument("--out_dir", default="report/figures/wandb")
    p.add_argument("--wait_ms", type=int, default=7000, help="render time per panel")
    args = p.parse_args()

    _, _, key = netrc.netrc().authenticators(args.api_host)
    auth = "Basic " + base64.b64encode(f"api:{key}".encode()).decode()
    app = urlparse(args.report_url)

    def add_auth(route, request):
        headers = dict(request.headers)
        headers["authorization"] = auth
        route.continue_(headers=headers)

    os.makedirs(args.out_dir, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=2)
        # The W&B web app talks to <app>/api/...; authenticate those calls with the API key.
        page.route(f"{app.scheme}://{app.netloc}/api/**", add_auth)
        page.goto(args.report_url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(15000)
        # Drop the cookie banner so it does not cover the panels.
        page.evaluate("""() => { for (const e of document.querySelectorAll('div,section,aside')) {
            if (getComputedStyle(e).position === 'fixed' && e.textContent.includes('We use cookies'))
                e.remove(); } }""")
        panels = page.locator("div.panel-bank__panel")
        for i in range(panels.count()):
            panel = panels.nth(i)
            panel.scroll_into_view_if_needed()  # panels render lazily
            page.wait_for_timeout(args.wait_ms)
            page.mouse.move(5, 5)  # keep hover controls out of the image
            titles = panel.locator(".PanelTitle")
            title = titles.first.inner_text().strip() if titles.count() else f"panel{i}"
            name = NAMES.get(title) or re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            panel.screenshot(path=os.path.join(args.out_dir, f"{name}.png"))
            print(f"{title!r} -> {name}.png")
        browser.close()


if __name__ == "__main__":
    main()

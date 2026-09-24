#!/usr/bin/env python
"""Headless-browser check for the built viewer HTML
(results/viewer/c1_cecum_t1_v1_viewer.html): loads the file exactly as a
person would (file:// URL, real Chromium via Playwright, SwiftShader
software WebGL since there's no GPU display on this server), captures
every console message / page error / failed request, exercises the UI
(config/tau/diff/headline toggles, the frame slider, a canvas click), and
compares the JS-computed category counts against
results/viewer/category_counts_python.json (written by
scripts/build_viewer.py directly from the Stage 1 export, independently
of anything the viewer's own JS does).

Requires `pip install playwright && playwright install chromium` (done
once, approved, into scratch/.venv -- see docs/viewer.md).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path("/data1_ycao/chua/projects/mrsp")
HTML_PATH = REPO / "results/viewer/c1_cecum_t1_v1_viewer.html"
COUNTS_PATH = REPO / "results/viewer/category_counts_python.json"
SCREENSHOT_PATH = REPO / "results/viewer/check_viewer_screenshot.png"
REPORT_PATH = REPO / "results/viewer/check_viewer_report.json"


def main():
    python_counts = json.loads(COUNTS_PATH.read_text())

    console_messages = []
    page_errors = []
    failed_requests = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"],
        )
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.on("console", lambda msg: console_messages.append({"type": msg.type, "text": msg.text}))
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on("requestfailed", lambda req: failed_requests.append({"url": req.url, "failure": req.failure}))

        page.goto(f"file://{HTML_PATH}")
        page.wait_for_function("window.__viewerComputeCounts !== undefined", timeout=30000)
        time.sleep(1.0)  # let the initial render settle

        # ---------------- JS counts, from the same code path the page runs ----------------
        js_counts = {}
        for config_name in ["oracle", "pred_depth_only", "pred_pose_only", "fully_predicted"]:
            js_counts[config_name] = {}
            for tau in [0.15, 0.25, 0.35, 0.50]:
                js_counts[config_name][str(tau)] = page.evaluate(
                    "([c, t]) => window.__viewerComputeCounts(c, t)", [config_name, tau]
                )

        # ---------------- exercise the UI ----------------
        page.select_option("#sel-config", "fully_predicted")
        page.select_option("#sel-tau", "0.35")
        time.sleep(0.2)
        page.click("#btn-headline")
        time.sleep(0.2)
        page.click("#btn-diff")
        page.select_option("#sel-config-a", "fully_predicted")
        page.select_option("#sel-config-b", "pred_pose_only")
        time.sleep(0.2)
        page.click("#btn-diff")  # back to single-config mode
        page.select_option("#sel-config", "oracle")
        page.select_option("#sel-tau", "0.25")
        time.sleep(0.2)

        slider = page.locator("#frame-slider")
        slider.fill("100")
        page.dispatch_event("#frame-slider", "input")
        time.sleep(0.2)

        canvas = page.locator("#canvas-wrap canvas")
        box = canvas.bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        time.sleep(0.2)
        region_info_html = page.locator("#region-info").inner_html()

        page.screenshot(path=str(SCREENSHOT_PATH))
        browser.close()

    # ---------------- compare ----------------
    mismatches = []
    for config_name, by_tau in js_counts.items():
        for tau_str, js_c in by_tau.items():
            py_c = python_counts["single_config"][config_name][tau_str]
            if js_c != py_c:
                mismatches.append({"config": config_name, "tau": tau_str, "js": js_c, "python": py_c})

    report = {
        "js_counts": js_counts,
        "python_counts_oracle_tau_0_25": python_counts["single_config"]["oracle"]["0.25"],
        "mismatches": mismatches,
        "counts_match": len(mismatches) == 0,
        "console_messages": console_messages,
        "console_errors": [m for m in console_messages if m["type"] == "error"],
        "page_errors": page_errors,
        "failed_requests": failed_requests,
        "region_info_after_click_nonempty": "Click a face" not in region_info_html,
        "screenshot": str(SCREENSHOT_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"oracle @ tau=0.25 -- JS: {js_counts['oracle']['0.25']}  Python: {python_counts['single_config']['oracle']['0.25']}")
    print(f"counts_match (all 16 config x tau): {report['counts_match']}")
    print(f"console errors: {len(report['console_errors'])}")
    print(f"page errors: {len(page_errors)}")
    print(f"failed requests: {len(failed_requests)}")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {SCREENSHOT_PATH}")

    if mismatches:
        print("*** MISMATCHES ***")
        print(json.dumps(mismatches, indent=2))
        sys.exit(1)
    if page_errors:
        print("*** PAGE ERRORS ***")
        print(json.dumps(page_errors, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()

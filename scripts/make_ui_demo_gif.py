"""Turn a single full-page screenshot of the running dashboard into a
smooth scrolling demo GIF, simulating scrolling through the real page.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "dashboard_check.png"
OUT_GIF = ROOT / "docs" / "demo_ui.gif"

VIEWPORT_HEIGHT = 900
SCROLL_STEP = 14
HOLD_FRAMES_START = 25
HOLD_FRAMES_END = 35
FRAME_MS = 40


def main() -> None:
    img = Image.open(SOURCE).convert("RGB")
    width, full_height = img.size

    if full_height <= VIEWPORT_HEIGHT:
        raise SystemExit("Source image shorter than viewport height; nothing to scroll.")

    max_scroll = full_height - VIEWPORT_HEIGHT
    frames: list[Image.Image] = []
    durations: list[int] = []

    top_frame = img.crop((0, 0, width, VIEWPORT_HEIGHT))
    for _ in range(HOLD_FRAMES_START):
        frames.append(top_frame)
        durations.append(FRAME_MS)

    y = 0
    while y < max_scroll:
        frames.append(img.crop((0, y, width, y + VIEWPORT_HEIGHT)))
        durations.append(FRAME_MS)
        y += SCROLL_STEP

    bottom_frame = img.crop((0, max_scroll, width, max_scroll + VIEWPORT_HEIGHT))
    for _ in range(HOLD_FRAMES_END):
        frames.append(bottom_frame)
        durations.append(FRAME_MS)

    OUT_GIF.parent.mkdir(exist_ok=True)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT_GIF} ({len(frames)} frames, {width}x{VIEWPORT_HEIGHT})")


if __name__ == "__main__":
    main()

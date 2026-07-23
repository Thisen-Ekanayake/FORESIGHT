"""Render the demo frames: a top-down navmesh map with the traced agent path, composited beside the first-person feed.

Coordinate convention (verified against habitat_sim 0.3.3 get_topdown_view): the navmesh raster is indexed
[row, col] with row = (z - z_min) / mpp and col = (x - x_min) / mpp. Both the raster and the path overlay
are derived from this same mapping, so they register exactly.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Top-down palette
FREE = (238, 238, 238)  # navigable floor
BLOCKED = (48, 48, 48)  # walls / off-navmesh
PLANNED = (150, 185, 255)  # full planned geodesic route (drawn thin, underneath)
TRAVELED = (34, 105, 230)  # path actually walked so far (drawn thick, on top)
START_C = (46, 184, 92)  # start marker (green)
GOAL_C = (222, 58, 58)  # goal marker (red)
CURRENT_C = (255, 255, 255)  # current position dot (white with a blue ring)

PANEL_BG = (24, 24, 24)
CAPTION_H = 42


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/home/thisen-ekanayake/miniforge3/envs/habitat/fonts/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _world_xz_to_grid(points_world: np.ndarray, bounds_min: np.ndarray, mpp: float) -> np.ndarray:
    """(N, 3) world points -> (N, 2) fractional grid coords as (col, row) = (x-pixel, y-pixel)."""
    pts = np.atleast_2d(points_world)
    col = (pts[:, 0] - bounds_min[0]) / mpp
    row = (pts[:, 2] - bounds_min[2]) / mpp
    return np.stack([col, row], axis=-1)


def _navmesh_base_image(topdown: np.ndarray) -> Image.Image:
    rgb = np.where(topdown[..., None], np.array(FREE, np.uint8), np.array(BLOCKED, np.uint8))
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def render_topdown_panel(
    topdown: np.ndarray,
    bounds_min: np.ndarray,
    mpp: float,
    planned_path: np.ndarray,
    traveled: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    """Draw the navmesh + planned route + traced path + markers, letterboxed into a (panel_w, panel_h) panel."""
    base = _navmesh_base_image(topdown)
    rows, cols = topdown.shape

    # Scale the raster (nearest-neighbour, so walls stay crisp) to fit the panel, preserving aspect.
    scale = min(panel_w / cols, panel_h / rows)
    disp_w, disp_h = max(1, int(cols * scale)), max(1, int(rows * scale))
    base = base.resize((disp_w, disp_h), Image.NEAREST)

    panel = Image.new("RGB", (panel_w, panel_h), PANEL_BG)
    ox, oy = (panel_w - disp_w) // 2, (panel_h - disp_h) // 2
    panel.paste(base, (ox, oy))

    draw = ImageDraw.Draw(panel)

    def to_px(points_world: np.ndarray):
        grid = _world_xz_to_grid(points_world, bounds_min, mpp)
        return [(float(c) * scale + ox, float(r) * scale + oy) for c, r in grid]

    if planned_path is not None and len(planned_path) >= 2:
        draw.line(to_px(planned_path), fill=PLANNED, width=2, joint="curve")
    if traveled is not None and len(traveled) >= 2:
        draw.line(to_px(traveled), fill=TRAVELED, width=4, joint="curve")

    def marker(world_pt, color, r=6, ring=None):
        (x, y) = to_px(world_pt.reshape(1, 3))[0]
        if ring is not None:
            draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=ring)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    marker(start, START_C)
    marker(goal, GOAL_C)
    if traveled is not None and len(traveled) >= 1:
        marker(traveled[-1], CURRENT_C, r=5, ring=TRAVELED)

    return panel


def _caption(width: int, text: str, font) -> Image.Image:
    bar = Image.new("RGB", (width, CAPTION_H), PANEL_BG)
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (CAPTION_H - th) // 2 - bbox[1]), text, fill=(235, 235, 235), font=font)
    return bar


def compose_frame(
    rgb: np.ndarray,
    topdown_panel: Image.Image,
    left_caption: str,
    right_caption: str,
) -> Image.Image:
    """Stack first-person (left) and top-down (right) panels side by side, each with a caption strip below.

    The first-person feed is scaled to the top-down panel height (preserving its aspect), so a high-res
    capture stays large and sharp rather than being shrunk to the panel width.
    """
    font = _load_font(22)
    fp = Image.fromarray(rgb).convert("RGB")
    panel_w, panel_h = topdown_panel.size

    body_h = panel_h
    fp_w = max(1, round(fp.width * body_h / fp.height))
    fp = fp.resize((fp_w, body_h), Image.LANCZOS)  # LANCZOS: crisp supersampled downscale from full-HD capture

    left = Image.new("RGB", (fp_w, body_h), PANEL_BG)
    left.paste(fp, (0, 0))
    right = Image.new("RGB", (panel_w, body_h), PANEL_BG)
    right.paste(topdown_panel, (0, 0))

    total_w = fp_w + panel_w
    canvas = Image.new("RGB", (total_w, body_h + CAPTION_H), PANEL_BG)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (fp_w, 0))
    canvas.paste(_caption(fp_w, left_caption, font), (0, body_h))
    canvas.paste(_caption(panel_w, right_caption, font), (fp_w, body_h))

    # H.264 needs even dimensions.
    w, h = canvas.size
    if w % 2 or h % 2:
        canvas = canvas.crop((0, 0, w - (w % 2), h - (h % 2)))
    return canvas

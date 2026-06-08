"""Demo: render text-on-path examples and save previews."""
from __future__ import annotations
import sys
import os
from PIL import Image, ImageDraw
# Make project root importable when running from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from Components.TextOnPath import (
    arc_text_layer,
    circle_text_layer,
    curved_text_layer,
    text_on_svg_path_layer,
)

OUT = os.path.join(os.path.dirname(__file__), '..', 'font_previews')
os.makedirs(OUT, exist_ok=True)

img_size = (1080, 1920)
center = (img_size[0] // 2, img_size[1] // 2 - 120)
font_path = "/home/benjamin_graf/.local/share/fonts/mc_thumbnails/barlow-condensed/BarlowCondensed-ExtraBoldItalic.ttf"
if not os.path.isfile(font_path):
    font_path = None

def new_bg(color=(10, 12, 22, 255)):
    return Image.new('RGBA', img_size, color)

# 1) SVG path headline
svg_path = "M 120 520 C 340 380, 740 380, 960 520"
layer_svg = text_on_svg_path_layer(
    "GOTT SPRICHT",
    font_path,
    96,
    svg_path,
    img_size,
    preset="swoosh_headline",
    align="center",
    start_offset="50%",
    stroke=(6, 16, 31, 255),
    fill=(255, 255, 255, 255),
)

# 2) Arc text
layer_arc = arc_text_layer(
    "JESUS RUFT DICH",
    font_path,
    108,
    center,
    radius=420,
    start_angle_deg=-155,
    end_angle_deg=-25,
    image_size=img_size,
    preset="energetic_arc",
    align="center",
    start_offset="50%",
    stroke=(6, 16, 31, 255),
    fill=(255, 255, 255, 255),
)

# 3) Circle badge top/bottom
badge_top = circle_text_layer(
    "LIVE SUNDAY",
    font_path,
    58,
    (540, 1360),
    190,
    img_size,
    outside=True,
    preset="badge_top",
    align="center",
    start_offset="50%",
    stroke=(6, 16, 31, 255),
    fill=(255, 210, 140, 255),
)
badge_bottom = circle_text_layer(
    "MOVE CHURCH",
    font_path,
    58,
    (540, 1360),
    190,
    img_size,
    outside=False,
    preset="badge_bottom",
    align="center",
    start_offset="50%",
    stroke=(6, 16, 31, 255),
    fill=(255, 210, 140, 255),
)

# 4) Free curved text helper
layer_curve = curved_text_layer(
    "DU BIST NICHT ALLEIN",
    font_path,
    74,
    img_size,
    (120, 980),
    (330, 910),
    (740, 910),
    (980, 980),
    preset="swoosh_headline",
    align="center",
    start_offset="50%",
    stroke=(6, 16, 31, 255),
    fill=(255, 245, 230, 255),
)

svg_bg = new_bg((10, 12, 22, 255))
svg_draw = ImageDraw.Draw(svg_bg)
svg_bg.alpha_composite(layer_svg, (0, 0))
svg_bg.alpha_composite(layer_curve, (0, 0))
svg_draw.line([(120, 520), (960, 520)], fill=(255, 255, 255, 26), width=2)
svg_path_out = os.path.join(OUT, 'result_svg_swoosh.png')
svg_bg.save(svg_path_out)

arc_bg = new_bg((14, 7, 10, 255))
arc_draw = ImageDraw.Draw(arc_bg)
arc_bg.alpha_composite(layer_arc, (0, 0))
arc_draw.arc((center[0] - 420, center[1] - 420, center[0] + 420, center[1] + 420), start=205, end=335, fill=(255, 255, 255, 26), width=2)
arc_path_out = os.path.join(OUT, 'result_arc_headline.png')
arc_bg.save(arc_path_out)

badge_bg = new_bg((18, 18, 20, 255))
badge_draw = ImageDraw.Draw(badge_bg)
badge_bg.alpha_composite(badge_top, (0, 0))
badge_bg.alpha_composite(badge_bottom, (0, 0))
badge_draw.ellipse((540 - 190, 1360 - 190, 540 + 190, 1360 + 190), outline=(255, 210, 140, 120), width=3)
badge_path_out = os.path.join(OUT, 'result_circle_badge.png')
badge_bg.save(badge_path_out)

combo_bg = new_bg((10, 12, 22, 255))
combo_draw = ImageDraw.Draw(combo_bg)
combo_bg.alpha_composite(layer_svg, (0, 0))
combo_bg.alpha_composite(layer_arc, (0, 0))
combo_bg.alpha_composite(layer_curve, (0, 0))
combo_bg.alpha_composite(badge_top, (0, 0))
combo_bg.alpha_composite(badge_bottom, (0, 0))
combo_draw.ellipse((540 - 190, 1360 - 190, 540 + 190, 1360 + 190), outline=(255, 210, 140, 120), width=3)
combo_path = os.path.join(OUT, 'text_on_path_demo.png')
combo_bg.save(combo_path)

print('Saved demo →', combo_path)
print('Saved example →', svg_path_out)
print('Saved example →', arc_path_out)
print('Saved example →', badge_path_out)

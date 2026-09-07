"""Build docs/og.png -- the 1200x630 card that Reddit, X, Discord, iMessage
and every other link unfurler shows instead of a bare URL.

Run by hand, NOT from the workflows: the output is static branding, not data,
so regenerating it twice a day would just churn a binary in git for no reason.
Re-run it only when the logo, wordmark or tagline actually change:

    python pipeline/nfl/build_og_image.py

The mark is the same path data as the inline SVG favicon in
dashboard_live.html, rasterised here because og:image cannot be an SVG or a
data: URI -- the scrapers need a real, absolutely-addressed PNG or JPEG.
Colors and the display font are pulled from the same values the page's CSS
uses, so the card and the site it links to look like one product.
"""
import pathlib
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "og.png"

W, H = 1200, 630
SS = 4                     # supersample factor; the fin is all curves and
                           # renders with visible stair-stepping at 1x

# Straight from dashboard_live.html's :root block.
BG      = (0x0E, 0x1A, 0x1C)
TEXT    = (0xED, 0xEF, 0xF2)
DIM     = (0x8F, 0xA6, 0xAB)
AQUA    = (0x00, 0xC2, 0xB8)   # --green, and h1 span's color
ORANGE  = (0xF5, 0x82, 0x1F)   # --amber
LINE    = (0x26, 0x40, 0x47)

FONT_DIR = pathlib.Path("C:/Windows/Fonts")
DISPLAY  = FONT_DIR / "GOTHICB.TTF"    # Century Gothic Bold == CSS --disp
BODY     = FONT_DIR / "segoeui.ttf"
BODY_SB  = FONT_DIR / "seguisb.ttf"


def cubic(p0, p1, p2, p3, n=80):
    """Sample one cubic bezier. Pillow has no path filling, so every curve
    in the source SVG becomes a dense run of points in one big polygon."""
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((
            u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
            u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1],
        ))
    return out


def fin_polygon():
    """The favicon's fin outline, in its own 512x512 viewBox coordinates."""
    pts = [(176, 372)]
    pts += cubic((176, 372), (176, 300), (186, 214), (236, 118))
    pts.append((258, 96))
    pts += cubic((258, 96), (284, 148), (316, 240), (332, 336))
    pts += cubic((332, 336), (306, 372), (246, 380), (176, 372))
    return pts


def fin_crease():
    """The faint darker line down the fin (stroke, not fill)."""
    return cubic((236, 118), (224, 190), (216, 268), (214, 340))


def blend(fg, bg, alpha):
    return tuple(round(f * alpha + b * (1 - alpha)) for f, b in zip(fg, bg))


def draw_mark(d, ox, oy, scale):
    """Place the fin + arrow with its 512-space origin at (ox, oy)."""
    T = lambda p: (ox + p[0] * scale, oy + p[1] * scale)

    d.polygon([T(p) for p in fin_polygon()], fill=AQUA)

    # opacity .35 over aqua in the SVG; flattened here since Pillow's polygon
    # fill has already laid the aqua down opaquely.
    d.line([T(p) for p in fin_crease()],
           fill=blend(BG, AQUA, 0.35), width=max(1, round(6 * scale)), joint="curve")

    arrow_w = max(1, round(10 * scale))
    for seg in [((356, 300), (388, 268)), ((388, 268), (388, 296)), ((388, 268), (360, 268))]:
        a, b = T(seg[0]), T(seg[1])
        d.line([a, b], fill=ORANGE, width=arrow_w)
        # Pillow has no round line caps; discs at the joints do the same job
        # and keep the arrowhead from looking chipped.
        for pt in (a, b):
            r = arrow_w / 2
            d.ellipse([pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r], fill=ORANGE)


def main():
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)

    f_word = ImageFont.truetype(str(DISPLAY), 132 * SS)
    f_lead = ImageFont.truetype(str(BODY_SB), 40 * SS)
    f_sub  = ImageFont.truetype(str(BODY), 30 * SS)
    f_dom  = ImageFont.truetype(str(BODY_SB), 27 * SS)

    # Mark on the left, wordmark and copy on the right.
    mark_scale = (300 * SS) / 284.0        # 284 = the mark's own height in viewBox units
    draw_mark(d, 96 * SS - 176 * mark_scale, 150 * SS - 96 * mark_scale, mark_scale)

    x = 400 * SS
    y = 138 * SS
    # "PHINS" light, "UP" aqua -- same split as the page's <h1><span>.
    d.text((x, y), "PHINS", font=f_word, fill=TEXT)
    phins_w = d.textlength("PHINS", font=f_word)
    d.text((x + phins_w + 24 * SS, y), "UP", font=f_word, fill=AQUA)

    y += 178 * SS
    d.text((x, y), "Model projections for NFL, MLB & NHL", font=f_lead, fill=TEXT)
    y += 60 * SS
    d.text((x, y), "Backtested, carried forward through every completed",
           font=f_sub, fill=DIM)
    y += 42 * SS
    d.text((x, y), "game, and graded in public.", font=f_sub, fill=DIM)

    d.line([(0, (H - 6) * SS), (W * SS, (H - 6) * SS)], fill=AQUA, width=12 * SS)
    d.text((x, (H - 92) * SS), "phinsup.net", font=f_dom, fill=AQUA)

    img.resize((W, H), Image.LANCZOS).save(OUT, "PNG", optimize=True)
    print(f"Built {OUT} -- {W}x{H}, {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()

"""
Team logo loading – uses local assets or generates styled badges.
"""
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from team_colors import TEAM_COLORS

# Map display names to logo filenames
TEAM_LOGO_KEYS = {
    "Red Bull Racing": "redbull",
    "Racing Bulls": "racingbulls",
    "RB": "racingbulls",
    "AlphaTauri": "racingbulls",
    "Ferrari": "ferrari",
    "Mercedes": "mercedes",
    "McLaren": "mclaren",
    "Aston Martin": "astonmartin",
    "Alpine": "alpine",
    "Haas F1 Team": "haas",
    "Williams": "williams",
    "Audi": "audi",
    "Kick Sauber": "audi",
    "Alfa Romeo": "audi",
    "Cadillac": "cadillac",
}

# Short names for badge fallback
TEAM_INITIALS = {
    "Red Bull Racing": "RB",
    "Racing Bulls": "RB",
    "RB": "RB",
    "Ferrari": "SF",
    "Mercedes": "ME",
    "McLaren": "MC",
    "Aston Martin": "AM",
    "Alpine": "AL",
    "Haas F1 Team": "HA",
    "Williams": "WI",
    "Audi": "AU",
    "Cadillac": "CA",
}


def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def create_badge(team_name: str, size: int = 28) -> "Image.Image":
    """Create a shield-shaped badge with team color and initials."""
    if not HAS_PIL:
        return None
    color = TEAM_COLORS.get(team_name, "#555566")
    rgb = _hex_to_rgb(color)
    initial = TEAM_INITIALS.get(team_name, team_name[:2].upper() if team_name else "?")
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    m = 1
    cx, cy = size // 2, size // 2
    pts = [
        (m, m + size // 6),
        (cx, m),
        (size - m - 1, m + size // 6),
        (size - m - 1, cy + size // 6),
        (cx, size - m - 1),
        (m, cy + size // 6),
    ]
    draw.polygon(pts, fill=rgb, outline=None)

    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(8, size // 3))
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), initial, font=font)
    except (AttributeError, TypeError):
        try:
            w, h = draw.textsize(initial, font=font)
            bbox = (0, 0, w, h)
        except Exception:
            bbox = (0, 0, size // 2, size // 2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), initial, fill="#ffffff", font=font)
    return img


def load_logo(team_name: str, size: int = 28) -> "Image.Image":
    """Load team logo from assets or create badge."""
    if not HAS_PIL:
        return None
    base = Path(__file__).parent
    key = TEAM_LOGO_KEYS.get(team_name, team_name.lower().replace(" ", "").replace("f1team", ""))
    for ext in ("png", "jpg", "webp"):
        for folder in ("logos", "assets/logos"):
            path = base / folder / f"{key}.{ext}"
            if path.exists():
                try:
                    img = Image.open(path).convert("RGBA")
                    img = img.resize((size, size), getattr(Image, "Resampling", Image).LANCZOS)
                    return img
                except Exception:
                    pass
    return create_badge(team_name, size)

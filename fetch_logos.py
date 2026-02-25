#!/usr/bin/env python3
"""
Download F1 team logos to logos/ folder.
Run once: python fetch_logos.py
"""
import urllib.request
from pathlib import Path

# URLs for team logos (Wikipedia Commons, public domain/fair use)
# Format: team_key -> URL
LOGO_URLS = {
    "redbull": "https://upload.wikimedia.org/wikipedia/en/6/6e/Red_Bull_Racing_logo.svg",
    "ferrari": "https://upload.wikimedia.org/wikipedia/en/d/d5/Scuderia_Ferrari_logo.svg",
    "mercedes": "https://upload.wikimedia.org/wikipedia/commons/9/90/Mercedes-Logo.svg",
    "mclaren": "https://upload.wikimedia.org/wikipedia/en/6/6b/McLaren_Racing_logo.svg",
    "astonmartin": "https://upload.wikimedia.org/wikipedia/en/5/5a/Aston_Martin_logo.svg",
    "alpine": "https://upload.wikimedia.org/wikipedia/commons/9/9c/Alpine_logo.svg",
    "haas": "https://upload.wikimedia.org/wikipedia/en/6/6e/Haas_F1_Team_logo.svg",
    "williams": "https://upload.wikimedia.org/wikipedia/en/6/6b/Williams_Racing_logo.svg",
    "audi": "https://upload.wikimedia.org/wikipedia/commons/9/92/Audi-Logo_2016.svg",
    "cadillac": "https://upload.wikimedia.org/wikipedia/commons/9/9c/Cadillac_logo.svg",
    "racingbulls": "https://upload.wikimedia.org/wikipedia/en/6/6e/Red_Bull_Racing_logo.svg",  # RB uses similar
}

# SVG not supported by PIL for display - use PNG fallbacks
# These are from Wikipedia Commons PNG versions
PNG_URLS = {
    "redbull": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6e/Red_Bull_Racing_logo.svg/200px-Red_Bull_Racing_logo.svg.png",
    "ferrari": "https://upload.wikimedia.org/wikipedia/en/thumb/d/d5/Scuderia_Ferrari_logo.svg/200px-Scuderia_Ferrari_logo.svg.png",
    "mercedes": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Mercedes-Logo.svg/200px-Mercedes-Logo.svg.png",
    "mclaren": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6b/McLaren_Racing_logo.svg/200px-McLaren_Racing_logo.svg.png",
    "astonmartin": "https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/Aston_Martin_logo.svg/200px-Aston_Martin_logo.svg.png",
    "alpine": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Alpine_logo.svg/200px-Alpine_logo.svg.png",
    "haas": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6e/Haas_F1_Team_logo.svg/200px-Haas_F1_Team_logo.svg.png",
    "williams": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6b/Williams_Racing_logo.svg/200px-Williams_Racing_logo.svg.png",
    "audi": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Audi-Logo_2016.svg/200px-Audi-Logo_2016.svg.png",
    "cadillac": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Cadillac_logo.svg/200px-Cadillac_logo.svg.png",
    "racingbulls": "https://upload.wikimedia.org/wikipedia/en/thumb/6/6e/Red_Bull_Racing_logo.svg/200px-Red_Bull_Racing_logo.svg.png",
}

def main():
    base = Path(__file__).parent
    logos_dir = base / "logos"
    logos_dir.mkdir(exist_ok=True)

    req = urllib.request.Request(
        "https://upload.wikimedia.org/wikipedia/en/thumb/6/6e/Red_Bull_Racing_logo.svg/200px-Red_Bull_Racing_logo.svg.png",
        headers={"User-Agent": "F1 Predictor/1.0"}
    )

    for key, url in PNG_URLS.items():
        path = logos_dir / f"{key}.png"
        if path.exists():
            print(f"  {key}: already exists")
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "F1-Predictor/1.0 (Educational)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                path.write_bytes(resp.read())
            print(f"  {key}: downloaded")
        except Exception as e:
            print(f"  {key}: failed - {e}")

    print("Done. Logos in:", logos_dir)


if __name__ == "__main__":
    main()

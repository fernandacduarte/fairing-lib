"""Extract the purple fairing comparison for the README from the full figure.

Run after examples/04_fairing.py if the source figure has been regenerated:

    python examples/04_readme_elbow.py

The crop keeps the title and first row, without resampling the scientific image.
"""

from pathlib import Path

from PIL import Image


IMG = Path(__file__).resolve().parents[1] / "docs" / "img"


if __name__ == "__main__":
    with Image.open(IMG / "11-elbow-k123.png") as source:
        # Bounds in the committed 1822 x 1100 figure; fail explicitly if its
        # layout changes so the crop can be reviewed instead of clipping a pipe.
        if source.size != (1822, 1100):
            raise ValueError("Source figure dimensions changed; review the README crop bounds.")
        target = IMG / "11-elbow-k123-readme.png"
        source.crop((0, 0, 1740, 570)).save(target)
        print("saved", target)

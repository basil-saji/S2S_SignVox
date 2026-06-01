from pathlib import Path

folders = [
    "notebooks",
    "reports",
    "docs",
    "assets",
    "data",
    "models"
]

files = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",

    "notebooks/sign2sound_phase1_backbone.ipynb",

    "docs/dataset.md",
    "docs/training.md",
    "docs/roadmap.md",

    "data/README.md",
    "models/README.md"
]

for folder in folders:
    Path(folder).mkdir(
        parents=True,
        exist_ok=True
    )

for file in files:
    Path(file).touch(
        exist_ok=True
    )

print("SIGN2SOUND repository structure created successfully.")
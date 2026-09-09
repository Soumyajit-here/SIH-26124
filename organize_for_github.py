from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent


# ============================================================
# PROTECTED DIRECTORIES
# These will NEVER be moved, deleted, or modified by the sorter.
# ============================================================

PROTECTED_DIRS = {
    ".git",
    ".github",
    "backend",
    "frontend",
    "edge_client",
    "docs",
    "scripts",
    "models",
    "tests",
    "demo",
}


# ============================================================
# PROTECTED FILES
# ============================================================

PROTECTED_FILES = {
    ".gitignore",
    "README.md",
    "LICENSE",
    "organize_for_github.py",
}


# ============================================================
# GENERATED OUTPUT DIRECTORIES
#
# These are intentionally NOT moved.
# They can remain locally and are ignored by Git.
# ============================================================

GENERATED_DIRS = {
    "runs",
    "audit",
    "inference_output",
    "prototype_demo_output",
    "rec_video_stress_test_output",
    "video_stress_test_output",
    "whatsapp_video_stress_test_output",
    "demo_replay",
    "edge_output",
    "__pycache__",
    ".pytest_cache",
    "hf_cache",
    "node_modules",
    "dist",
    "build",
}


# ============================================================
# DATASET / ARCHIVE FILES
#
# NEVER automatically move these.
# ============================================================

DATASET_EXTENSIONS = {
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
}


# ============================================================
# DOCUMENTATION
# ============================================================

DOCUMENT_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
}


# ============================================================
# PROJECT SCRIPTS
# ============================================================

SCRIPT_EXTENSIONS = {
    ".py",
    ".ps1",
    ".bat",
    ".cmd",
    ".sh",
}


# ============================================================
# DEMO VIDEO
# ============================================================

DEMO_VIDEO_NAMES = {
    "WhatsApp Video 2026-09-05 at 4.30.48 PM.mp4",
}


# ============================================================
# OTHER VIDEOS
# These are left alone unless specifically identified above.
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# ============================================================
# MODEL FILES
# BOTH .pt FILES ARE INTENTIONALLY INCLUDED
# ============================================================

MODEL_EXTENSIONS = {
    ".pt",
}


# ============================================================
# LARGE FILE THRESHOLD
# Archives/datasets are independently protected above.
# ============================================================

LARGE_FILE_MB = 100


# ============================================================
# HELPERS
# ============================================================

def format_size(path: Path) -> str:
    """Return a human-readable file size."""

    if not path.is_file():
        return "DIRECTORY"

    size = path.stat().st_size

    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"

    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"

    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def size_mb(path: Path) -> float:
    """Return size in MB."""

    if not path.is_file():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def next_available_path(target: Path) -> Path:
    """
    Never overwrite an existing file or directory.

    Example:
        best.pt
        best_1.pt
        best_2.pt
    """

    if not target.exists():
        return target

    parent = target.parent
    stem = target.stem
    suffix = target.suffix

    counter = 1

    while True:

        candidate = (
            parent / f"{stem}_{counter}{suffix}"
        )

        if not candidate.exists():
            return candidate

        counter += 1


# ============================================================
# FILE CLASSIFICATION
# ============================================================

def classify_file(path: Path) -> tuple[str | None, str]:
    """
    Determine whether a file should be moved.

    Returns:
        destination, reason

    destination=None means leave untouched.
    """

    name = path.name
    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # Protected files
    # --------------------------------------------------------

    if name in PROTECTED_FILES:
        return None, "protected file"

    # --------------------------------------------------------
    # Dataset archives
    # --------------------------------------------------------

    if suffix in DATASET_EXTENSIONS:
        return None, "dataset/archive — leave outside repository"

    # --------------------------------------------------------
    # Large files
    # --------------------------------------------------------

    if size_mb(path) >= LARGE_FILE_MB:
        return None, "large file — leave outside repository"

    # --------------------------------------------------------
    # Known demo video
    # --------------------------------------------------------

    if name in DEMO_VIDEO_NAMES:
        return "demo", "selected demonstration video"

    # --------------------------------------------------------
    # Other videos stay where they are
    # --------------------------------------------------------

    if suffix in VIDEO_EXTENSIONS:
        return None, "video — left untouched"

    # --------------------------------------------------------
    # BOTH MODEL CHECKPOINTS
    # --------------------------------------------------------

    if suffix in MODEL_EXTENSIONS:
        return "models", "trained YOLO model checkpoint"

    # --------------------------------------------------------
    # Documentation
    # --------------------------------------------------------

    if suffix in DOCUMENT_EXTENSIONS:
        return "docs", "documentation"

    # --------------------------------------------------------
    # Python / utility scripts
    # --------------------------------------------------------

    if suffix in SCRIPT_EXTENSIONS:
        return "scripts", "project/utility script"

    # --------------------------------------------------------
    # Unknown files
    # --------------------------------------------------------

    return None, "unknown file — left untouched"


# ============================================================
# COLLECT MOVES
# ============================================================

def collect_moves():
    """
    Build a list of files that can safely be moved.

    IMPORTANT:
    No file is modified here.
    """

    moves = []

    for item in ROOT.iterdir():

        # ----------------------------------------------------
        # Directories
        # ----------------------------------------------------

        if item.is_dir():

            if item.name in PROTECTED_DIRS:
                continue

            if item.name in GENERATED_DIRS:
                continue

            # Unknown directories are NEVER moved automatically.
            continue

        # ----------------------------------------------------
        # Files
        # ----------------------------------------------------

        if item.is_file():

            destination, reason = classify_file(item)

            if destination is None:
                continue

            destination_dir = (
                ROOT / destination
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            target = next_available_path(
                destination_dir / item.name
            )

            moves.append(
                (
                    item,
                    target,
                    reason,
                )
            )

    return moves


# ============================================================
# DISPLAY DRY RUN
# ============================================================

def print_plan(moves):

    print("\n")
    print("=" * 90)
    print("SIH26124 — GITHUB PROJECT ORGANIZATION")
    print("=" * 90)

    if not moves:

        print(
            "\nNo files need to be moved automatically."
        )

    else:

        for source, target, reason in moves:

            print("\n" + "-" * 90)

            print(
                f"FROM:\n"
                f"  {source.relative_to(ROOT)}"
            )

            print(
                f"TO:\n"
                f"  {target.relative_to(ROOT)}"
            )

            print(
                f"REASON:\n"
                f"  {reason}"
            )

            print(
                f"SIZE:\n"
                f"  {format_size(source)}"
            )

    print("\n" + "=" * 90)
    print(
        "DRY RUN — NO FILES HAVE BEEN MOVED."
    )
    print("=" * 90)


# ============================================================
# SHOW PROTECTED/IGNORED ITEMS
# ============================================================

def show_left_untouched():

    print("\n")
    print("=" * 90)
    print("ITEMS INTENTIONALLY LEFT UNTOUCHED")
    print("=" * 90)

    for item in sorted(
        ROOT.iterdir(),
        key=lambda p: p.name.lower()
    ):

        if item.name in PROTECTED_DIRS:
            continue

        if item.name in PROTECTED_FILES:
            continue

        # Generated directories
        if (
            item.is_dir()
            and item.name in GENERATED_DIRS
        ):

            print(
                f"\n[GENERATED] {item.name}/"
                "\n  LEFT UNTOUCHED."
                "\n  Should be ignored by Git."
            )

        # Archives
        elif (
            item.is_file()
            and item.suffix.lower()
            in DATASET_EXTENSIONS
        ):

            print(
                f"\n[DATASET / ARCHIVE] {item.name}"
                f"\n  Size: {format_size(item)}"
                "\n  LEFT OUTSIDE REPOSITORY."
            )

        # Remaining videos
        elif (
            item.is_file()
            and item.suffix.lower()
            in VIDEO_EXTENSIONS
        ):

            print(
                f"\n[VIDEO] {item.name}"
                f"\n  Size: {format_size(item)}"
                "\n  LEFT UNTOUCHED."
            )

    print("\n" + "=" * 90)


# ============================================================
# APPLY MOVES
# ============================================================

def apply_moves(moves):

    print("\nApplying file organization...")

    for source, target, reason in moves:

        print("\nMoving:")
        print(
            f"  {source.relative_to(ROOT)}"
        )

        print("→")

        print(
            f"  {target.relative_to(ROOT)}"
        )

        print(
            f"  Reason: {reason}"
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # shutil.move MOVES the item.
        # It does NOT delete it.
        # The file simply changes location.
        shutil.move(
            str(source),
            str(target)
        )


# ============================================================
# CREATE GITIGNORE
# ============================================================

def create_gitignore():

    path = ROOT / ".gitignore"

    if path.exists():

        print(
            "\n.gitignore already exists."
        )

        print(
            "It will NOT be overwritten."
        )

        return

    content = r"""
# ============================================================
# Python
# ============================================================

__pycache__/
*.py[cod]
*.pyo
.pytest_cache/
.mypy_cache/

# Virtual environments
.venv/
venv/
env/

# ============================================================
# Secrets / environment
# ============================================================

.env
.env.*
!.env.example

# ============================================================
# Node / React
# ============================================================

node_modules/
frontend/dist/
frontend/build/

# ============================================================
# IDE
# ============================================================

.vscode/
.idea/

# ============================================================
# OS
# ============================================================

.DS_Store
Thumbs.db

# ============================================================
# Logs / temporary files
# ============================================================

*.log
*.tmp
*.bak
*.swp
~$*

# ============================================================
# Databases
# ============================================================

*.db
*.sqlite
*.sqlite3

# ============================================================
# YOLO generated output
# ============================================================

runs/
wandb/
weights/
checkpoints/

# ============================================================
# Generated project outputs
# ============================================================

audit/
inference_output/
prototype_demo_output/
rec_video_stress_test_output/
video_stress_test_output/
whatsapp_video_stress_test_output/
demo_replay/
edge_output/
hf_cache/

# ============================================================
# Dataset directories
# ============================================================

datasets/
dataset/

# Archive files
*.zip
*.tar
*.tar.gz
*.7z
*.rar

# ============================================================
# Large media
# ============================================================

*.mp4
*.avi
*.mov
*.mkv
*.webm

# ============================================================
# IMPORTANT:
# DO NOT IGNORE .pt MODEL CHECKPOINTS
#
# best.pt and best (model_2).pt are intentionally intended
# to be included in the GitHub repository.
# ============================================================

# *.pt
# *.onnx
# *.engine

# ============================================================
# Notebook checkpoints
# ============================================================

.ipynb_checkpoints/
"""

    path.write_text(
        content.strip() + "\n",
        encoding="utf-8"
    )

    print(
        "\nCreated .gitignore."
    )

    print(
        "IMPORTANT: .pt files are NOT ignored."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Safely organize SIH26124 project for GitHub."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually move approved files. "
            "Without this flag the script only shows a dry run."
        ),
    )

    parser.add_argument(
        "--gitignore",
        action="store_true",
        help=(
            "Create a .gitignore if one does not already exist."
        ),
    )

    args = parser.parse_args()

    print("\n")
    print("=" * 90)
    print("SIH26124 — GITHUB ORGANIZER")
    print("=" * 90)

    print(
        f"\nProject root:\n{ROOT}"
    )

    print(
        "\nSAFETY RULES:"
    )

    print(
        "  • No deletion"
        "\n  • No database modification"
        "\n  • No dataset deletion"
        "\n  • No generated-output deletion"
        "\n  • Existing project directories protected"
        "\n  • Unknown directories untouched"
        "\n  • Existing destination files are never overwritten"
    )

    moves = collect_moves()

    print_plan(moves)

    show_left_untouched()

    # --------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------

    if not args.apply:

        print(
            "\nNothing has been changed."
        )

        print(
            "\nTo actually organize the approved files:"
        )

        print(
            "  python organize_for_github.py --apply --gitignore"
        )

        return

    # --------------------------------------------------------
    # APPLY
    # --------------------------------------------------------

    if moves:

        print(
            "\nYou are about to move the files displayed above."
        )

        print(
            "No files will be deleted."
        )

        confirmation = input(
            "\nType ORGANIZE to continue: "
        ).strip()

        if confirmation != "ORGANIZE":

            print(
                "\nCancelled. No files moved."
            )

            return

        apply_moves(moves)

    else:

        print(
            "\nNo file moves required."
        )

    # --------------------------------------------------------
    # GITIGNORE
    # --------------------------------------------------------

    if args.gitignore:
        create_gitignore()

    print("\n")
    print("=" * 90)
    print("ORGANIZATION COMPLETE")
    print("=" * 90)

    print(
        "\nVerify everything before committing:"
    )

    print(
        "\n  git status"
    )

    print(
        "\nThe two model checkpoints should appear under:"
        "\n  models/best.pt"
        "\n  models/best (model_2).pt"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
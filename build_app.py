#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"


def run(command, *, check=True):
    print("+", " ".join(map(str, command)))
    return subprocess.run(command, cwd=ROOT, check=check)


def require_path(relative, label):
    path = ROOT / relative
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    return path


def add_data(command, relative, destination=None, required=False):
    source = ROOT / relative
    if not source.exists():
        if required:
            raise SystemExit(f"Missing required bundled resource: {source}")
        print(f"Skipping missing optional resource: {source}")
        return
    destination = destination or source.name
    command += ["--add-data", f"{source}{os.pathsep}{destination}"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build and optionally archive a shareable InkNote desktop application.")
    parser.add_argument("--target-arch", choices=("arm64", "x86_64", "universal2"), default=None)
    parser.add_argument("--debug", action="store_true", help="Also build a console executable for crash diagnostics.")
    parser.add_argument("--sign", metavar="IDENTITY", help="Code-sign the resulting .app.")
    parser.add_argument("--zip", action="store_true", help="Create a platform-specific ZIP in dist/.")
    parser.add_argument("--no-clean", action="store_true", help="Keep existing build/dist directories.")
    return parser.parse_args()


def check_dependencies():
    try:
        import PyInstaller
    except ImportError as exc:
        raise SystemExit(
            "PyInstaller is not installed. Run:\n"
            f"  {sys.executable} -m pip install -U pyinstaller"
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed in this Python environment. Run your project's "
            "dependency installation first."
        ) from exc

    try:
        import PySide6
    except ImportError as exc:
        raise SystemExit(
            "PySide6 is not installed in this Python environment. Run:\n"
            f"  {sys.executable} -m pip install -U PySide6"
        ) from exc

    print(f"Python:      {sys.version.split()[0]}")
    print(f"PyInstaller: {PyInstaller.__version__}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"PySide6:     {PySide6.__version__}")


def choose_architecture(requested):
    if requested:
        return requested
    machine = platform.machine().lower()
    return "arm64" if machine in {"arm64", "aarch64"} else "x86_64"


def base_command(architecture):
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        "InkNote",
        "--paths",
        str(ROOT),
        "--exclude-module",
        "PyQt5",
        "--exclude-module",
        "PyQt6",
        "--collect-data",
        "wordfreq",
        "--hidden-import",
        "writing_cnn.data",
        "--hidden-import",
        "writing_cnn.model",
        "--hidden-import",
        "writing_cnn.prediction",
        "--hidden-import",
        "writing_cnn.segmentation",
        "--hidden-import",
        "math_cnn.images",
        "--hidden-import",
        "math_cnn.model",
        "--hidden-import",
        "sympy",
        "--hidden-import",
        "PIL._imaging",
        "--exclude-module",
        "tensorflow",
        "--exclude-module",
        "tensorflow_datasets",
        "--exclude-module",
        "array_record",
        "main.py",
    ]

    if sys.platform == "darwin":
        command[8:8] = ["--target-architecture", architecture]

    add_data(command, "checkpoints", "checkpoints", required=True)
    add_data(command, "assets", "assets")
    return command


def build_app(architecture):
    command = base_command(architecture)
    run(command)
    artifact = DIST / "InkNote.app" if sys.platform == "darwin" else DIST / "InkNote" / ("InkNote.exe" if sys.platform == "win32" else "InkNote")
    if not artifact.exists():
        raise SystemExit(
            f"PyInstaller completed, but the expected application was not found: {artifact}\n"
            f"Contents of dist:\n{chr(10).join(str(p.name) for p in DIST.iterdir()) if DIST.exists() else '(dist missing)'}"
        )
    return artifact


def build_debug(architecture):
    debug_root = ROOT / "dist_debug"
    shutil.rmtree(debug_root, ignore_errors=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--console",
        "--name",
        "InkNote-debug",
        "--distpath",
        str(debug_root),
        "--workpath",
        str(ROOT / "build_debug"),
        "--paths",
        str(ROOT),
        "--exclude-module",
        "PyQt5",
        "--exclude-module",
        "PyQt6",
        "--collect-data",
        "wordfreq",
        "--hidden-import",
        "writing_cnn.data",
        "--hidden-import",
        "writing_cnn.model",
        "--hidden-import",
        "writing_cnn.prediction",
        "--hidden-import",
        "writing_cnn.segmentation",
        "--hidden-import",
        "math_cnn.images",
        "--hidden-import",
        "math_cnn.model",
        "--hidden-import",
        "sympy",
        "--hidden-import",
        "PIL._imaging",
        "--exclude-module",
        "tensorflow",
        "--exclude-module",
        "tensorflow_datasets",
        "--exclude-module",
        "array_record",
        "main.py",
    ]
    if sys.platform == "darwin":
        command[8:8] = ["--target-architecture", architecture]
    add_data(command, "checkpoints", "checkpoints", required=True)
    add_data(command, "assets", "assets")
    run(command)
    executable = "InkNote-debug.exe" if sys.platform == "win32" else "InkNote-debug"
    debug_exe = debug_root / "InkNote-debug" / executable
    if not debug_exe.exists():
        raise SystemExit(f"Debug executable was not created: {debug_exe}")
    return debug_exe


def main():
    args = parse_args()

    require_path("main.py", "main.py")
    require_path("writing_cnn", "writing_cnn package")
    require_path("math_cnn", "math_cnn package")
    require_path("checkpoints", "checkpoints directory")
    check_dependencies()

    architecture = choose_architecture(args.target_arch)
    print(f"Target architecture: {architecture}")

    if not args.no_clean:
        shutil.rmtree(BUILD, ignore_errors=True)
        shutil.rmtree(DIST, ignore_errors=True)
        shutil.rmtree(ROOT / "build_debug", ignore_errors=True)
        shutil.rmtree(ROOT / "dist_debug", ignore_errors=True)

    app = build_app(architecture)
    print(f"Built app: {app}")

    if args.debug:
        debug_exe = build_debug(architecture)
        print(f"Built debug executable: {debug_exe}")

    if args.sign and sys.platform != "darwin":
        raise SystemExit("--sign is only available for macOS builds.")
    if args.sign:
        run(["codesign", "--deep", "--force", "--verbose", "--sign", args.sign, str(app)])

    if args.zip:
        platform_name = {"darwin": "macOS", "win32": "Windows"}.get(sys.platform, "Linux")
        archive_base = DIST / f"InkNote-{platform_name}"
        bundle_name = "InkNote.app" if sys.platform == "darwin" else "InkNote"
        zip_path = shutil.make_archive(str(archive_base), "zip", DIST, bundle_name)
        print(f"ZIP: {zip_path}")

    print()
    print(f"App:   {app}")
    print(f"Run:   {'open ' if sys.platform == 'darwin' else ''}{app}")
    if args.debug:
        print("Debug executable:")
        print(f"  {debug_exe}")
        print("Run debug with:")
        print(f"  {debug_exe}")


if __name__ == "__main__":
    main()

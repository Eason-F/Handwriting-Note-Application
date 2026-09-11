#!/usr/bin/env python3
"""Create an isolated build environment and produce a shareable InkNote ZIP."""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / '.package-venv'


def run(command):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description='Install build dependencies and create a shareable InkNote ZIP.')
    parser.add_argument('--target-arch', choices=('arm64', 'x86_64', 'universal2'))
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sign')
    args = parser.parse_args()
    if not ENVIRONMENT.exists():
        print(f'Creating isolated packaging environment: {ENVIRONMENT}', flush=True)
        venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)
    scripts = 'Scripts' if sys.platform == 'win32' else 'bin'
    python = ENVIRONMENT / scripts / ('python.exe' if sys.platform == 'win32' else 'python')
    run([python, '-m', 'pip', 'install', '--upgrade', 'pip'])
    run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements-app.txt'])
    build_arguments = ['--zip']
    if args.target_arch:
        build_arguments += ['--target-arch', args.target_arch]
    if args.debug:
        build_arguments.append('--debug')
    if args.sign:
        build_arguments += ['--sign', args.sign]
    run([python, ROOT / 'build_app.py', *build_arguments])


if __name__ == '__main__':
    main()

#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 the OpenRAC contributors
#
# Sets up this RTX test checkout on a Mac and starts its launcher ("OpenRAC RTX Test",
# dev.openrac.rtx-test, beside any other OpenRAC launcher):
#
#   tools/setup_rtx_mac.sh            everything, then the launcher
#   tools/setup_rtx_mac.sh --no-run   everything but starting the launcher
#
# It installs what is missing with Homebrew (cmake, ninja, MoltenVK, Node, Python 3.12) and Rust
# with rustup, fetches the decomps, applies OpenRAC's patches to them, configures the port's
# release build and installs the launcher's packages. The launcher then builds the game, asks for
# the disc and bakes the probes (Enhanced graphics) as on any machine.
#
# Ray tracing: Enhanced graphics on, RTX on in its settings. Apple's hardware ray tracing is the
# M3's and later; an M1 or M2 traces in shaders, much slower. Renderer "Vulkan" in the launcher's
# settings is the faster one today; "Metal" draws without MoltenVK but its Enhanced path is slow.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN=1
[[ "${1:-}" == "--no-run" ]] && RUN=0

say() { printf '\n== %s\n' "$*"; }

if ! command -v brew >/dev/null; then
    echo "Homebrew is needed: https://brew.sh" >&2
    exit 1
fi

say "Tools"
missing=()
for formula in cmake ninja molten-vk node python@3.12; do
    brew list --versions "$formula" >/dev/null 2>&1 || missing+=("$formula")
done
if ((${#missing[@]})); then
    brew install "${missing[@]}"
fi
if ! command -v cargo >/dev/null && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$(brew --prefix)/bin:$PATH"
PYTHON="$(brew --prefix python@3.12)/bin/python3.12"
xcode-select -p >/dev/null 2>&1 || {
    echo "The Command Line Tools are needed: xcode-select --install" >&2
    exit 1
}

say "Sources"
git -C "$ROOT" submodule update --init --recursive
"$PYTHON" "$ROOT/tools/sources.py" apply rac1/pal

say "Port build (release)"
cmake -S "$ROOT/port" --preset release -DPython3_EXECUTABLE="$PYTHON"

say "Launcher packages"
cd "$ROOT/launcher"
corepack enable >/dev/null 2>&1 || npm install -g corepack
yarn install

cat <<EOF

Set up. In the launcher:
  1. Settings: Developer on, the repository at $ROOT, Python at $PYTHON.
  2. Add the game (rac1 PAL) with your disc image; the launcher builds it.
  3. Mods: Enhanced graphics on, and bake the probes when it asks (they go to probes-rtx/).
  4. Settings: renderer Vulkan (faster today) or Metal; in the game, RTX on.
Glowing lamps and metal (Clank) need the material maps once the game is set up:
  git clone https://github.com/momo1212221/openrac-graphics-preparation.git port/build/graphics-preparation
  node port/enhanced/tools/sidecars/generate_sidecars.mjs --prep port/build/graphics-preparation \\
      --levels assets/port/rac1/pal --out <install>/active/rac1/data/texture_packs/rac1-pal
EOF

if ((RUN)); then
    say "Starting the launcher"
    exec yarn tauri dev
fi

#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 the OpenRAC contributors
#
# Writes the RTX test's material maps (metal, glow, normals; see
# enhanced/renderer/material_sidecars.h) into the install's texture pack, so the game finds them:
#
#   tools/rtx_material_maps.sh [INSTALL] [generate_sidecars.mjs options, e.g. --force]
#
# INSTALL is the launcher's install folder (its Settings); without it, the RTX test launcher's
# own setting (dev.openrac.rtx-test/launcher.json installDir), else ~/OpenRAC. The maps are data,
# not in git: run this on each machine once the launcher has set the game up (it exports the
# levels into assets/port/rac1/pal), and again after the levels change.
# It clones (or updates) openrac-graphics-preparation into port/build/graphics-preparation.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREP="$ROOT/port/build/graphics-preparation"
LEVELS="$ROOT/assets/port/rac1/pal"
PREP_URL="https://github.com/momo1212221/openrac-graphics-preparation.git"

INSTALL=""
if [[ $# -gt 0 && "$1" != --* ]]; then
    INSTALL="$1"
    shift
fi
if [[ -z "$INSTALL" ]]; then
    CONFIG="$HOME/Library/Application Support/dev.openrac.rtx-test/launcher.json"
    if [[ -f "$CONFIG" ]]; then
        INSTALL="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("installDir") or "")' "$CONFIG" 2>/dev/null || true)"
    fi
fi
INSTALL="${INSTALL:-$HOME/OpenRAC}"
OUT="$INSTALL/active/rac1/data/texture_packs/rac1-pal"

# Node: Homebrew's, or one in ~/.local/node (as some of these Macs have).
export PATH="$HOME/.local/node/bin:$PATH"
command -v brew >/dev/null && export PATH="$(brew --prefix)/bin:$PATH"
if ! command -v node >/dev/null; then
    echo "Node is needed (brew install node)." >&2
    exit 1
fi

if [[ ! -d "$LEVELS" ]] || [[ -z "$(ls -A "$LEVELS" 2>/dev/null)" ]]; then
    echo "No exported levels in $LEVELS yet: add the game (rac1 PAL) in the launcher first." >&2
    exit 1
fi
if [[ ! -d "$INSTALL/active/rac1/data" ]]; then
    echo "No game data in $INSTALL/active/rac1/data: is $INSTALL the launcher's install folder?" >&2
    exit 1
fi

if [[ -d "$PREP/.git" ]]; then
    git -C "$PREP" pull --ff-only --quiet || echo "warning: could not update $PREP; using it as it is" >&2
else
    git clone --quiet "$PREP_URL" "$PREP"
fi

echo "Material maps for $LEVELS into $OUT"
node "$ROOT/port/enhanced/tools/sidecars/generate_sidecars.mjs" \
    --prep "$PREP" --levels "$LEVELS" --out "$OUT" "$@"
echo "Done. With texture_pack = 1 the game's log says \"material sidecars: N texture(s) with maps\"."

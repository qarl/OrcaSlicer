#!/usr/bin/env bash
# Regenerate the vendored standalone amplification core at src/libslic3r/PrintMan/Core/.
# The core is host-agnostic, header-only, standard C++ (no pxr/Eigen/TBB/libslic3r). It is copied from
# the standalone repo (github.com/qarl/printman), then its internal includes are RENAMED from
# "printman/" to "PrintMan/Core/" so the copy lives under the fork's own PrintMan/ tree without the
# lowercase "printman/" prefix colliding (case-insensitively) with it. Do NOT hand-edit the copy;
# change it in the standalone and rerun this.
set -euo pipefail
STANDALONE="${1:-$HOME/src/printman}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/src/libslic3r/PrintMan/Core"
HEADERS="amplify.hpp band_usd.hpp geom.hpp palette.hpp shader.hpp slicer.hpp subdiv.hpp usd.hpp"
mkdir -p "$DEST"
for h in $HEADERS; do
    sed 's|#include "printman/|#include "PrintMan/Core/|g' "$STANDALONE/include/printman/$h" > "$DEST/$h"
done
(cd "$STANDALONE" && git rev-parse HEAD) > "$DEST/SOURCE_COMMIT"
echo "vendored + renamed $HEADERS into PrintMan/Core from $STANDALONE @ $(cd "$STANDALONE" && git rev-parse --short HEAD)"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_BINARY=false
SKIP_OPT=false
for arg in "$@"; do
    case "$arg" in
        --skip-binary) SKIP_BINARY=true ;;
        --skip-opt) SKIP_OPT=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

check_wasm32_support() {
    local cc="${CC:-clang}"
    if ! echo "int main(){return 0;}" | \
        "$cc" -target wasm32-unknown-unknown -c -o /dev/null -x c - 2>/dev/null; then
        echo "Error: '$cc' does not support the wasm32-unknown-unknown target." >&2
        echo "Install an LLVM/clang toolchain with wasm backend support (e.g. 'sudo apt-get install llvm' on Debian/Ubuntu)." >&2
        exit 1
    fi
    if ! command -v wasm-pack >/dev/null 2>&1; then
        echo "Error: wasm-pack not found. Install with: cargo install wasm-pack" >&2
        exit 1
    fi
}

echo "Building WASM library..."
(cd "$SCRIPT_DIR/library" && npm install && npm run build)

if [ "$SKIP_BINARY" = false ]; then
    echo "Checking wasm build prerequisites..."
    check_wasm32_support

    echo "Building WASM binary..."
    rm -rf "$SCRIPT_DIR/pkg"   # start clean so stale wasm-bindgen snippets don't accumulate
    (cd "$SCRIPT_DIR" && wasm-pack build --target web --profile wasm --no-opt)

    # wasm-bindgen is invoked directly so we can pass --keep-lld-exports,
    # which preserves the LLD symbols that loadable extensions import.
    # wasm-pack cannot forward that flag (rustwasm/wasm-pack#1092).
    echo "Re-running wasm-bindgen with --keep-lld-exports..."
    WASM_BINDGEN="$(find "$HOME/Library/Caches/.wasm-pack" "$HOME/.cache/.wasm-pack" -name wasm-bindgen -type f 2>/dev/null | sort -V | tail -1 || true)"
    if [ -z "$WASM_BINDGEN" ]; then
        echo "Error: could not locate wasm-pack's cached wasm-bindgen." >&2
        exit 1
    fi
    "$WASM_BINDGEN" \
        --target web \
        --keep-lld-exports \
        --out-dir "$SCRIPT_DIR/pkg" \
        "$REPO_ROOT/target/wasm32-unknown-unknown/wasm/ggsql_wasm.wasm"

    if [ "$SKIP_OPT" = false ]; then
        echo "Optimising WASM binary..."
        (cd "$SCRIPT_DIR" && wasm-opt pkg/ggsql_wasm_bg.wasm -o pkg/ggsql_wasm_bg.wasm -Oz --all-features)
    else
        echo "Skipping wasm-opt (--skip-opt)."
    fi

    echo "Adding snippets/ to package files..."
    (cd "$SCRIPT_DIR/pkg" && npm pkg set 'files[]=snippets/')
else
    echo "Skipping WASM binary build (--skip-binary)."
fi

SPATIALITE_TAG="spatialite-5.1.0-wasm"
SPATIALITE_URL="https://github.com/ggsql-dev/sqlite-wasm-rs/releases/download/$SPATIALITE_TAG/mod_spatialite.wasm"

# SPATIALITE_WASM overrides the download with a locally built binary.
if [ -n "${SPATIALITE_WASM:-}" ]; then
    echo "Using local mod_spatialite.wasm: $SPATIALITE_WASM"
    cp "$SPATIALITE_WASM" "$SCRIPT_DIR/pkg/mod_spatialite.wasm"
else
    CACHED="$REPO_ROOT/target/wasm-extensions/$SPATIALITE_TAG/mod_spatialite.wasm"
    if [ ! -f "$CACHED" ]; then
        echo "Downloading mod_spatialite.wasm ($SPATIALITE_TAG)..."
        mkdir -p "$(dirname "$CACHED")"
        curl -sSfL -o "$CACHED.tmp" "$SPATIALITE_URL"
        mv "$CACHED.tmp" "$CACHED"
    else
        echo "Using cached mod_spatialite.wasm: $CACHED"
    fi
    cp "$CACHED" "$SCRIPT_DIR/pkg/mod_spatialite.wasm"
fi

echo "Building WASM demo and Quarto integration..."
(cd "$SCRIPT_DIR/demo" && npm install && npm run build)

echo "Copying output to doc/wasm..."
rm -rf "$REPO_ROOT/doc/wasm"
cp -r "$SCRIPT_DIR/demo/dist" "$REPO_ROOT/doc/wasm"

echo "Done! Output is in: $REPO_ROOT/doc/wasm"

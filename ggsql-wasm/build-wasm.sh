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

check_prerequisites() {
    local cc="${CC:-clang}"
    if ! echo "int main(){return 0;}" | \
        "$cc" -target wasm32-unknown-unknown -c -o /dev/null -x c - 2>/dev/null; then
        echo "Error: '$cc' does not support the wasm32-unknown-unknown target." >&2
        echo "Install an LLVM/clang toolchain with wasm backend support (e.g. 'sudo apt-get install llvm' on Debian/Ubuntu)." >&2
        exit 1
    fi

    if ! command -v wasm-opt >/dev/null 2>&1; then
        echo "Error: wasm-opt not found. Install binaryen or run: cargo install wasm-opt" >&2
        exit 1
    fi

    local expected_version
    expected_version="$(awk '/^name = "wasm-bindgen"$/{f=1;next} f&&/^version = /{gsub(/[",]/,"");print $3;exit}' "$REPO_ROOT/Cargo.lock")"
    if [ -z "$expected_version" ]; then
        echo "Error: could not find the wasm-bindgen version in Cargo.lock." >&2
        exit 1
    fi
    if ! command -v wasm-bindgen >/dev/null 2>&1 || \
        [ "$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')" != "$expected_version" ]; then
        echo "Error: wasm-bindgen $expected_version is required." >&2
        echo "Install it with: cargo install -f wasm-bindgen-cli --version $expected_version" >&2
        exit 1
    fi
    echo "Using wasm-bindgen $expected_version"
}

echo "Checking wasm build prerequisites..."
check_prerequisites

if [ "$SKIP_BINARY" = false ]; then
    echo "Building WASM binary..."
    (cd "$SCRIPT_DIR" && cargo build \
        --target wasm32-unknown-unknown \
        --profile wasm \
        -p ggsql-wasm)

    rm -rf "$SCRIPT_DIR/pkg/dist"
    wasm-bindgen \
        --target web \
        --keep-lld-exports \
        --out-dir "$SCRIPT_DIR/pkg/dist" \
        "$REPO_ROOT/target/wasm32-unknown-unknown/wasm/ggsql_wasm.wasm"

    if [ "$SKIP_OPT" = false ]; then
        echo "Optimising WASM binary..."
        wasm-opt \
            "$SCRIPT_DIR/pkg/dist/ggsql_wasm_bg.wasm" \
            -o "$SCRIPT_DIR/pkg/dist/ggsql_wasm_bg.wasm" \
            -Oz \
            --enable-bulk-memory \
            --enable-nontrapping-float-to-int \
            --enable-reference-types \
            --enable-sign-ext \
            --enable-mutable-globals \
            --enable-multivalue
    else
        echo "Skipping wasm-opt (--skip-opt)."
    fi
else
    echo "Skipping WASM binary build (--skip-binary)."
    if [ ! -f "$SCRIPT_DIR/pkg/dist/ggsql_wasm_bg.wasm" ]; then
        echo "Error: --skip-binary needs pkg/dist/ggsql_wasm_bg.wasm; run once without it first." >&2
        exit 1
    fi
fi

echo "Building npm package..."
(cd "$SCRIPT_DIR/pkg" && npm install && npm run build)

SPATIALITE_TAG="spatialite-5.1.0-wasm"
SPATIALITE_URL="https://github.com/ggsql-dev/sqlite-wasm-rs/releases/download/$SPATIALITE_TAG/mod_spatialite.wasm"

# SPATIALITE_WASM overrides the download with a locally built binary.
if [ -n "${SPATIALITE_WASM:-}" ]; then
    echo "Using local mod_spatialite.wasm: $SPATIALITE_WASM"
    SPATIALITE_SOURCE="$SPATIALITE_WASM"
else
    SPATIALITE_SOURCE="$REPO_ROOT/target/wasm-extensions/$SPATIALITE_TAG/mod_spatialite.wasm"
    if [ ! -f "$SPATIALITE_SOURCE" ]; then
        echo "Downloading mod_spatialite.wasm ($SPATIALITE_TAG)..."
        mkdir -p "$(dirname "$SPATIALITE_SOURCE")"
        curl -sSfL -o "$SPATIALITE_SOURCE.tmp" "$SPATIALITE_URL"
        mv "$SPATIALITE_SOURCE.tmp" "$SPATIALITE_SOURCE"
    else
        echo "Using cached mod_spatialite.wasm: $SPATIALITE_SOURCE"
    fi
fi

echo "Building WASM demo and Quarto integration..."
(cd "$SCRIPT_DIR/demo" && npm install && npm run build)
cp "$SPATIALITE_SOURCE" "$SCRIPT_DIR/demo/dist/mod_spatialite.wasm"

echo "Copying output to doc/wasm..."
rm -rf "$REPO_ROOT/doc/wasm"
cp -r "$SCRIPT_DIR/demo/dist" "$REPO_ROOT/doc/wasm"

echo "Done! Output is in: $REPO_ROOT/doc/wasm"

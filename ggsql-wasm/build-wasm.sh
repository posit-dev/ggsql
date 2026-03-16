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
}

echo "Building WASM library..."
(cd "$SCRIPT_DIR/library" && npm install && npm run build)

if [ "$SKIP_BINARY" = false ]; then
    echo "Checking wasm32 compiler support..."
    check_wasm32_support

    echo "Building WASM binary..."
    (cd "$SCRIPT_DIR" && wasm-pack build --target web --profile wasm --no-opt)

    if [ "$SKIP_OPT" = false ]; then
        echo "Optimising WASM binary..."
        (cd "$SCRIPT_DIR" && wasm-opt pkg/ggsql_wasm_bg.wasm -o pkg/ggsql_wasm_bg.wasm -Oz --all-features)
    else
        echo "Skipping wasm-opt (--skip-opt)."
    fi
else
    echo "Skipping WASM binary build (--skip-binary)."
fi

echo "Building WASM demo and Quarto integration..."
(cd "$SCRIPT_DIR/demo" && npm install && npm run build)

echo "Copying output to doc/wasm..."
rm -rf "$REPO_ROOT/doc/wasm"
cp -r "$SCRIPT_DIR/demo/dist" "$REPO_ROOT/doc/wasm"

echo "Done! Output is in: $REPO_ROOT/doc/wasm"

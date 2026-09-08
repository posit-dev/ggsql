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
    # The schema versions have to agree exactly, so pick the cached binary
    # matching the wasm-bindgen the crate was built against rather than the
    # newest one on the machine.
    WB_VERSION="$(awk '/^name = "wasm-bindgen"$/{f=1;next} f&&/^version = /{gsub(/[",]/,"");print $3;exit}' "$REPO_ROOT/Cargo.lock")"
    WASM_BINDGEN=""
    if [ -n "$WB_VERSION" ]; then
        WASM_BINDGEN="$(find "$HOME/Library/Caches/.wasm-pack" "$HOME/.cache/.wasm-pack" \
            -path "*wasm-bindgen-cargo-install-$WB_VERSION/wasm-bindgen" -type f 2>/dev/null | head -1 || true)"
    fi
    if [ -z "$WASM_BINDGEN" ]; then
        echo "Error: no cached wasm-bindgen ${WB_VERSION:-(version unknown)} found." >&2
        echo "Install it with: cargo install -f wasm-bindgen-cli --version $WB_VERSION" >&2
        exit 1
    fi
    echo "Using wasm-bindgen $WB_VERSION"
    "$WASM_BINDGEN" \
        --target web \
        --keep-lld-exports \
        --out-dir "$SCRIPT_DIR/pkg" \
        "$REPO_ROOT/target/wasm32-unknown-unknown/wasm/ggsql_wasm.wasm"

    if [ "$SKIP_OPT" = false ]; then
        echo "Optimising WASM binary..."
        # The features rustc actually emits, named one by one. wasm-opt rejects
        # them unless told to expect them, and `--all-features` is not the
        # shortcut: it enables post-MVP proposals browsers still reject.
        (cd "$SCRIPT_DIR" && wasm-opt pkg/ggsql_wasm_bg.wasm -o pkg/ggsql_wasm_bg.wasm -Oz \
            --enable-bulk-memory \
            --enable-nontrapping-float-to-int \
            --enable-reference-types \
            --enable-sign-ext \
            --enable-mutable-globals \
            --enable-multivalue)
    else
        echo "Skipping wasm-opt (--skip-opt)."
    fi

else
    echo "Skipping WASM binary build (--skip-binary)."
    if [ ! -f "$SCRIPT_DIR/pkg/ggsql_wasm_bg.wasm" ]; then
        echo "Error: --skip-binary needs an existing pkg/; run once without it first." >&2
        exit 1
    fi
fi

# Outside the --skip-binary guard on purpose. These are copies, not a build,
# and `--skip-binary` means "don't recompile the wasm" — a wrapper edit has to
# reach the package either way, or you debug a fix that was never in the
# bundle. Cheap and idempotent, so running it every time costs nothing.
#
# The hand-written wrapper is the package entry point, not wasm-pack's
# generated glue: it owns the resize and font wiring a page actually needs.
# Both live at './' so the import path is the same here and once published.
echo "Adding wrapper, fonts and snippets to the package..."
cp "$SCRIPT_DIR/js/ggsql.js" "$SCRIPT_DIR/pkg/ggsql.js"
cp "$SCRIPT_DIR/js/ggsql.d.ts" "$SCRIPT_DIR/pkg/ggsql.d.ts"
mkdir -p "$SCRIPT_DIR/pkg/fonts"
cp "$SCRIPT_DIR/fonts/roboto-"*.ttf "$SCRIPT_DIR/fonts/OFL-Roboto.txt" "$SCRIPT_DIR/pkg/fonts/"
# Guarded because `files[]=` *appends*: run twice over one package.json and the
# file list accumulates duplicates. wasm-pack rewrites package.json on every
# full build, so "does `main` already point at the wrapper" is exactly the
# question of whether this has been done to *this* package.json.
if [ "$(cd "$SCRIPT_DIR/pkg" && npm pkg get main)" != '"ggsql.js"' ]; then
    (
        cd "$SCRIPT_DIR/pkg"
        npm pkg set 'files[]=snippets/'
        npm pkg set 'files[]=ggsql.js'
        npm pkg set 'files[]=ggsql.d.ts'
        npm pkg set 'types=ggsql.d.ts'
        npm pkg set 'files[]=fonts/'
        npm pkg set 'main=ggsql.js'
        npm pkg set 'module=ggsql.js'
        npm pkg set 'exports[.]=./ggsql.js'
        npm pkg set 'exports[./fonts/*]=./fonts/*'
    )
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

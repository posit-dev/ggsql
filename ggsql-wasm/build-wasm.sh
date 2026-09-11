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

# Each check guards one step, and is called only when that step is going to
# run: `--skip-binary` is for iterating on TypeScript and the demo, so it must
# not demand the wasm toolchain, and `--skip-opt` must not demand wasm-opt.
supports_wasm() {
    echo "int main(){return 0;}" | \
        "$1" -target wasm32-unknown-unknown -c -o /dev/null -x c - 2>/dev/null
}

# Homebrew's clang, if it is installed and can target wasm. `brew --prefix
# llvm` answers with the path the formula would occupy whether or not it is
# there, so the executable itself is the test.
brew_llvm_clang() {
    command -v brew >/dev/null 2>&1 || return 1
    local prefix
    prefix="$(brew --prefix llvm 2>/dev/null)" || return 1
    [ -x "$prefix/bin/clang" ] || return 1
    printf '%s' "$prefix/bin/clang"
}

# The C in the dependency tree (tree-sitter's parser, SQLite) is compiled by
# `cc-rs`. Apple's clang carries no wasm backend, so on a stock macOS the
# default compiler cannot build any of it — find the Homebrew LLVM that can
# rather than make every contributor export it.
#
# `CC` alone is not enough: cc-rs archives with `llvm-ar` for a wasm target and
# resolves that by name, not from `AR`, so the toolchain's directory has to be
# reachable. It goes on the *end* of `PATH` — everything else in there is a
# name Apple's toolchain does not also provide, so appending adds `llvm-ar`
# without shadowing the host compiler for anyone's build scripts.
check_compiler() {
    local cc="${CC:-clang}"
    if supports_wasm "$cc"; then
        return
    fi
    # Only when the caller named no compiler. An explicit `CC` that cannot do
    # the job is a mistake to report, not one to silently work around.
    if [ -z "${CC:-}" ]; then
        local fallback
        if fallback="$(brew_llvm_clang)" && supports_wasm "$fallback"; then
            echo "Using Homebrew LLVM ('$cc' has no wasm backend): $fallback"
            export CC="$fallback"
            export AR="$(dirname "$fallback")/llvm-ar"
            export PATH="$PATH:$(dirname "$fallback")"
            return
        fi
    fi
    echo "Error: '$cc' does not support the wasm32-unknown-unknown target." >&2
    echo "Install an LLVM/clang toolchain with wasm backend support:" >&2
    echo "  macOS:         brew install llvm" >&2
    echo "  Debian/Ubuntu: sudo apt-get install llvm" >&2
    exit 1
}

check_wasm_opt() {
    if ! command -v wasm-opt >/dev/null 2>&1; then
        echo "Error: wasm-opt not found. Install binaryen or run: cargo install wasm-opt" >&2
        exit 1
    fi
}

# The schema the CLI writes has to match the one the crate was built against,
# so the version is read from Cargo.lock rather than taking whatever is on
# PATH.
check_wasm_bindgen() {
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

if [ "$SKIP_BINARY" = false ]; then
    # Everything the binary steps need, checked before the long compile rather
    # than after it.
    echo "Checking wasm build prerequisites..."
    check_compiler
    check_wasm_bindgen
    if [ "$SKIP_OPT" = false ]; then check_wasm_opt; fi

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
        # The features rustc actually emits, named one by one. wasm-opt rejects
        # them unless told to expect them, and `--all-features` is not the
        # shortcut: it enables post-MVP proposals browsers still reject —
        # binaryen 132 emits compact imports under it, which a browser refuses
        # to compile ("Invalid import kind 127").
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

# The npm version is stamped from the crate's, not maintained beside it: two
# hand-edited numbers drift, and the mismatch would only show up at publish
# time. `cargo pkgid` prints `<url>#<version>` or `<url>#<name>@<version>`
# depending on whether the directory name matches the package name.
CRATE_VERSION="$( (cd "$SCRIPT_DIR" && cargo pkgid -p ggsql-wasm) | sed -e 's/.*#//' -e 's/.*[@:]//')"
if [ -z "$CRATE_VERSION" ]; then
    echo "Error: could not read the ggsql-wasm version from cargo pkgid." >&2
    exit 1
fi
PKG_VERSION="$(cd "$SCRIPT_DIR/pkg" && npm pkg get version | tr -d '"')"
if [ "$CRATE_VERSION" != "$PKG_VERSION" ]; then
    echo "Stamping pkg/package.json version: $PKG_VERSION -> $CRATE_VERSION"
    (cd "$SCRIPT_DIR/pkg" && npm pkg set "version=$CRATE_VERSION")
fi

# `npm install` follows the stamp, so package-lock.json picks the version up in
# the same run.
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

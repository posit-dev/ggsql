# Building Cross-Platform Installers

ggsql ships native installers for Windows, macOS, and Linux. Windows (NSIS / MSI) and Linux (Deb) installers are built via [cargo-packager](https://github.com/crabnebula-dev/cargo-packager); macOS installers are built directly with Apple's `pkgbuild`, then code-signed and notarized.

## Quick Start

### Prerequisites

1. **For Windows / Linux installers — install cargo-packager**:

   ```bash
   cargo install cargo-packager --locked
   ```

2. **Platform-specific requirements**:
   - **Windows**: No additional requirements (uses built-in NSIS, optionally WiX if installed)
   - **macOS**: Xcode Command Line Tools, plus [`dylibbundler`](https://github.com/auriamg/macdylibbundler) (`brew install dylibbundler`) for bundling Arrow / DuckDB dynamic libraries
   - **Linux**: `sudo apt-get install libgtk-3-dev libwebkit2gtk-4.0-dev libappindicator3-dev librsvg2-dev patchelf`

### Build Installers Locally

```bash
# Windows
cd ggsql-cli
cargo packager --release --formats nsis    # Creates .exe installer (NSIS)
cargo packager --release --formats wix     # Creates .msi installer (WiX)

# Linux
cd ggsql-cli
cargo packager --release --formats deb     # Creates .deb package (Debian/Ubuntu)
```

Output for cargo-packager: `ggsql-cli/target/release/packager/`.

**macOS** uses a separate `pkgbuild` flow (matches what CI ships). From the workspace root:

```bash
# Build the binaries
cargo build --release --bin ggsql --bin ggsql-jupyter

# Bundle dylibs alongside the binaries
dylibbundler -cd -of -b -x target/release/ggsql         -d ./libs/ -p "@executable_path/../lib/ggsql/"
dylibbundler -cd -of -b -x target/release/ggsql-jupyter -d ./libs/ -p "@executable_path/../lib/ggsql/"

# Stage payload and build an unsigned .pkg
mkdir -p pkg-payload/usr/local/bin pkg-payload/usr/local/lib/ggsql
cp target/release/ggsql target/release/ggsql-jupyter pkg-payload/usr/local/bin/
cp -R ./libs/. pkg-payload/usr/local/lib/ggsql/
pkgbuild \
  --root ./pkg-payload \
  --install-location / \
  --identifier co.posit.ggsql \
  --version 0.0.0-dev \
  ggsql-dev.pkg
```

CI additionally codesigns the binaries with `entitlements.plist` (Developer ID Application), signs the `.pkg` (Developer ID Installer), and notarizes via `xcrun notarytool` — see [`.github/workflows/release-packages.yml`](.github/workflows/release-packages.yml). Local builds without those creds produce an unsigned `.pkg` that's fine for testing but will be Gatekeeper-blocked on other machines.

## Available Formats

| Platform    | Format | Tool             | Output                          |
| ----------- | ------ | ---------------- | ------------------------------- |
| **Windows** | NSIS   | `cargo packager --formats nsis` | `ggsql_0.1.0_x64-setup.exe`     |
| **Windows** | MSI    | `cargo packager --formats wix`  | `ggsql_0.1.0_x64_en-US.msi`     |
| **macOS**   | PKG    | `pkgbuild` (see above)          | `ggsql_0.1.0_x86_64.pkg`, `ggsql_0.1.0_aarch64.pkg` |
| **Linux**   | Debian | `cargo packager --formats deb`  | `ggsql_0.1.0_amd64.deb`         |

## What Gets Packaged

The installers include:

- ✅ **ggsql** - Main CLI binary
- ✅ **ggsql-jupyter** - ggsql Jupyter kernel

## Configuration

Windows / Linux installer configuration is in `ggsql-cli/Cargo.toml` under `[package.metadata.packager]` (the macOS `.pkg` build is driven entirely by the `pkgbuild` invocation in the workflow, not by this metadata):

```toml
[package.metadata.packager]
product-name = "ggsql"
identifier = "com.ggsql.app"
category = "DeveloperTool"
publisher = "ggsql Team"
icons = ["../doc/assets/logo.png"]
license-file = "../LICENSE.md"
binaries = [
  { path = "ggsql", main = true },
  { path = "ggsql-jupyter", main = false },
]
```

## Automated Releases

GitHub Actions automatically builds installers for all platforms when you push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow (`.github/workflows/release-packages.yml`) will:

1. **Build installers** for Windows (NSIS + MSI), macOS (PKG, x86_64 + aarch64, signed and notarized), and Linux (Deb, x86_64 + aarch64)
2. **Create a GitHub Release** with all installers attached
3. **Generate release notes** automatically

You can also trigger builds manually from the Actions tab.

## Installation

### Windows

**Option 1: NSIS Installer (Recommended for users)**

- Double-click `ggsql_0.1.0_x64-setup.exe`
- Automatically adds to PATH

**Option 2: MSI Installer (Recommended for enterprises)**

- Double-click `ggsql_0.1.0_x64_en-US.msi`
- Follows Windows Installer standards
- Supports silent installation: `msiexec /i ggsql_0.1.0_x64_en-US.msi /quiet`

### macOS

**PKG Installer**

- Double-click `ggsql_0.1.0_x86_64.pkg` (Intel) or `ggsql_0.1.0_aarch64.pkg` (Apple Silicon)
- Installs `ggsql` and `ggsql-jupyter` into `/usr/local/bin/` and bundled dylibs into `/usr/local/lib/ggsql<version>/`
- CLI command-line install: `sudo installer -pkg ggsql_0.1.0_aarch64.pkg -target /`

### Linux

**Debian/Ubuntu** (.deb):

```bash
sudo dpkg -i ggsql_0.1.0_amd64.deb
```

## Testing Locally

After building an installer, test it:

### Windows

```powershell
# Install
.\ggsql_0.1.0_x64-setup.exe

# Verify
ggsql --version

# Uninstall
# Settings → Apps → Find "ggsql" → Uninstall
```

### macOS

```bash
# Install
sudo installer -pkg ggsql_0.1.0_aarch64.pkg -target /

# Verify
ggsql --version
ggsql-jupyter --version

# Uninstall
sudo rm /usr/local/bin/ggsql /usr/local/bin/ggsql-jupyter
sudo rm -rf /usr/local/lib/ggsql*
```

### Linux

```bash
# Debian
sudo dpkg -i ggsql_0.1.0_amd64.deb
ggsql --version

# Uninstall
sudo apt-get remove ggsql
```

## Troubleshooting

### Windows: "Windows protected your PC"

This happens with unsigned installers. Click "More info" → "Run anyway". For production, sign the installer with a code signing certificate.

### macOS: ".pkg can't be opened" / "unidentified developer"

Locally-built `.pkg` files are unsigned and Gatekeeper will block double-clicking them. Either install from the command line:

```bash
sudo installer -pkg ggsql-dev.pkg -target /
```

…or right-click → Open to bypass Gatekeeper for that file. Official releases from GitHub are signed with the Developer ID Installer certificate and notarized, so they install without warnings.

### Linux: Missing dependencies

If the Deb/RPM package fails to install, ensure you have the required system libraries:

```bash
sudo apt-get install -f  # Debian/Ubuntu
sudo dnf install <missing-packages>  # Fedora
```

## Advanced: Custom Builds

### Cross-compilation

Build for different architectures:

```bash
# macOS: Build for Apple Silicon
rustup target add aarch64-apple-darwin
cargo build --release --target aarch64-apple-darwin --bin ggsql --bin ggsql-jupyter
# Then run the dylibbundler + pkgbuild flow shown above against
# target/aarch64-apple-darwin/release/ instead of target/release/.

# Linux: Build for ARM64
rustup target add aarch64-unknown-linux-gnu
cargo packager --release --target aarch64-unknown-linux-gnu --formats deb
```

## Resources

- [cargo-packager Documentation](https://packager.crabnebula.dev/)
- [GitHub Releases](https://github.com/georgestagg/ggsql/releases)
- [Issue Tracker](https://github.com/georgestagg/ggsql/issues)

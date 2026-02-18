# Building Cross-Platform Installers

ggsql uses [cargo-packager](https://github.com/crabnebula-dev/cargo-packager) to create native installers for Windows, macOS, and Linux.

## Quick Start

### Prerequisites

1. **Install cargo-packager**:
   ```bash
   cargo install cargo-packager --locked
   ```

2. **Platform-specific requirements**:
   - **Windows**: No additional requirements (uses built-in NSIS, optionally WiX if installed)
   - **macOS**: Xcode Command Line Tools
   - **Linux**: `sudo apt-get install libgtk-3-dev libwebkit2gtk-4.0-dev libappindicator3-dev librsvg2-dev patchelf`

### Build Installers Locally

From the `src/` directory:

```bash
# Windows
cd src
cargo packager --release --formats nsis    # Creates .exe installer (NSIS)
cargo packager --release --formats wix     # Creates .msi installer (WiX)

# macOS
cd src
cargo packager --release --formats dmg     # Creates .dmg disk image

# Linux
cd src
cargo packager --release --formats deb     # Creates .deb package (Debian/Ubuntu)
```

Output location: `src/target/release/packager/`

## Available Formats

| Platform | Format | Command | Output |
|----------|--------|---------|--------|
| **Windows** | NSIS | `--formats nsis` | `ggsql_0.1.0_x64-setup.exe` (12MB) |
| **Windows** | MSI | `--formats wix` | `ggsql_0.1.0_x64_en-US.msi` (15MB) |
| **macOS** | DMG | `--formats dmg` | `ggsql_0.1.0_x64.dmg` |
| **macOS** | App Bundle | `--formats app` | `ggsql.app` |
| **Linux** | Debian | `--formats deb` | `ggsql_0.1.0_amd64.deb` |

## What Gets Packaged

The installers include:

- ✅ **ggsql** - Main CLI binary

**Not included** (install separately if needed):
- **ggsql-rest** - REST API server: `cargo install ggsql --features rest-api`
- **ggsql-jupyter** - Jupyter kernel: `cargo install ggsql-jupyter`

## Configuration

Installer configuration is in `src/Cargo.toml` under `[package.metadata.packager]`:

```toml
[package.metadata.packager]
product-name = "ggsql"
identifier = "com.ggsql.app"
category = "DeveloperTool"
publisher = "ggsql Team"
icons = ["../doc/assets/icon.svg", "../doc/assets/logo.png"]
license-file = "../LICENSE.md"
binaries = [
  { path = "ggsql", main = true },
  { path = "ggsql-rest", main = false },
]
```

## Automated Releases

GitHub Actions automatically builds installers for all platforms when you push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow (`.github/workflows/release-installers.yml`) will:

1. **Build installers** for Windows (NSIS + MSI), macOS (DMG), and Linux (Deb)
2. **Create a GitHub Release** with all installers attached
3. **Generate release notes** automatically

You can also trigger builds manually from the Actions tab.

## Installation

### Windows

**Option 1: NSIS Installer (Recommended for users)**
- Double-click `ggsql_0.1.0_x64-setup.exe`
- Choose components (CLI only, or CLI + REST API)
- Automatically adds to PATH

**Option 2: MSI Installer (Recommended for enterprises)**
- Double-click `ggsql_0.1.0_x64_en-US.msi`
- Follows Windows Installer standards
- Supports silent installation: `msiexec /i ggsql_0.1.0_x64_en-US.msi /quiet`

### macOS

**DMG Installer**
- Open `ggsql_0.1.0_x64.dmg`
- Drag `ggsql.app` to Applications folder
- Or copy binaries directly to `/usr/local/bin`

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
ggsql-rest --version  # if installed

# Uninstall
# Settings → Apps → Find "ggsql" → Uninstall
```

### macOS
```bash
# Install
hdiutil attach ggsql_0.1.0_x64.dmg
cp -r /Volumes/ggsql/ggsql.app /Applications/

# Verify
/Applications/ggsql.app/Contents/MacOS/ggsql --version
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

### macOS: "ggsql.app is damaged"

This happens with unsigned apps. Run:
```bash
xattr -cr /Applications/ggsql.app
```

### Linux: Missing dependencies

If the Deb/RPM package fails to install, ensure you have the required system libraries:
```bash
sudo apt-get install -f  # Debian/Ubuntu
sudo dnf install <missing-packages>  # Fedora
```

## Advanced: Custom Builds

### Include Jupyter kernel

By default, ggsql-jupyter is not included (it requires Python). To package it:

1. Build ggsql-jupyter:
   ```bash
   cargo build --release --package ggsql-jupyter
   ```

2. Update `src/Cargo.toml`:
   ```toml
   binaries = [
     { path = "ggsql", main = true },
     { path = "ggsql-rest", main = false },
     { path = "../target/release/ggsql-jupyter", main = false },
   ]
   ```

3. Rebuild the installer

### Cross-compilation

Build for different architectures:

```bash
# macOS: Build for Apple Silicon
rustup target add aarch64-apple-darwin
cargo packager --release --target aarch64-apple-darwin --formats dmg

# Linux: Build for ARM64
rustup target add aarch64-unknown-linux-gnu
cargo packager --release --target aarch64-unknown-linux-gnu --formats deb
```

## Resources

- [cargo-packager Documentation](https://packager.crabnebula.dev/)
- [GitHub Releases](https://github.com/georgestagg/ggsql/releases)
- [Issue Tracker](https://github.com/georgestagg/ggsql/issues)

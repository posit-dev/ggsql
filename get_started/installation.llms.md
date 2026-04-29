# Installation

While all of the examples on this website uses the ggsql WebAssembly build to let you run it in the browser, you’ll likely want it on your own computer, interfacing with your local data.

ggsql provides native installers for Windows, macOS, and Linux. Choose the best option for your platform below.

### All platforms

![](../assets/logos/jupyter.svg) ![](../assets/logos/quarto.svg)

## Jupyter kernel

To use ggsql in Jupyter or Quarto notebooks, install the Jupyter kernel using either `uv` (recommended) or `cargo`. The kernel is also part of the main installer linked to above.

#### Using uv

[uv](https://docs.astral.sh/uv/) is the fastest way to install the binaries:

``` bash
uv tool install ggsql-jupyter
ggsql-jupyter --install
```

#### Using cargo

If you have a Rust toolchain installed you can install with cargo:

``` bash
cargo install ggsql-jupyter
ggsql-jupyter --install
```

![](../assets/logos/vscode.svg) ![](../assets/logos/positron.svg)

## VS Code / Positron extension

For syntax highlighting and language support in VS Code or Positron, install the ggsql extension. You can either install it directly from the [extension marketplace](https://open-vsx.org/extension/ggsql/ggsql) from within the IDE or download and install it manually (in the *Extensions* view, click the `...` menu, select “Install from VSIX…”, and choose the downloaded file.)

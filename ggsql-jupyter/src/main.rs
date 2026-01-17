//! ggsql Jupyter Kernel
//!
//! A Jupyter kernel for executing ggsql queries with rich Vega-Lite visualizations.

mod display;
mod executor;
mod kernel;
mod message;

use anyhow::{Context, Result};
use clap::Parser;
use message::ConnectionInfo;
use std::env;
use std::fs;
use std::process::Command;

#[derive(Parser)]
#[command(name = "ggsql-jupyter")]
#[command(about = "Jupyter kernel for ggsql", long_about = None)]
struct Args {
    /// Path to the Jupyter connection file
    #[arg(short = 'f', long = "connection-file")]
    connection_file: Option<String>,

    /// Install the kernel spec
    #[arg(long)]
    install: bool,

    /// Install the kernel spec for the current user only (used with --install)
    #[arg(long, requires = "install")]
    user: bool,

    /// Install the kernel spec system-wide (used with --install, may require sudo)
    #[arg(long, requires = "install", conflicts_with = "user")]
    sys_prefix: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    tracing::info!("ggsql Jupyter Kernel v{}", env!("CARGO_PKG_VERSION"));

    // Parse command-line arguments
    let args = Args::parse();

    // Handle --install
    if args.install {
        return install_kernel(args.user, args.sys_prefix);
    }

    // Normal kernel operation
    let connection_file = args
        .connection_file
        .context("Connection file is required (use -f or --connection-file)")?;

    tracing::info!("Loading connection file: {}", connection_file);

    // Load connection info
    let connection = ConnectionInfo::from_file(&connection_file)?;

    tracing::info!("Creating kernel server");

    // Create and run kernel
    let mut kernel = kernel::KernelServer::new(connection).await?;

    tracing::info!("Kernel ready, starting event loop");

    kernel.run().await?;

    tracing::info!("Kernel shutdown complete");

    Ok(())
}

/// Install the kernel spec using `jupyter kernelspec install`
fn install_kernel(user: bool, sys_prefix: bool) -> Result<()> {
    println!("Installing ggsql Jupyter kernel...");

    // Create a temporary directory for the kernel spec
    let temp_dir = env::temp_dir().join("ggsql-kernel-install");
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).context("Failed to clean temporary directory")?;
    }
    fs::create_dir_all(&temp_dir).context("Failed to create temporary directory")?;

    // Get the path to the current executable
    let exe_path = env::current_exe().context("Failed to get current executable path")?;

    // Copy the executable to the temp directory
    let dest_exe = temp_dir.join(if cfg!(windows) {
        "ggsql-jupyter.exe"
    } else {
        "ggsql-jupyter"
    });

    fs::copy(&exe_path, &dest_exe).context("Failed to copy executable")?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&dest_exe)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dest_exe, perms)?;
    }

    // Create kernel.json
    let kernel_json = serde_json::json!({
        "argv": [
            "{resource_dir}/ggsql-jupyter",
            "-f",
            "{connection_file}"
        ],
        "display_name": "ggsql",
        "language": "ggsql",
        "interrupt_mode": "signal",
        "env": {},
        "metadata": {
            "debugger": false
        }
    });

    let kernel_json_path = temp_dir.join("kernel.json");
    fs::write(
        &kernel_json_path,
        serde_json::to_string_pretty(&kernel_json)?,
    )
    .context("Failed to write kernel.json")?;

    // Build the jupyter kernelspec install command
    let mut cmd = Command::new("jupyter");
    cmd.arg("kernelspec")
        .arg("install")
        .arg(&temp_dir)
        .arg("--name")
        .arg("ggsql");

    if user {
        cmd.arg("--user");
    } else if sys_prefix {
        cmd.arg("--sys-prefix");
    } else {
        // Default to --user
        cmd.arg("--user");
    }

    println!(
        "Running: jupyter kernelspec install {} --name ggsql{}",
        temp_dir.display(),
        if user || (!sys_prefix) {
            " --user"
        } else {
            " --sys-prefix"
        }
    );

    // Execute the command
    let status = cmd
        .status()
        .context("Failed to execute 'jupyter kernelspec install'. Is Jupyter installed?")?;

    // Clean up temporary directory
    fs::remove_dir_all(&temp_dir).context("Failed to remove temporary directory")?;

    if status.success() {
        println!("\n✓ ggsql kernel installed successfully!");
        println!("\nTo verify installation, run:");
        println!("  jupyter kernelspec list");
        Ok(())
    } else {
        anyhow::bail!("jupyter kernelspec install failed with status: {}", status);
    }
}

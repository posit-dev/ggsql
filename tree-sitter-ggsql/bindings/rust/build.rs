use std::path::PathBuf;
use std::process::Command;

fn main() {
    let check = Command::new("tree-sitter").arg("--version").output();
    match check {
        Ok(output) if output.status.success() => {}
        _ => {
            println!("tree-sitter-cli not found. Attempting to install...");
            let installation = Command::new("npm")
                .args(["install", "-g", "tree-sitter-cli"])
                .status();

            match installation {
                Ok(installation) if installation.success() => {}
                _ => {
                    eprintln!("Failed to install tree-sitter-cli.")
                }
            }
        }
    }

    let regenerate = Command::new("tree-sitter").arg("generate").status();

    match regenerate {
        Ok(regenerate) if regenerate.success() => {}
        _ => {
            eprintln!("Failed to regenerate tree sitter grammar.");
        }
    }

    let dir: PathBuf = ["src"].iter().collect();

    cc::Build::new()
        .include(&dir)
        .file(dir.join("parser.c"))
        .compile("tree-sitter-ggsql");
}

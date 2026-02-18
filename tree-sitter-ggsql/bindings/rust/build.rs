use std::path::PathBuf;
use std::process::Command;

/// Find tree-sitter executable, checking PATH and common npm global locations
fn find_tree_sitter() -> Option<PathBuf> {
    // First, check if tree-sitter is in PATH
    if Command::new("tree-sitter")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from("tree-sitter"));
    }

    // On Windows, check common npm global install locations
    #[cfg(windows)]
    {
        // Check C:\npm\prefix (GitHub Actions setup-node location)
        for ext in &["cmd", "ps1", "exe"] {
            let npm_path = PathBuf::from(r"C:\npm\prefix").join(format!("tree-sitter.{}", ext));
            if npm_path.exists() {
                return Some(npm_path);
            }
        }

        if let Some(appdata) = std::env::var_os("APPDATA") {
            for ext in &["cmd", "ps1", "exe"] {
                let npm_path = PathBuf::from(&appdata)
                    .join("npm")
                    .join(format!("tree-sitter.{}", ext));
                if npm_path.exists() {
                    return Some(npm_path);
                }
            }
        }

        // Also check USERPROFILE\AppData\Roaming\npm
        if let Some(userprofile) = std::env::var_os("USERPROFILE") {
            for ext in &["cmd", "ps1", "exe"] {
                let npm_path = PathBuf::from(&userprofile)
                    .join("AppData")
                    .join("Roaming")
                    .join("npm")
                    .join(format!("tree-sitter.{}", ext));
                if npm_path.exists() {
                    return Some(npm_path);
                }
            }
        }
    }

    None
}

fn run_tree_sitter(
    tree_sitter: &PathBuf,
    grammar_dir: &PathBuf,
) -> std::io::Result<std::process::ExitStatus> {
    // Check if this is a PowerShell script
    let ext = tree_sitter
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    if ext == "ps1" {
        // Run PowerShell scripts through powershell.exe
        Command::new("powershell")
            .args(["-ExecutionPolicy", "Bypass", "-File"])
            .arg(tree_sitter)
            .arg("generate")
            .current_dir(grammar_dir)
            .status()
    } else {
        // Run cmd/exe directly
        Command::new(tree_sitter)
            .arg("generate")
            .current_dir(grammar_dir)
            .status()
    }
}

fn main() {
    // CARGO_MANIFEST_DIR points to tree-sitter-ggsql/ where Cargo.toml and grammar.js live
    let grammar_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = grammar_dir.join("src");
    let parser_c = src_dir.join("parser.c");

    // Re-run this build script if the env var changes.
    println!("cargo:rerun-if-env-changed=GGSQL_SKIP_GENERATE");

    // By default, always regenerate parser.c from grammar.js so local builds
    // pick up grammar changes automatically. CI sets GGSQL_SKIP_GENERATE=1 to
    // skip this step when pre-generated parser files are provided as artifacts.
    let skip_generate = std::env::var("GGSQL_SKIP_GENERATE").is_ok();

    if skip_generate {
        if !parser_c.exists() {
            panic!(
                "GGSQL_SKIP_GENERATE is set but src/parser.c does not exist. \
                 Either run `tree-sitter generate` first, or unset GGSQL_SKIP_GENERATE."
            );
        }
    } else {
        let tree_sitter = find_tree_sitter().unwrap_or_else(|| {
            panic!("tree-sitter-cli not found. Please install it: npm install -g tree-sitter-cli");
        });

        let generate_result = run_tree_sitter(&tree_sitter, &grammar_dir);

        match generate_result {
            Ok(status) if status.success() => {}
            Ok(status) => {
                panic!("tree-sitter generate failed with status: {}", status);
            }
            Err(e) => {
                panic!("Failed to run tree-sitter generate: {}", e);
            }
        }
    }

    // The generated files are in the grammar_dir/src directory
    cc::Build::new()
        .include(&src_dir)
        .file(&parser_c)
        .compile("tree-sitter-ggsql");
}

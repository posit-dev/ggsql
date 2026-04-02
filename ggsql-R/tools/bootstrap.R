# Bootstrap script for building ggsql R package outside the monorepo.
#
# When R CMD check copies the package to a temp directory, the path dependency
# to the ggsql Rust crate (../../../src) breaks. This script copies the
# required workspace crates into the R package source and creates a local
# workspace Cargo.toml so cargo can resolve all dependencies.
#
# Called from tools/config.R before Makevars generation.

bootstrap_rust_workspace <- function() {
  # Check if bootstrap has already been done (e.g., from a tarball)
  if (file.exists("src/ggsql-lib/Cargo.toml")) {
    message("Bootstrap: using previously bootstrapped workspace.")
    return(invisible())
  }

  # Locate the monorepo root relative to the R package directory
  repo_root <- normalizePath("..", mustWork = FALSE)

  ggsql_src <- file.path(repo_root, "src")
  ts_src <- file.path(repo_root, "tree-sitter-ggsql")
  root_cargo <- file.path(repo_root, "Cargo.toml")
  root_lock <- file.path(repo_root, "Cargo.lock")

  if (!file.exists(file.path(ggsql_src, "Cargo.toml"))) {
    stop(
      "Cannot find ggsql Rust crate at '", ggsql_src, "'.\n",
      "Either build from the monorepo or use a source tarball created with R CMD build.",
      call. = FALSE
    )
  }

  message("Bootstrap: copying workspace crates into R package...")

  # Copy the ggsql main crate
  copy_dir(ggsql_src, "src/ggsql-lib")

  # Copy tree-sitter-ggsql
  copy_dir(ts_src, "src/tree-sitter-ggsql")

  # Copy workspace Cargo.lock for reproducible builds
  if (file.exists(root_lock)) {
    file.copy(root_lock, "src/Cargo.lock", overwrite = TRUE)
  }

  # Create a local workspace Cargo.toml by adapting the root one
  create_workspace_toml(root_cargo)

  message("Bootstrap: done.")
}

copy_dir <- function(from, to) {
  if (dir.exists(to)) {
    unlink(to, recursive = TRUE)
  }
  dir.create(to, recursive = TRUE, showWarnings = FALSE)

  # Copy all files preserving structure
  files <- list.files(from, recursive = TRUE, all.files = TRUE, no.. = TRUE)

  # Exclude target/ and .cargo/ directories
  files <- files[!grepl("^target/|/target/|^\\.cargo/|/\\.cargo/", files)]

  for (f in files) {
    src <- file.path(from, f)
    dst <- file.path(to, f)
    dir.create(dirname(dst), recursive = TRUE, showWarnings = FALSE)
    file.copy(src, dst, overwrite = TRUE)
  }
}

create_workspace_toml <- function(root_cargo_path) {
  lines <- readLines(root_cargo_path)

  # Remove multi-line array sections: members, default-members, exclude
  lines <- remove_toml_array(lines, "members")
  lines <- remove_toml_array(lines, "default-members")
  lines <- remove_toml_array(lines, "exclude")

  # Remove profile sections
  lines <- remove_toml_section(lines, "profile.")

  # Insert our members right after [workspace]
  ws_idx <- which(grepl("^\\[workspace\\]$", lines))
  if (length(ws_idx) > 0) {
    lines <- append(
      lines,
      'members = ["rust", "ggsql-lib", "tree-sitter-ggsql"]',
      after = ws_idx[1]
    )
  }

  # Adjust workspace dependency paths
  lines <- gsub('path = "src"', 'path = "ggsql-lib"', lines, fixed = TRUE)

  # Remove trailing blank lines
  while (length(lines) > 0 && trimws(lines[length(lines)]) == "") {
    lines <- lines[-length(lines)]
  }

  writeLines(lines, "src/Cargo.toml")
}

# Remove a multi-line TOML array (key = [\n...\n])
remove_toml_array <- function(lines, key) {
  in_array <- FALSE
  keep <- rep(TRUE, length(lines))
  for (i in seq_along(lines)) {
    if (!in_array && grepl(paste0("^", key, "\\s*=\\s*\\["), lines[i])) {
      in_array <- TRUE
      keep[i] <- FALSE
      # Check if array closes on same line
      if (grepl("\\]", lines[i])) {
        in_array <- FALSE
      }
      next
    }
    if (in_array) {
      keep[i] <- FALSE
      if (grepl("\\]", lines[i])) {
        in_array <- FALSE
      }
    }
  }
  lines[keep]
}

# Remove a TOML section and its contents (e.g., [profile.wasm])
remove_toml_section <- function(lines, prefix) {
  pattern <- paste0("^\\[", gsub("\\.", "\\\\.", prefix))
  in_section <- FALSE
  keep <- rep(TRUE, length(lines))
  for (i in seq_along(lines)) {
    if (grepl(pattern, lines[i])) {
      in_section <- TRUE
      keep[i] <- FALSE
      next
    }
    if (in_section) {
      if (grepl("^\\[", lines[i])) {
        in_section <- FALSE
        keep[i] <- TRUE
      } else {
        keep[i] <- FALSE
      }
    }
  }
  lines[keep]
}

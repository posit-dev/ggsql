---
name: Release
about: Checklist for releasing a new version of ggsql
title: "Release ggsql X.Y.Z"
labels: release
assignees: ''
---

- [ ] Create new branch `release-version-X-Y-Z`.
- [ ] Bump version numbers, all to match:
    - ggsql
      - Main package `version` in `[workspace.package]` section in `Cargo.toml`.
      - All packages `version` in `[workspace.dependencies]` section in `Cargo.toml`.
    - tree-sitter-ggsql
      - `version` in `tree-sitter-ggsql/package.json`.
      - `version` in `metadata` in `tree-sitter-ggsql/tree-sitter.json`.
      - `version` in `[project]` section in `tree-sitter-ggsql/pyproject.toml`.
      - `__version__` in `tree-sitter-ggsql/bindings/python/__init__.py`.
    - ggsql-jupyter
      - `version` in `[project]` section in `ggsql-jupyter/pyproject.toml`.
    - ggsql-vscode
      - `version` in `ggsql-vscode/package.json`.
- [ ] Update lock files:
    - Run `cargo build`, ensure `Cargo.lock` updates.
    - Run `(cd ggsql-wasm && ./build-wasm.sh && cd demo && npm install)`, ensure that `ggsql-wasm/demo/package-lock.json` updates.
    - Run `(cd ggsql-vscode && npm install)`, ensure that `ggsql-vscode/package-lock.json` updates.
- [ ] Ensure news bullets are up to date, add a new header for this release.
    - `CHANGELOG.md`
    - `ggsql-vscode/CHANGELOG.md`
- [ ] Push the branch, make a PR.
- [ ] Watch CI. Once all green, and we are happy, squash-merge the PR.
- [ ] Create a new GitHub release, create a new tag in the form `vX.Y.Z`
    - Select the commit we just squash merged in for the new tag.
    - Click the button to generate a changelog automatically so our friends all get tagged in the release.
    - Check the box to set as the latest release.
- After releasing, watch CI for issues. If anything is red, we must fix, bump patch version, and try again.
   - For minor issues, we might want to manually fix up and publish. Currently George's npm and cargo accounts must be used to do this.
- After all green, open Issues in downstream repos asking to update their dependency on ggsql to the latest version,
    - [ ] [ggsql-r](https://github.com/posit-dev/ggsql-r)
    - [ ] [ggsql-python](https://github.com/posit-dev/ggsql-python)
    - [ ] [ggsql-duckdb](https://github.com/posit-dev/ggsql-duckdb)
- [ ] Update the ggsql skill at [posit-dev/skills](https://github.com/posit-dev/skills)

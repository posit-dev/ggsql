/*
 * Copies the canonical ggsql agent skill into the extension before packaging.
 *
 * The canonical source within this repo is doc/vendor/SKILL.md, which
 * ggsql-cli/build.rs keeps in sync with the posit-dev/skills repository
 * (rebuild the CLI with GGSQL_UPDATE_SKILL=1 to refresh it). The extension
 * registers the skills/ directory as an agent skill root, so the packaged
 * copy must live inside the extension; this script materialises it at
 * package time rather than keeping a second, hand-maintained copy that
 * would drift.
 */

const fs = require('fs');
const path = require('path');

const repoRoot = path.join(__dirname, '..', '..');
const source = path.join(repoRoot, 'doc', 'vendor', 'SKILL.md');
const destDir = path.join(__dirname, '..', 'skills', 'ggsql');
const dest = path.join(destDir, 'SKILL.md');

const content = fs.readFileSync(source, 'utf8');

// Positron discovers skills by name/description frontmatter; fail loudly if
// the canonical file ever loses them instead of shipping a broken skill.
for (const field of ['name:', 'description:']) {
    if (!content.startsWith('---') || !content.includes(`\n${field}`)) {
        console.error(`sync-skill: ${source} is missing '${field}' frontmatter`);
        process.exit(1);
    }
}

fs.mkdirSync(destDir, { recursive: true });
fs.writeFileSync(dest, content);
console.log(`sync-skill: copied ${path.relative(repoRoot, source)} -> ${path.relative(repoRoot, dest)}`);

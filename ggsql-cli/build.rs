use regex::Regex;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

struct DocEntry {
    category: Option<String>,
    topic: String,
    title: String,
    body: String,
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let doc_dir = manifest_dir.parent().unwrap().join("doc");
    let syntax_dir = doc_dir.join("syntax");
    let quarto_yml = doc_dir.join("_quarto.yml");
    let skill_cache = doc_dir.join("vendor").join("SKILL.md");

    println!("cargo:rerun-if-changed={}", syntax_dir.display());
    println!("cargo:rerun-if-changed={}", quarto_yml.display());
    println!("cargo:rerun-if-changed={}", skill_cache.display());
    println!("cargo:rerun-if-env-changed=GGSQL_UPDATE_SKILL");

    let site_url = read_site_url(&quarto_yml)
        .unwrap_or_else(|| "https://ggsql.org".to_string())
        .trim_end_matches('/')
        .to_string();

    let mut entries: Vec<DocEntry> = Vec::new();

    for qmd_path in walk_qmd(&syntax_dir) {
        let rel = qmd_path.strip_prefix(&syntax_dir).unwrap().to_path_buf();
        let parts: Vec<String> = rel
            .iter()
            .map(|c| c.to_string_lossy().into_owned())
            .collect();

        if parts.is_empty() {
            continue;
        }
        let filename = parts.last().unwrap();
        if filename == "index.qmd" {
            continue;
        }
        let Some(stem) = filename.strip_suffix(".qmd") else {
            continue;
        };

        let Some((category, topic)) = classify(&parts, stem) else {
            continue;
        };

        let content = fs::read_to_string(&qmd_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {}", qmd_path.display(), e));
        let (title, body) = split_frontmatter(&content, stem);

        let file_dir_rel = match rel.parent() {
            Some(p) if p.as_os_str().is_empty() => "syntax".to_string(),
            Some(p) => format!(
                "syntax/{}",
                p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/")
            ),
            None => "syntax".to_string(),
        };

        let body = normalize(&body, &site_url, &file_dir_rel);

        entries.push(DocEntry {
            category,
            topic,
            title,
            body,
        });
    }

    entries.sort_by(|a, b| {
        let ac = a.category.as_deref().unwrap_or("");
        let bc = b.category.as_deref().unwrap_or("");
        ac.cmp(bc).then(a.topic.cmp(&b.topic))
    });

    let skill = load_skill(&skill_cache);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("docs_data.rs");
    let mut out = String::new();
    out.push_str("pub struct DocEntry {\n");
    out.push_str("    pub category: Option<&'static str>,\n");
    out.push_str("    pub topic: &'static str,\n");
    out.push_str("    pub title: &'static str,\n");
    out.push_str("    pub body: &'static str,\n");
    out.push_str("}\n\n");
    out.push_str("pub const DOCS: &[DocEntry] = &[\n");
    for e in &entries {
        out.push_str("    DocEntry {\n");
        match &e.category {
            Some(c) => out.push_str(&format!("        category: Some({:?}),\n", c)),
            None => out.push_str("        category: None,\n"),
        }
        out.push_str(&format!("        topic: {:?},\n", e.topic));
        out.push_str(&format!("        title: {:?},\n", e.title));
        out.push_str(&format!("        body: {:?},\n", e.body));
        out.push_str("    },\n");
    }
    out.push_str("];\n\n");

    out.push_str("pub struct SkillContent {\n");
    out.push_str("    pub name: &'static str,\n");
    out.push_str("    pub description: &'static str,\n");
    out.push_str("    pub body: &'static str,\n");
    out.push_str("    pub available: bool,\n");
    out.push_str("}\n\n");
    out.push_str("pub const SKILL: SkillContent = SkillContent {\n");
    out.push_str(&format!("    name: {:?},\n", skill.name));
    out.push_str(&format!("    description: {:?},\n", skill.description));
    out.push_str(&format!("    body: {:?},\n", skill.body));
    out.push_str(&format!("    available: {},\n", skill.available));
    out.push_str("};\n");

    fs::write(&out_path, out)
        .unwrap_or_else(|e| panic!("failed to write {}: {}", out_path.display(), e));
}

struct SkillData {
    name: String,
    description: String,
    body: String,
    available: bool,
}

const SKILL_URL: &str =
    "https://raw.githubusercontent.com/posit-dev/skills/main/ggsql/ggsql/SKILL.md";

fn load_skill(cache_path: &Path) -> SkillData {
    if env::var("GGSQL_UPDATE_SKILL").is_ok() {
        if let Some(body) = fetch_skill() {
            write_if_changed(cache_path, body.as_bytes());
            return parse_skill(&body);
        }
    }

    match fs::read_to_string(cache_path) {
        Ok(raw) => parse_skill(&raw),
        Err(_) => {
            println!(
                "cargo:warning=SKILL.md not available (no cached copy at {}); run with GGSQL_UPDATE_SKILL=1 to fetch. `ggsql skill` will report unavailable.",
                cache_path.display()
            );
            SkillData {
                name: "ggsql".to_string(),
                description: String::new(),
                body: String::new(),
                available: false,
            }
        }
    }
}

fn fetch_skill() -> Option<String> {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(15)))
        .build();
    let agent: ureq::Agent = config.into();

    match agent.get(SKILL_URL).call() {
        Ok(resp) => match resp.into_body().read_to_string() {
            Ok(body) => Some(body),
            Err(e) => {
                println!(
                    "cargo:warning=Failed to read SKILL.md body from {}: {}",
                    SKILL_URL, e
                );
                None
            }
        },
        Err(e) => {
            println!(
                "cargo:warning=Failed to fetch SKILL.md from {}: {}",
                SKILL_URL, e
            );
            None
        }
    }
}

fn parse_skill(raw: &str) -> SkillData {
    let raw = raw.trim_start_matches('\u{FEFF}');
    let (fm, body) = if let Some(rest) = raw.strip_prefix("---\n") {
        match rest.find("\n---\n") {
            Some(end) => (&rest[..end], &rest[end + 5..]),
            None => ("", raw),
        }
    } else if let Some(rest) = raw.strip_prefix("---\r\n") {
        match rest.find("\r\n---\r\n") {
            Some(end) => (&rest[..end], &rest[end + 7..]),
            None => ("", raw),
        }
    } else {
        ("", raw)
    };

    let name = extract_yaml_scalar(fm, "name").unwrap_or_else(|| "ggsql".to_string());
    let description = extract_yaml_scalar(fm, "description").unwrap_or_default();
    let body = body.trim_start_matches(['\n', '\r']).to_string();
    let available = !body.is_empty();

    SkillData {
        name,
        description,
        body,
        available,
    }
}

fn extract_yaml_scalar(frontmatter: &str, key: &str) -> Option<String> {
    let prefix = format!("{}:", key);
    for line in frontmatter.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(&prefix) {
            let value = trimmed[prefix.len()..].trim();
            let value = value.trim_matches('"').trim_matches('\'');
            return Some(value.to_string());
        }
    }
    None
}

fn write_if_changed(path: &Path, content: &[u8]) {
    if let Ok(existing) = fs::read(path) {
        if existing == content {
            return;
        }
    }
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(e) = fs::write(path, content) {
        println!(
            "cargo:warning=Failed to update skill cache at {}: {}",
            path.display(),
            e
        );
    }
}

fn walk_qmd(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return out;
    }
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk_qmd(&path));
        } else if path.extension().is_some_and(|ext| ext == "qmd") {
            out.push(path);
        }
    }
    out
}

fn classify(parts: &[String], stem: &str) -> Option<(Option<String>, String)> {
    match parts.len() {
        2 => match parts[0].as_str() {
            "clause" => Some((None, stem.to_string())),
            "coord" => Some((Some("coord".to_string()), stem.to_string())),
            _ => None,
        },
        3 => match (parts[0].as_str(), parts[1].as_str()) {
            ("layer", "type") => Some((Some("layer".to_string()), stem.to_string())),
            ("layer", "position") => Some((Some("position".to_string()), stem.to_string())),
            ("scale", "type") => Some((Some("scale".to_string()), stem.to_string())),
            ("scale", "aesthetic") => {
                let topic = strip_ordering_prefix(stem);
                Some((Some("aesthetic".to_string()), topic))
            }
            _ => None,
        },
        _ => None,
    }
}

fn strip_ordering_prefix(stem: &str) -> String {
    let Some(idx) = stem.find('_') else {
        return stem.to_string();
    };
    let prefix = &stem[..idx];
    if prefix.is_empty() {
        return stem.to_string();
    }
    let is_digits = prefix.chars().all(|c| c.is_ascii_digit());
    let is_single_letter = prefix.len() == 1 && prefix.chars().next().unwrap().is_ascii_uppercase();
    if is_digits || is_single_letter {
        stem[idx + 1..].to_string()
    } else {
        stem.to_string()
    }
}

fn split_frontmatter(content: &str, fallback_title: &str) -> (String, String) {
    let content = content.trim_start_matches('\u{FEFF}');
    let (fm_body, rest) = if let Some(rest) = content.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---\n") {
            (&rest[..end], &rest[end + 5..])
        } else if let Some(end) = rest.find("\n---\r\n") {
            (&rest[..end], &rest[end + 6..])
        } else {
            return (fallback_title.to_string(), content.to_string());
        }
    } else if let Some(rest) = content.strip_prefix("---\r\n") {
        if let Some(end) = rest.find("\r\n---\r\n") {
            (&rest[..end], &rest[end + 7..])
        } else {
            return (fallback_title.to_string(), content.to_string());
        }
    } else {
        return (fallback_title.to_string(), content.to_string());
    };

    let title = extract_title(fm_body).unwrap_or_else(|| fallback_title.to_string());
    (title, rest.trim_start_matches(['\n', '\r']).to_string())
}

fn extract_title(frontmatter: &str) -> Option<String> {
    for line in frontmatter.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("title:") {
            let val = rest.trim();
            let val = val.trim_matches('"').trim_matches('\'');
            return Some(val.to_string());
        }
    }
    None
}

fn normalize(body: &str, site_url: &str, file_dir_rel: &str) -> String {
    let xref_re = Regex::new(r"\[([^\]]*)\]\(([^)]*?\.qmd)(?:#[^)]*)?\)").unwrap();
    let body = xref_re.replace_all(body, "$1").to_string();

    let fence_re = Regex::new(r"```\{ggsql\}").unwrap();
    let body = fence_re.replace_all(&body, "```ggsql").to_string();

    let img_re = Regex::new(r"!\[([^\]]*)\]\(([^)]+)\)(?:\{[^}]*\})?").unwrap();
    let body = img_re
        .replace_all(&body, |caps: &regex::Captures| {
            let alt = &caps[1];
            let url = &caps[2];
            if url.starts_with("http://") || url.starts_with("https://") || url.starts_with('/') {
                format!("![{}]({})", alt, url)
            } else {
                format!("![{}]({}/{}/{})", alt, site_url, file_dir_rel, url)
            }
        })
        .to_string();

    body
}

fn read_site_url(path: &Path) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("site-url:") {
            let val = rest.trim().trim_matches('"').trim_matches('\'').to_string();
            if !val.is_empty() {
                return Some(val);
            }
        }
    }
    None
}

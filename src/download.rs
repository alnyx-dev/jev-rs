//! Model provisioning: fetch a pre-exported ONNX bundle from the Hugging Face
//! Hub.
//!
//! The reference model ships a ready-to-run ONNX export, so the engine can be
//! provisioned without Python, torch or optimum — just download the files.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// The reference model: a zeroshot NLI head on top of bge-m3.
pub const DEFAULT_REPO: &str = "MoritzLaurer/bge-m3-zeroshot-v2.0";

/// Files of the ONNX export, relative to the `onnx/` folder of the repository.
const BUNDLE: &[&str] = &[
    "model.onnx",
    "model.onnx_data",
    "config.json",
    "tokenizer.json",
];

const HUB: &str = "https://huggingface.co";

/// Download the ONNX bundle of `repo` into `out_dir`.
///
/// Existing files are kept unless `force` is set. Interrupted downloads are
/// resumed through an HTTP range request, which matters for the 2.3 GB of
/// external weights.
pub fn download_model(repo: &str, out_dir: &Path, force: bool) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;

    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());

    println!("[*] Model repository: {repo}");
    println!("[*] Target directory: {}", out_dir.display());
    if let Some(_t) = &token {
        println!("[*] Using HF_TOKEN from the environment");
    }

    for name in BUNDLE {
        let target = out_dir.join(name);
        if target.exists() && !force {
            let size = fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
            if size > 0 {
                println!("[+] {name:<18} already present ({})", human_size(size));
                continue;
            }
        }

        download_file(repo, name, &target, token.as_deref())?;
    }

    println!("[+] Model ready: {}", out_dir.display());
    Ok(())
}

/// Download a single file through a resumable `.part` file.
fn download_file(repo: &str, name: &str, target: &Path, token: Option<&str>) -> Result<(), String> {
    let url = format!("{HUB}/{repo}/resolve/main/onnx/{name}");
    let part = target.with_extension(format!(
        "{}part",
        target
            .extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));

    let resume_from = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    let agent = ureq::Agent::new_with_defaults();
    let mut request = agent.get(&url);
    if let Some(t) = token {
        request = request.header("Authorization", &format!("Bearer {t}"));
    }
    if resume_from > 0 {
        request = request.header("Range", &format!("bytes={resume_from}-"));
    }

    let mut response = request.call().map_err(|e| match e {
        ureq::Error::StatusCode(404) => {
            format!("{name}: not found in '{repo}' (is the repository an ONNX export?)")
        }
        ureq::Error::StatusCode(code) => format!("{name}: the server answered {code}"),
        other => format!("{name}: download failed: {other}"),
    })?;

    // A server that ignores the range header answers 200 with the full body.
    let append = resume_from > 0 && response.status() == 206;
    if resume_from > 0 && !append {
        println!("[ ] {name}: restarting (the server ignored the range request)");
    }

    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
        + if append { resume_from } else { 0 };

    let mut file = if append {
        OpenOptions::new()
            .append(true)
            .open(&part)
            .map_err(|e| format!("{name}: cannot open {}: {e}", part.display()))?
    } else {
        File::create(&part).map_err(|e| format!("{name}: cannot create {}: {e}", part.display()))?
    };

    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut done = if append { resume_from } else { 0 };
    let mut last_print = std::time::Instant::now();

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|e| format!("{name}: read failed: {e}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|e| format!("{name}: write failed: {e}"))?;
        done += read as u64;

        // Throttled: a progress line every 500 ms, not every read.
        if total > 0 && last_print.elapsed() >= std::time::Duration::from_millis(500) {
            print_progress(name, done, total);
            last_print = std::time::Instant::now();
        }
    }
    file.flush().ok();
    if total > 0 {
        print_progress(name, done, total);
        println!();
    }

    if done == 0 {
        return Err(format!("{name}: the server returned an empty body"));
    }

    fs::rename(&part, target)
        .map_err(|e| format!("{name}: cannot move the download into place: {e}"))?;
    println!("[+] {name:<18} {} downloaded", human_size(done));
    Ok(())
}

fn print_progress(name: &str, done: u64, total: u64) {
    let percent = (done as f64 / total as f64 * 100.0).min(100.0);
    print!(
        "\r[ ] {name:<18} {:>9} / {:>9}  ({:>5.1}%)",
        human_size(done),
        human_size(total),
        percent
    );
    let _ = std::io::stdout().flush();
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Where the model bundle lives: next to the executable, else in the working
/// directory — the same lookup the engine uses for the model itself.
pub fn default_model_dir() -> PathBuf {
    let relative = PathBuf::from("model_onnx");
    if relative.exists() {
        return relative;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("model_onnx");
            if candidate.exists() {
                return candidate;
            }
        }
    }
    relative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_size_is_readable() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(2 * 1024 * 1024), "2.0 MB");
    }

    #[test]
    fn bundle_covers_the_onnx_export() {
        assert!(BUNDLE.contains(&"model.onnx"));
        assert!(BUNDLE.contains(&"model.onnx_data"));
        assert!(BUNDLE.contains(&"config.json"));
        assert!(BUNDLE.contains(&"tokenizer.json"));
    }
}

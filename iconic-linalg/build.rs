//! Detect system BLAS at build time.
//!
//! On Linux, emits `cargo:rustc-cfg=blas_system` when OpenBLAS is found.
//! The `blas` module uses this cfg to include the platform BLAS dispatch path.

fn main() {
    #[cfg(target_os = "linux")]
    {
        if has_openblas() {
            println!("cargo:rustc-cfg=blas_system");
        }
    }
}

#[cfg(target_os = "linux")]
fn has_openblas() -> bool {
    // Try pkg-config first.
    if std::process::Command::new("pkg-config")
        .args(["--exists", "openblas"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return true;
    }

    // Try a direct link test.
    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".into());
    let src = format!("{out_dir}/blas_probe.c");
    let exe = format!("{out_dir}/blas_probe");
    if std::fs::write(&src, "int main(void){return 0;}\n").is_err() {
        return false;
    }
    let ok = std::process::Command::new("cc")
        .args([&src, "-o", &exe, "-lopenblas", "-lm"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&exe);
    ok
}

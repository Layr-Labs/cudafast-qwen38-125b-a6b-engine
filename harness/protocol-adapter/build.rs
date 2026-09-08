//! Links the `ds4-engine` backend against `libds4qwen`, the shared library
//! `tools/ds4/build.sh` produces from the ds4 submodule's CUDA core objects
//! plus `ds4_shim/ds4_shim.c`. The default (mock) build links nothing.
//!
//! `DS4_LIB_DIR` names the directory holding `libds4qwen.so`. It defaults to
//! `<repo>/.build/ds4`, the staging directory `tools/ds4/build.sh` writes.
//! The path is baked in as an rpath so the staged binary runs without
//! `LD_LIBRARY_PATH`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ds4_shim/ds4_shim.h");
    println!("cargo:rerun-if-env-changed=DS4_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_DS4_ENGINE").is_none() {
        return;
    }
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = std::env::var("DS4_LIB_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("../../.build/ds4"));
    let lib_dir = std::fs::canonicalize(&lib_dir).unwrap_or(lib_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=ds4qwen");
}

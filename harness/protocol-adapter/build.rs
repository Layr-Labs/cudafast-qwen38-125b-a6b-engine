//! Links the `ds4-engine` backend against `libds4qwen`, the shared library
//! `tools/ds4/build.sh` produces from the ds4 submodule's CUDA core objects
//! plus `ds4_shim/ds4_shim.c`. The default (mock) build links nothing.
//!
//! `DS4_LIB_DIR` names the directory holding `libds4qwen.so`. It defaults to
//! `<repo>/.build/ds4`, the staging directory `tools/ds4/build.sh` writes.
//! The path is baked in as an rpath so the staged binary runs without
//! `LD_LIBRARY_PATH`. CUDA's library directories join the same legacy
//! `DT_RPATH`: benchd sanitizes the worker environment, and `DT_RUNPATH` is not
//! inherited while the loader resolves libds4qwen's libcudart/libcublas
//! dependencies.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ds4_shim/ds4_shim.h");
    println!("cargo:rerun-if-env-changed=DS4_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var_os("CARGO_FEATURE_DS4_ENGINE").is_none() {
        return;
    }
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = std::env::var("DS4_LIB_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("../../.build/ds4"));
    let lib_dir = std::fs::canonicalize(&lib_dir).unwrap_or(lib_dir);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    let cuda_home = std::path::PathBuf::from(
        std::env::var_os("CUDA_HOME").unwrap_or_else(|| "/usr/local/cuda".into()),
    );
    let target_lib = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("x86_64") => Some(cuda_home.join("targets/x86_64-linux/lib")),
        Ok("aarch64") => Some(cuda_home.join("targets/sbsa-linux/lib")),
        _ => None,
    };
    let mut runtime_dirs = vec![lib_dir];
    for dir in target_lib.into_iter().chain([cuda_home.join("lib64")]) {
        let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
        if dir.is_dir() && !runtime_dirs.contains(&dir) {
            runtime_dirs.push(dir);
        }
    }
    // GNU new dtags make -rpath a non-transitive DT_RUNPATH. The worker needs
    // this path one level deeper, while resolving libds4qwen's CUDA NEEDED
    // entries, so deliberately emit the transitive legacy DT_RPATH.
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    for dir in runtime_dirs {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=ds4qwen");
}

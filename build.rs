fn main() {
    // crispembed-sys exposes its freshly built native libraries only to direct
    // dependants. Add paths on the final executable; dependency link args alone
    // do not propagate to it on macOS.
    if let Ok(dir) = std::env::var("DEP_CRISPEMBED_LIB_DIR") {
        for path in [
            dir.clone(),
            format!("{dir}/ggml/src"),
            format!("{dir}/ggml/src/ggml-metal"),
            "@executable_path/../lib/bsearch".to_owned(),
        ] {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
        }
    }
}

fn main() {
    // Python symbols resolve at import time; lets a plain `cargo build --workspace` link on macOS.
    pyo3_build_config::add_extension_module_link_args();
}

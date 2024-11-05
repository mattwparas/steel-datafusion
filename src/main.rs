use steel_datafusion::build_module;

fn main() {
    build_module()
        .emit_package_to_file("libsteel_datafusion", "datafusion.scm")
        .unwrap()
}

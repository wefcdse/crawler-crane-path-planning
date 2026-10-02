fn main() {
    let config = include_str!("../../config_default.json");
    // 写文件
    std::fs::write("config.json", config).unwrap();
}

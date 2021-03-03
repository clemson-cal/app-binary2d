use std::path::Path;
use serde::{Serialize, Deserialize};




// ============================================================================
pub fn parent_directory(path_str: &str) -> String {
    match Path::new(&path_str).parent().and_then(Path::to_str) {
        None     => ".",
        Some("") => ".",
        Some(parent) => parent,
    }.into()
}

pub fn write_cbor<T: Serialize>(value: &T, path_str: &str) -> anyhow::Result<()> {
    println!("write {}", path_str);
    let file = std::fs::File::create(&path_str)?;
    let mut buffer = std::io::BufWriter::new(file);
    Ok(ciborium::ser::into_writer(&value, &mut buffer)?)
}

pub fn read_cbor<T: for<'de> Deserialize<'de>>(path_str: &str) -> anyhow::Result<T> {
    let file = std::fs::File::open(path_str)?;
    let buffer = std::io::BufReader::new(file);
    Ok(ciborium::de::from_reader(buffer)?)
}

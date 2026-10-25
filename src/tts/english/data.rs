//! The phonemizer's data, deflated in the binary (`data/README.md` names
//! the sources; `tools/tts_data.py` writes the files).

pub const US_GOLD: &[u8] = include_bytes!("data/us_gold.json.z");
pub const US_SILVER: &[u8] = include_bytes!("data/us_silver.json.z");
pub const GB_GOLD: &[u8] = include_bytes!("data/gb_gold.json.z");
pub const GB_SILVER: &[u8] = include_bytes!("data/gb_silver.json.z");
pub const TAGGER_WEIGHTS: &[u8] = include_bytes!("data/tagger_weights.json.z");
pub const TAGGER_TAGDICT: &[u8] = include_bytes!("data/tagger_tagdict.json.z");
pub const TAGGER_CLASSES: &[u8] = include_bytes!("data/tagger_classes.json.z");

pub fn inflate(bytes: &[u8]) -> Result<Vec<u8>, String> {
    miniz_oxide::inflate::decompress_to_vec_zlib(bytes)
        .map_err(|e| format!("cannot inflate the phonemizer data: {e:?}"))
}

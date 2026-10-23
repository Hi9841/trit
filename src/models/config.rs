//! The `config.json` of a model folder, read by key path. One reader for
//! every family, so each error names the missing key the same way.

use serde_json::Value;

pub struct Json(Value);

impl Json {
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text)
            .map(Self)
            .map_err(|e| format!("config.json is not JSON: {e}"))
    }

    /// The value at `path`, keys joined by dots (`encoder_config.hidden_size`);
    /// `Null` when a key is absent.
    pub fn get(&self, path: &str) -> &Value {
        path.split('.').fold(&self.0, |value, key| &value[key])
    }

    /// A whole number that the model needs.
    pub fn num(&self, path: &str) -> Result<usize, String> {
        self.get(path)
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("config.json has no {path}"))
    }

    pub fn float(&self, path: &str) -> Option<f64> {
        self.get(path).as_f64()
    }

    pub fn str(&self, path: &str) -> Option<&str> {
        self.get(path).as_str()
    }

    pub fn flag(&self, path: &str) -> Option<bool> {
        self.get(path).as_bool()
    }
}

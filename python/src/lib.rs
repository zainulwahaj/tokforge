//! Python bindings: `tokforge.Tokenizer` wraps one `tokforge::Engine` (imported here as
//! `tokforge_core`, since this crate's own module is also called `tokforge`). Every call that tokenizes
//! releases the GIL, and the batch methods spread their inputs over rayon's thread pool.

use std::path::PathBuf;

use pyo3::{
    exceptions::{PyOSError, PyValueError},
    prelude::*,
};
use rayon::prelude::*;
use tokforge_core::{CoreMode, EncodeOptions, Engine, Error, SpecialTokens};

fn value_error(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn options(special_tokens: &str, add_special_tokens: bool, core: &str) -> PyResult<EncodeOptions> {
    let special_tokens = match special_tokens {
        "literal" => SpecialTokens::Literal,
        "recognize" => SpecialTokens::Recognize,
        other => {
            return Err(PyValueError::new_err(format!(
                "special_tokens must be \"literal\" or \"recognize\", not {other:?}"
            )))
        }
    };
    let core = match core {
        "auto" => CoreMode::Auto,
        "backend" => CoreMode::Backend,
        "verify" => CoreMode::Verify,
        other => {
            return Err(PyValueError::new_err(format!(
                "core must be \"auto\", \"backend\" or \"verify\", not {other:?}"
            )))
        }
    };
    Ok(EncodeOptions {
        add_special_tokens,
        special_tokens,
        core,
    })
}

/// A tokenizer loaded from a Hugging Face `tokenizer.json`.
#[pyclass(module = "tokforge", frozen)]
struct Tokenizer {
    engine: Engine,
}

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    #[pyo3(signature = (path, *, special_tokens = "literal", add_special_tokens = false, core = "auto"))]
    fn from_file(
        py: Python<'_>,
        path: PathBuf,
        special_tokens: &str,
        add_special_tokens: bool,
        core: &str,
    ) -> PyResult<Self> {
        let options = options(special_tokens, add_special_tokens, core)?;
        let bytes = std::fs::read(&path)
            .map_err(|e| PyOSError::new_err(format!("{}: {e}", path.display())))?;
        let engine = py
            .detach(|| Engine::from_bytes(&bytes, options))
            .map_err(value_error)?;
        Ok(Self { engine })
    }

    #[staticmethod]
    #[pyo3(signature = (json, *, special_tokens = "literal", add_special_tokens = false, core = "auto"))]
    fn from_str(
        py: Python<'_>,
        json: &str,
        special_tokens: &str,
        add_special_tokens: bool,
        core: &str,
    ) -> PyResult<Self> {
        let options = options(special_tokens, add_special_tokens, core)?;
        let engine = py
            .detach(|| Engine::from_bytes(json.as_bytes(), options))
            .map_err(value_error)?;
        Ok(Self { engine })
    }

    /// Number of tokens in `text`.
    fn count(&self, py: Python<'_>, text: &str) -> PyResult<usize> {
        py.detach(|| self.engine.count(text)).map_err(value_error)
    }

    /// Token ids of `text`.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.detach(|| self.engine.encode_ids(text))
            .map_err(value_error)
    }

    /// Token ids of every text, computed in parallel; results keep the input order.
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        py.detach(|| {
            texts
                .par_iter()
                .map(|text| self.engine.encode_ids(text))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(value_error)
    }

    /// Token counts of every text, computed in parallel; results keep the input order.
    fn count_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<usize>> {
        py.detach(|| {
            texts
                .par_iter()
                .map(|text| self.engine.count(text))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(value_error)
    }

    #[pyo3(signature = (ids, *, skip_special_tokens = false))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        py.detach(|| self.engine.decode(&ids, skip_special_tokens))
            .map_err(value_error)
    }

    /// The longest prefix of `text` the engine finds within `max_tokens` (a source slice).
    fn truncate(&self, py: Python<'_>, text: &str, max_tokens: usize) -> PyResult<String> {
        py.detach(|| self.engine.truncate(text, max_tokens))
            .map(|chunk| chunk.text)
            .map_err(value_error)
    }

    /// Contiguous slices of `text`, each within `max_tokens`, that join back to `text`.
    fn chunk(&self, py: Python<'_>, text: &str, max_tokens: usize) -> PyResult<Vec<String>> {
        py.detach(|| self.engine.chunk(text, max_tokens))
            .map(|chunks| chunks.into_iter().map(|c| c.text).collect())
            .map_err(value_error)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.engine.artifact().vocab_size
    }

    /// SHA-256 of the original tokenizer.json bytes.
    #[getter]
    fn sha256(&self) -> &str {
        &self.engine.artifact().sha256
    }

    /// Which core serves this artifact: the custom byte-level core's name, or "backend".
    #[getter]
    fn core(&self) -> &str {
        &self.engine.artifact().core
    }

    fn __repr__(&self) -> String {
        let artifact = self.engine.artifact();
        format!(
            "Tokenizer(vocab_size={}, core={:?}, sha256={:?})",
            artifact.vocab_size,
            artifact.core,
            &artifact.sha256[..12]
        )
    }
}

#[pymodule]
fn tokforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

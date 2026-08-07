use anyhow::{anyhow, Result};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow::array::{Array, GenericStringArray};
use arrow::datatypes::DataType;
use serde::Deserialize;

pub struct FoundFile {
    pub path: PathBuf,
    pub safe_name: String,
}

/// Scans directories for parquet/jsonl/jsonl.gz files and computes safe names.
pub fn scan_inputs(dirs: &[PathBuf]) -> Result<Vec<FoundFile>> {
    let mut files = Vec::new();
    for dir in dirs {
        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path();
                if is_supported_input(path) {
                    let safe_name = path.strip_prefix(dir)?.to_string_lossy().replace(['/', '\\'], "__");
                    files.push(FoundFile {
                        path: path.to_path_buf(),
                        safe_name,
                    });
                }
            }
        }
    }
    Ok(files)
}

fn is_supported_input(path: &Path) -> bool {
    match path.extension().and_then(|s| s.to_str()) {
        Some("parquet" | "jsonl") => true,
        Some("gz") => path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|name| name.ends_with(".jsonl.gz")),
        _ => false,
    }
}

pub fn read_any_stream<F>(path: &Path, mut callback: F) -> Result<()>
where F: FnMut(&str, &str, &str) {
    read_any_examples(path, |example| {
        callback(&example.condition, &example.instruction, &example.response);
    })
}

pub fn read_any_examples<F>(path: &Path, callback: F) -> Result<()>
where F: FnMut(Example) {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "parquet" => read_parquet_examples(path, callback),
        "jsonl" => read_jsonl_examples(path, callback),
        "gz" if path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|name| name.ends_with(".jsonl.gz")) =>
        {
            read_jsonl_gz_examples(path, callback)
        }
        _ => Err(anyhow!("Unsupported extension: {}", ext)),
    }
}

// --- Zero-Copy Readers ---

fn read_jsonl_examples<F>(path: &Path, mut callback: F) -> anyhow::Result<()>
where F: FnMut(Example) {
    let file = File::open(path)?;
    read_jsonl_reader(BufReader::new(file), &mut callback)
}

fn read_jsonl_gz_examples<F>(path: &Path, mut callback: F) -> anyhow::Result<()>
where F: FnMut(Example) {
    let file = File::open(path)?;
    let gz = MultiGzDecoder::new(file);
    read_jsonl_reader(BufReader::new(gz), &mut callback)
}

#[derive(Clone, Debug)]
pub struct Example {
    pub condition: String,
    pub instruction: String,
    pub response: String,
    pub prompt_messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct JsonRow {
    #[serde(default)]
    condition: Option<String>,
    #[serde(default)]
    instruction: Option<String>,
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    messages: Option<Vec<Message>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub reasoning_content: String,
}

fn read_jsonl_reader<R, F>(mut reader: BufReader<R>, callback: &mut F) -> anyhow::Result<()>
where
    R: Read,
    F: FnMut(Example),
{
    let mut line = String::new();
    let mut line_no = 0usize;
    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        line_no += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: JsonRow = serde_json::from_str(trimmed)
            .map_err(|e| anyhow!("JSON Error at line {}: {}", line_no, e))?;
        emit_json_row(row, callback);
    }
    Ok(())
}

fn emit_json_row<F>(row: JsonRow, callback: &mut F)
where F: FnMut(Example) {
    if let Some(response) = row.response {
        let condition = row.condition.unwrap_or_else(|| "direct".to_owned());
        let instruction = row.instruction.unwrap_or_default();
        callback(Example {
            prompt_messages: hrm_row_to_messages(&condition, &instruction),
            condition,
            instruction,
            response,
        });
        return;
    }

    let Some(messages) = row.messages else {
        return;
    };
    let mut history: Vec<Message> = Vec::new();
    for msg in messages {
        if msg.role.eq_ignore_ascii_case("assistant") && !msg.content.trim().is_empty() {
            let instruction = serialize_history(&history);
            let response = if msg.reasoning_content.trim().is_empty() {
                msg.content.clone()
            } else {
                format!("{}\n\n{}", msg.reasoning_content.trim(), msg.content.trim())
            };
            callback(Example {
                condition: "direct".to_owned(),
                instruction,
                response,
                prompt_messages: history.clone(),
            });
        }
        history.push(msg);
    }
}

fn hrm_row_to_messages(condition: &str, instruction: &str) -> Vec<Message> {
    let mut messages = Vec::new();
    if !condition.trim().is_empty() && condition != "direct" {
        messages.push(Message {
            role: "system".to_owned(),
            content: format!("Task condition: {}", condition.trim()),
            reasoning_content: String::new(),
        });
    }
    messages.push(Message {
        role: "user".to_owned(),
        content: instruction.to_owned(),
        reasoning_content: String::new(),
    });
    messages
}

fn serialize_history(messages: &[Message]) -> String {
    let mut chunks = Vec::new();
    for msg in messages {
        let content = msg.content.trim();
        if content.is_empty() {
            continue;
        }
        let label = match msg.role.to_ascii_lowercase().as_str() {
            "system" => "System",
            "user" => "User",
            "assistant" => "Assistant",
            "tool" => "Tool",
            _ => msg.role.as_str(),
        };
        chunks.push(format!("{}:\n{}", label, content));
    }
    chunks.join("\n\n")
}

fn read_parquet_examples<F>(path: &Path, mut callback: F) -> anyhow::Result<()>
where F: FnMut(Example) {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    for batch in reader {
        let batch = batch?;
        // We try both Utf8 (i32 offsets) and LargeUtf8 (i64 offsets)
        let c_col = batch.column_by_name("condition").ok_or_else(|| anyhow!("Missing condition"))?;
        let i_col = batch.column_by_name("instruction").ok_or_else(|| anyhow!("Missing instruction"))?;
        let r_col = batch.column_by_name("response").ok_or_else(|| anyhow!("Missing response"))?;

        // Inner processing loop macro to deduplicate code for DataType types
        macro_rules! process_batch {
            ($c_arr:expr, $i_arr:expr, $r_arr:expr) => {
                for i in 0..batch.num_rows() {
                    let c = $c_arr.value(i);
                    let inst = $i_arr.value(i);
                    let resp = $r_arr.value(i);
                    callback(Example {
                        condition: c.to_owned(),
                        instruction: inst.to_owned(),
                        response: resp.to_owned(),
                        prompt_messages: hrm_row_to_messages(c, inst),
                    });
                }
            }
        }

        match (c_col.data_type(), i_col.data_type(), r_col.data_type()) {
            (DataType::Utf8, DataType::Utf8, DataType::Utf8) => {
                process_batch!(
                c_col.as_any().downcast_ref::<GenericStringArray<i32>>().unwrap(),
                i_col.as_any().downcast_ref::<GenericStringArray<i32>>().unwrap(),
                r_col.as_any().downcast_ref::<GenericStringArray<i32>>().unwrap()
            );
            }
            (DataType::LargeUtf8, DataType::LargeUtf8, DataType::LargeUtf8) => {
                process_batch!(
                c_col.as_any().downcast_ref::<GenericStringArray<i64>>().unwrap(),
                i_col.as_any().downcast_ref::<GenericStringArray<i64>>().unwrap(),
                r_col.as_any().downcast_ref::<GenericStringArray<i64>>().unwrap()
            );
            }
            _ => {
                return Err(anyhow!("Skipping batch with mixed/unsupported string types in {:?}", path));
            }
        }
    }
    Ok(())
}

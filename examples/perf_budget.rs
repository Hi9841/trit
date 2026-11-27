//! Deterministic performance-budget gate for CI (`bench/perf-budget.json`).
//!
//! Run with:
//! `TRIT_PROFILE_FIXED_TEAM=1 TRIT_DEVICE=cpu TRIT_ISA=scalar cargo run --locked --release -p trit --features profile --example perf_budget -- bench/perf-budget.json`
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::process::ExitCode;

#[cfg(not(feature = "profile"))]
fn main() -> ExitCode {
    eprintln!("perf_budget requires `--features profile`");
    ExitCode::from(2)
}

#[cfg(feature = "profile")]
fn main() -> ExitCode {
    let budget_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench/perf-budget.json".to_string());

    if std::env::var("TRIT_PROFILE_FIXED_TEAM").as_deref() != Ok("1") {
        eprintln!("FAIL: set TRIT_PROFILE_FIXED_TEAM=1 for deterministic team sizing");
        return ExitCode::from(2);
    }

    let raw = match std::fs::read_to_string(&budget_path) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("FAIL: could not read {budget_path}: {err}");
            return ExitCode::from(1);
        }
    };
    let doc: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("FAIL: invalid JSON in {budget_path}: {err}");
            return ExitCode::from(1);
        }
    };

    let expected_threads = doc["fixed_team_threads"].as_u64().unwrap_or(16) as usize;
    let actual_threads = trit::threads(trit::Spread::AllCores);
    if actual_threads != expected_threads {
        eprintln!("FAIL: Spread::AllCores threads = {actual_threads}, expected {expected_threads}");
        return ExitCode::from(1);
    }

    let (peak_buf_bytes, conv_columns_tasks, output_len, output_hash) =
        match trit::perf_budget_probe() {
            Ok(v) => v,
            Err(err) => {
                eprintln!("FAIL: perf_budget_probe error: {err}");
                return ExitCode::from(1);
            }
        };

    let max_peak_buf_bytes = match doc["whisper_stem_peak_buf_bytes"].as_u64() {
        Some(v) => v,
        None => {
            eprintln!("FAIL: missing integer `whisper_stem_peak_buf_bytes` in {budget_path}");
            return ExitCode::from(1);
        }
    };
    let max_conv_columns_tasks = match doc["whisper_stem_conv_columns_tasks"].as_u64() {
        Some(v) => v,
        None => {
            eprintln!("FAIL: missing integer `whisper_stem_conv_columns_tasks` in {budget_path}");
            return ExitCode::from(1);
        }
    };
    let expected_output_len = doc["whisper_stem_output_len"].as_u64().unwrap_or(576_000) as usize;
    let expected_hash = match doc["whisper_stem_output_hash"].as_str() {
        Some(v) => v,
        None => {
            eprintln!("FAIL: missing string `whisper_stem_output_hash` in {budget_path}");
            return ExitCode::from(1);
        }
    };
    let actual_hash = format!("{output_hash:016x}");

    println!("perf_budget ({budget_path}):");
    println!("  threads: {actual_threads} (expected {expected_threads})");
    println!("  whisper_stem_peak_buf_bytes: {peak_buf_bytes} (ceiling {max_peak_buf_bytes})");
    println!(
        "  whisper_stem_conv_columns_tasks: {conv_columns_tasks} (ceiling {max_conv_columns_tasks})"
    );
    println!(
        "  whisper_stem_output_len: {output_len} (expected {expected_output_len}, hash {actual_hash}, expected {expected_hash})"
    );

    let mut failed = false;
    if output_len != expected_output_len {
        eprintln!("FAIL: output_len {output_len} != {expected_output_len}");
        failed = true;
    }
    if actual_hash != expected_hash {
        eprintln!("FAIL: whisper_stem_output_hash {actual_hash} != {expected_hash}");
        failed = true;
    }
    if peak_buf_bytes > max_peak_buf_bytes {
        eprintln!(
            "FAIL: whisper_stem_peak_buf_bytes {peak_buf_bytes} exceeds ceiling {max_peak_buf_bytes}"
        );
        failed = true;
    }
    if conv_columns_tasks > max_conv_columns_tasks {
        eprintln!(
            "FAIL: whisper_stem_conv_columns_tasks {conv_columns_tasks} exceeds ceiling {max_conv_columns_tasks}"
        );
        failed = true;
    }

    if failed {
        ExitCode::from(1)
    } else {
        println!("PASS");
        ExitCode::SUCCESS
    }
}

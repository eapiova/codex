//! Optional fail-open routing hook for idle native TUI turns.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use codex_app_server_protocol::Turn;
use codex_app_server_protocol::UserInput;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::openai_models::ReasoningEffort;
use serde::Deserialize;
use serde::Serialize;
use tokio::io::AsyncWriteExt;

const ROUTER_COMMAND_ENV: &str = "CODEX_TURN_ROUTER_COMMAND";
// Python enforces the configured routing budget (one to sixty seconds). This
// outer bound only covers process startup/cleanup around that budget.
const ROUTER_TIMEOUT: Duration = Duration::from_secs(65);
const RECORD_TIMEOUT: Duration = Duration::from_secs(5);
const HOST_ROUTER_CONTRACT_MARKER: &str = "Host-routed execution contract v1.";

#[derive(Serialize)]
struct TurnRouterRequest<'a> {
    version: u8,
    thread_id: String,
    prompt: String,
    cwd: &'a Path,
    model: &'a str,
    effort: Option<&'a ReasoningEffort>,
    collaboration_mode: &'a str,
    developer_instructions: Option<&'a str>,
}

#[derive(Serialize)]
struct TurnRouterCompletionRequest<'a> {
    version: u8,
    event: &'static str,
    thread_id: String,
    turn_id: &'a str,
    status: &'a codex_app_server_protocol::TurnStatus,
    duration_ms: Option<i64>,
}

#[derive(Serialize)]
struct TurnRouterStartRequest<'a> {
    version: u8,
    event: &'static str,
    thread_id: String,
    turn_id: &'a str,
    record_token: &'a str,
}

#[derive(Serialize)]
struct TurnRouterAbortRequest<'a> {
    version: u8,
    event: &'static str,
    record_token: &'a str,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TurnRouterDecision {
    pub(crate) apply: bool,
    pub(crate) model: Option<String>,
    pub(crate) effort: Option<ReasoningEffort>,
    pub(crate) developer_instructions: Option<String>,
    pub(crate) notice: Option<String>,
    pub(crate) record_token: Option<String>,
}

pub(crate) async fn route_idle_turn(
    thread_id: impl ToString,
    items: &[UserInput],
    cwd: &Path,
    model: &str,
    effort: Option<&ReasoningEffort>,
    collaboration_mode: Option<&CollaborationMode>,
) -> Option<TurnRouterDecision> {
    let command = std::env::var_os(ROUTER_COMMAND_ENV)?;
    if command.is_empty()
        || collaboration_mode
            .is_some_and(|mode| !matches!(mode.mode, ModeKind::Default | ModeKind::Plan))
    {
        return None;
    }
    let prompt = items
        .iter()
        .filter_map(|item| match item {
            UserInput::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if prompt.trim().is_empty() {
        return None;
    }
    let request = TurnRouterRequest {
        version: 1,
        thread_id: thread_id.to_string(),
        prompt,
        cwd,
        model,
        effort,
        collaboration_mode: collaboration_mode
            .map(|mode| match mode.mode {
                ModeKind::Plan => "plan",
                _ => "default",
            })
            .unwrap_or("default"),
        developer_instructions: collaboration_mode
            .and_then(|mode| mode.settings.developer_instructions.as_deref()),
    };
    let body = serde_json::to_vec(&request).ok()?;
    let child = tokio::process::Command::new(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .ok()?;
    exchange_with_router(child, &body, ROUTER_TIMEOUT).await
}

async fn exchange_with_router(
    child: tokio::process::Child,
    body: &[u8],
    timeout: Duration,
) -> Option<TurnRouterDecision> {
    let output = run_router_command(child, body, timeout).await?;
    serde_json::from_slice(&output).ok()
}

async fn run_router_command(
    mut child: tokio::process::Child,
    body: &[u8],
    timeout: Duration,
) -> Option<Vec<u8>> {
    let mut stdin = child.stdin.take()?;
    if stdin.write_all(body).await.is_err() || stdin.shutdown().await.is_err() {
        return None;
    }
    // Tokio's Unix ChildStdin shutdown is a no-op. Drop the pipe explicitly so
    // routers that read JSON until EOF can finish before the timeout.
    drop(stdin);
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() || output.stdout.len() > 16 * 1024 {
        return None;
    }
    Some(output.stdout)
}

pub(crate) fn record_turn_completion(thread_id: impl ToString, turn: &Turn) {
    let Some(command) = std::env::var_os(ROUTER_COMMAND_ENV) else {
        return;
    };
    if command.is_empty() {
        return;
    }
    let request = TurnRouterCompletionRequest {
        version: 1,
        event: "turn_completed",
        thread_id: thread_id.to_string(),
        turn_id: &turn.id,
        status: &turn.status,
        duration_ms: turn.duration_ms,
    };
    let Ok(body) = serde_json::to_vec(&request) else {
        return;
    };
    tokio::spawn(async move {
        let Ok(child) = tokio::process::Command::new(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
        else {
            return;
        };
        let _ = run_router_command(child, &body, RECORD_TIMEOUT).await;
    });
}

pub(crate) async fn bind_turn_start(record_token: &str, thread_id: impl ToString, turn_id: &str) {
    let Some(command) = std::env::var_os(ROUTER_COMMAND_ENV) else {
        return;
    };
    if command.is_empty() {
        return;
    }
    let request = TurnRouterStartRequest {
        version: 1,
        event: "turn_started",
        thread_id: thread_id.to_string(),
        turn_id,
        record_token,
    };
    let Ok(body) = serde_json::to_vec(&request) else {
        return;
    };
    let Ok(child) = tokio::process::Command::new(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    else {
        return;
    };
    let _ = run_router_command(child, &body, RECORD_TIMEOUT).await;
}

pub(crate) async fn abort_turn_start(record_token: &str) {
    let Some(command) = std::env::var_os(ROUTER_COMMAND_ENV) else {
        return;
    };
    if command.is_empty() {
        return;
    }
    let request = TurnRouterAbortRequest {
        version: 1,
        event: "turn_aborted",
        record_token,
    };
    let Ok(body) = serde_json::to_vec(&request) else {
        return;
    };
    let Ok(child) = tokio::process::Command::new(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    else {
        return;
    };
    let _ = run_router_command(child, &body, RECORD_TIMEOUT).await;
}

impl TurnRouterDecision {
    pub(crate) fn validated_route(&self) -> Option<(&str, &ReasoningEffort)> {
        if !self.apply {
            return None;
        }
        Some((self.model.as_deref()?, self.effort.as_ref()?))
    }
}

pub(crate) fn without_host_router_contract(instructions: &str) -> &str {
    if instructions.starts_with(HOST_ROUTER_CONTRACT_MARKER) {
        return "";
    }
    let Some(index) = instructions.rfind(&format!("\n{HOST_ROUTER_CONTRACT_MARKER}")) else {
        return instructions;
    };
    instructions[..index].trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_applied_decision_is_rejected() {
        let decision = TurnRouterDecision {
            apply: true,
            model: Some("gpt-5.6-sol".to_string()),
            effort: None,
            developer_instructions: None,
            notice: None,
            record_token: None,
        };
        assert!(decision.validated_route().is_none());
    }

    #[test]
    fn shadow_decision_never_has_an_applied_route() {
        let decision = TurnRouterDecision {
            apply: false,
            model: Some("gpt-5.6-sol".to_string()),
            effort: Some(ReasoningEffort::Ultra),
            developer_instructions: None,
            notice: None,
            record_token: None,
        };
        assert!(decision.validated_route().is_none());
    }

    #[test]
    fn prior_host_contract_is_removed_without_losing_user_instructions() {
        assert_eq!(
            without_host_router_contract(
                "Keep this.\nHost-routed execution contract v1. stale contract"
            ),
            "Keep this."
        );
        assert_eq!(
            without_host_router_contract("Host-routed execution contract v1. stale contract"),
            ""
        );
        assert_eq!(without_host_router_contract("Keep this."), "Keep this.");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn router_receives_eof_after_request_body() {
        let child = tokio::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(
                "cat >/dev/null; printf '%s' \
                 '{\"apply\":true,\"model\":\"gpt-5.6-sol\",\"effort\":\"medium\"}'",
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn test router");

        let decision =
            exchange_with_router(child, br#"{\"prompt\":\"test\"}"#, Duration::from_secs(1))
                .await
                .expect("router should observe EOF and return a decision");

        let (model, effort) = decision.validated_route().expect("valid applied route");
        assert_eq!(model, "gpt-5.6-sol");
        assert_eq!(effort, &ReasoningEffort::Medium);
    }
}

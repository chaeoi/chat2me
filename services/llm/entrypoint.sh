#!/bin/sh
set -eu

VOICE_ROLE="${VOICE_ROLE:-chat2me-llm}"

. /entrypoint-lib.sh

json_text_value() {
  key="$1"
  sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p"
}

json_number_value() {
  key="$1"
  sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\\([0-9][0-9]*\\).*/\\1/p"
}

print_progress() {
  label="$1"
  completed="$2"
  total="$3"
  status="$4"

  case "$completed" in ''|*[!0-9]*) completed=0 ;; esac
  case "$total" in ''|*[!0-9]*) total=0 ;; esac

  if [ "$total" -gt 0 ]; then
    awk -v label="$label" -v done="$completed" -v total="$total" -v status="$status" '
      BEGIN {
        width = 24
        pct = int(done * 100 / total)
        if (pct > 100) pct = 100
        filled = int(pct * width / 100)
        bar = ""
        for (i = 0; i < width; i++) bar = bar (i < filled ? "#" : "-")
        printf("[llm] %s [%s] %3d%% %.1f/%.1f MB %s\n", label, bar, pct, done / 1048576, total / 1048576, status)
      }'
  else
    awk -v label="$label" -v done="$completed" -v status="$status" '
      BEGIN {
        printf("[llm] %s %.1f MB %s\n", label, done / 1048576, status)
      }'
  fi
}

pull_model_with_progress() {
  pull_target_model="$1"
  fifo="/tmp/ollama-pull.$$"
  rm -f "$fifo"
  mkfifo "$fifo"

  curl -fsS -N \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$pull_target_model\",\"stream\":true}" \
    http://127.0.0.1:11434/api/pull >"$fifo" &
  curl_pid="$!"

  api_error=0
  last_key=""
  while IFS= read -r line; do
    error="$(printf '%s\n' "$line" | json_text_value error)"
    if [ -n "$error" ]; then
      echo "[llm] pull error: $error" >&2
      api_error=1
      continue
    fi

    status="$(printf '%s\n' "$line" | json_text_value status)"
    completed="$(printf '%s\n' "$line" | json_number_value completed)"
    total="$(printf '%s\n' "$line" | json_number_value total)"
    digest="$(printf '%s\n' "$line" | json_text_value digest)"

    if [ -n "$completed" ] && [ -n "$total" ] && [ "$total" -gt 0 ] 2>/dev/null; then
      pct=$((completed * 100 / total))
      bucket=$((pct / 5 * 5))
      key="${digest:-blob}:$bucket"
      if [ "$key" != "$last_key" ] || [ "$pct" -eq 100 ]; then
        print_progress "$pull_target_model" "$completed" "$total" "$status"
        last_key="$key"
      fi
    elif [ -n "$status" ]; then
      echo "[llm] $pull_target_model: $status"
    fi
  done <"$fifo"

  curl_status=0
  wait "$curl_pid" || curl_status="$?"
  rm -f "$fifo"

  [ "$curl_status" -eq 0 ] && [ "$api_error" -eq 0 ]
}

print_ollama_log_tail() {
  [ -f "$OLLAMA_LOG_FILE" ] || return 0
  echo "[llm] recent LLM log:" >&2
  tail -n 80 "$OLLAMA_LOG_FILE" >&2 || true
}

prefix_ollama_logs() {
  while IFS= read -r line || [ -n "$line" ]; do
    timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    printf '%s [info] [%s] [ollama] %s\n' "$timestamp" "$VOICE_ROLE" "$line" >> "$OLLAMA_LOG_FILE" 2>/dev/null || true
  done
}

init_config
load_runtime_env
OLLAMA_LOG_FILE="/app/log/$VOICE_ROLE.log"
mkdir -p "$(dirname "$OLLAMA_LOG_FILE")"
: "${OLLAMA_MODEL:?OLLAMA_MODEL must be set in runtime.env}"

model_ok() {
  check_target_model="$1"
  /bin/ollama show "$check_target_model" >/dev/null 2>&1 \
    && /bin/ollama run "$check_target_model" "只回答OK" >/dev/null 2>&1
}

ensure_model_ready() {
  ensure_target_model="$1"
  label="$2"

  echo "[llm] validating $label: $ensure_target_model"
  if model_ok "$ensure_target_model"; then
    echo "$ensure_target_model is ready"
    return 0
  fi

  echo "$ensure_target_model is missing or invalid; re-downloading"
  /bin/ollama rm "$ensure_target_model" >/dev/null 2>&1 || true

  until pull_model_with_progress "$ensure_target_model"; do
    echo "Retrying Ollama model pull: $ensure_target_model"
    sleep 30
  done

  echo "[llm] validating model after download: $ensure_target_model"
  if model_ok "$ensure_target_model"; then
    echo "$ensure_target_model is ready"
  else
    echo "$ensure_target_model was downloaded but failed runtime validation" >&2
    print_ollama_log_tail
  fi
}

OLLAMA_LOG_PIPE="/tmp/ollama-log.$$"
rm -f "$OLLAMA_LOG_PIPE"
mkfifo "$OLLAMA_LOG_PIPE"
prefix_ollama_logs <"$OLLAMA_LOG_PIPE" &
OLLAMA_LOGGER_PID="$!"

/bin/ollama serve >"$OLLAMA_LOG_PIPE" 2>&1 &
OLLAMA_PID="$!"

until /bin/ollama list >/dev/null 2>&1; do
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[llm] Ollama exited before it became ready" >&2
    print_ollama_log_tail
    exit 1
  fi
  sleep 2
done

(
  ensure_model_ready "$OLLAMA_MODEL" "local chat model"

  if [ -n "${INTENT_MODEL:-}" ]; then
    if [ "$INTENT_MODEL" = "$OLLAMA_MODEL" ]; then
      echo "[llm] local intent model matches local chat model: $INTENT_MODEL"
    else
      ensure_model_ready "$INTENT_MODEL" "local intent model"
    fi
  fi
) &

trap 'kill "$OLLAMA_PID" "$OLLAMA_LOGGER_PID" 2>/dev/null || true; rm -f "$OLLAMA_LOG_PIPE"' INT TERM
exec "$@"

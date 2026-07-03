#!/bin/sh
# Shared entrypoint helpers, sourced by config.sh / audio.sh / llm entrypoint.
# Copied into every image as /entrypoint-lib.sh.

DEFAULT_CONFIG_DIR="${DEFAULT_CONFIG_DIR:-/defaults/config}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-$CONFIG_DIR/runtime.env}"

normalize_log_level() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    debug|info|warning|error) printf '%s' "$1" | tr '[:upper:]' '[:lower:]' ;;
    warn) echo "warning" ;;
    err) echo "error" ;;
    *) echo "$2" ;;
  esac
}

log_level_value() {
  case "$(normalize_log_level "$1" info)" in
    debug) echo 10 ;;
    info) echo 20 ;;
    warning) echo 30 ;;
    error) echo 40 ;;
    *) echo 20 ;;
  esac
}

runtime_env_value() {
  key="$1"
  [ -f "$RUNTIME_CONFIG_PATH" ] || return 1
  sed -n "s/^${key}=//p" "$RUNTIME_CONFIG_PATH" | tail -n 1
}

chat2me_log() {
  level="$(normalize_log_level "${1:-info}" info)"
  message="$2"
  role="${VOICE_ROLE:-chat2me}"
  file_level="$(normalize_log_level "${CHAT2ME_LOG_LEVEL:-$(runtime_env_value CHAT2ME_LOG_LEVEL || true)}" info)"
  console_level="$(normalize_log_level "${CHAT2ME_CONSOLE_LOG_LEVEL:-$(runtime_env_value CHAT2ME_CONSOLE_LOG_LEVEL || true)}" warning)"
  log_dir="/app/log"
  level_value="$(log_level_value "$level")"
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  if [ "$level_value" -ge "$(log_level_value "$file_level")" ] && [ -n "$log_dir" ]; then
    mkdir -p "$log_dir" 2>/dev/null || true
    printf '%s [%s] [%s] %s\n' "$timestamp" "$level" "$role" "$message" >> "$log_dir/$role.log" 2>/dev/null || true
  fi
  if [ "$level_value" -ge "$(log_level_value "$console_level")" ]; then
    printf '[%s] %s: %s\n' "$role" "$level" "$message" >&2
  fi
}

init_config() {
  if [ ! -d "$DEFAULT_CONFIG_DIR" ]; then
    return
  fi

  mkdir -p "$CONFIG_DIR"
  for source_file in "$DEFAULT_CONFIG_DIR"/*; do
    [ -f "$source_file" ] || continue
    target_file="$CONFIG_DIR/$(basename "$source_file")"
    if [ ! -e "$target_file" ]; then
      cp "$source_file" "$target_file"
      chat2me_log info "Initialized config: $target_file"
    fi
  done
}

load_runtime_env() {
  [ -f "$RUNTIME_CONFIG_PATH" ] || return

  protected_keys="$(mktemp)"
  env | sed 's/=.*//' > "$protected_keys"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      ''|*[!A-Za-z0-9_]*) continue ;;
    esac

    if ! grep -Fxq "$key" "$protected_keys"; then
      export "$key=$value"
    fi
  done < "$RUNTIME_CONFIG_PATH"
  rm -f "$protected_keys"
}

#!/bin/sh
set -eu

MODELS_DIR="${MODELS_DIR:-/models}"
DEFAULT_MODELS_DIR="${DEFAULT_MODELS_DIR:-/opt/chat2me-default-models}"

. /entrypoint-lib.sh

default_voice_model_set() {
  case "$VOICE_ROLE" in
    chat2me-speech) echo "kws" ;;
    chat2me-asr) echo "asr-service" ;;
    chat2me-tts) echo "tts-service" ;;
    *) echo "kws,speech" ;;
  esac
}

model_selected() {
  case ",$MODEL_SET," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

lock_key() {
  printf '%s' "$MODEL_SET" | tr -c 'A-Za-z0-9_.-' '-'
}

now_seconds() {
  date +%s
}

path_mtime_seconds() {
  stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

process_start_time() {
  if [ -r "/proc/$1/stat" ]; then
    awk '{print $22}' "/proc/$1/stat" 2>/dev/null || true
  fi
}

model_lock_value() {
  if [ -f "$LOCK_DIR/$1" ]; then
    sed -n '1p' "$LOCK_DIR/$1" 2>/dev/null || true
  fi
}

model_lock_is_stale() {
  lock_stale_reason=""
  [ -d "$LOCK_DIR" ] || return 1

  now="$(now_seconds)"
  heartbeat="$LOCK_DIR/heartbeat"
  if [ -f "$heartbeat" ]; then
    heartbeat_mtime="$(path_mtime_seconds "$heartbeat")"
    heartbeat_age=$((now - heartbeat_mtime))
    if [ "$heartbeat_age" -le "$LOCK_STALE_SECONDS" ]; then
      return 1
    fi
    lock_stale_reason="heartbeat stale for ${heartbeat_age}s"
    return 0
  fi

  lock_pid="$(model_lock_value pid)"
  lock_start_time="$(model_lock_value start_time)"
  if [ -n "$lock_pid" ] && [ -n "$lock_start_time" ]; then
    current_start_time="$(process_start_time "$lock_pid")"
    if [ "$current_start_time" = "$lock_start_time" ]; then
      return 1
    fi
    lock_stale_reason="owner pid is not running"
    return 0
  fi

  lock_mtime="$(path_mtime_seconds "$LOCK_DIR")"
  lock_age=$((now - lock_mtime))
  if [ "$lock_age" -ge "$LOCK_STALE_SECONDS" ]; then
    lock_stale_reason="legacy lock is ${lock_age}s old"
    return 0
  fi
  return 1
}

write_model_lock_metadata() {
  hostname_value="$(hostname 2>/dev/null || echo unknown)"
  {
    echo "$LOCK_OWNER" > "$LOCK_DIR/owner"
    echo "$$" > "$LOCK_DIR/pid"
    process_start_time "$$" > "$LOCK_DIR/start_time"
    echo "$hostname_value" > "$LOCK_DIR/hostname"
    now_seconds > "$LOCK_DIR/created_at"
    echo "$MODEL_SET" > "$LOCK_DIR/model_set"
  }
}

start_model_lock_heartbeat() {
  touch "$LOCK_DIR/heartbeat"
  (
    while :; do
      touch "$LOCK_DIR/heartbeat" 2>/dev/null || exit 0
      sleep "$LOCK_HEARTBEAT_SECONDS"
    done
  ) &
  LOCK_HEARTBEAT_PID="$!"
}

cleanup_model_lock() {
  if [ -n "${LOCK_HEARTBEAT_PID:-}" ]; then
    heartbeat_pid="$LOCK_HEARTBEAT_PID"
    LOCK_HEARTBEAT_PID=""
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi

  if [ -n "${LOCK_DIR:-}" ] && [ -d "$LOCK_DIR" ]; then
    owner_on_disk="$(model_lock_value owner)"
    if [ "$owner_on_disk" = "${LOCK_OWNER:-}" ] || [ -z "$owner_on_disk" ]; then
      rm -f \
        "$LOCK_DIR/owner" \
        "$LOCK_DIR/pid" \
        "$LOCK_DIR/start_time" \
        "$LOCK_DIR/hostname" \
        "$LOCK_DIR/created_at" \
        "$LOCK_DIR/model_set" \
        "$LOCK_DIR/heartbeat"
      rmdir "$LOCK_DIR" 2>/dev/null || true
    fi
  fi
}

normalize_key() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

env_flag_enabled() {
  case "$(normalize_key "${1:-}")" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

known_value_error() {
  name="$1"
  value="$2"
  allowed="$3"
  echo "$name '$value' is not supported. Allowed values: $allowed" >&2
  exit 1
}

resolve_kws_model() {
  KWS_MODEL_NAME="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
  case "$KWS_MODEL_NAME" in
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20)
      KWS_MODEL_URL="${KWS_MODEL_URL:-}"
      KWS_MODEL_SHA256="${KWS_MODEL_SHA256:-68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6}"
      ;;
    *)
      known_value_error "KWS_MODEL" "$KWS_MODEL_NAME" "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
      ;;
  esac
}

resolve_asr_model() {
  VOICE_ASR_ENGINE="$(normalize_key "${VOICE_ASR_ENGINE:-sensevoice}")"
  case "$VOICE_ASR_ENGINE" in
    sensevoice)
      VOICE_ASR_ENGINE="sensevoice"
      VOICE_ASR_MODEL="${VOICE_ASR_MODEL:-sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09}"
      case "$VOICE_ASR_MODEL" in
        sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09)
          ASR_MODEL_URL="${ASR_MODEL_URL:-}"
          ASR_MODEL_SHA256="${ASR_MODEL_SHA256:-7305f7905bfcf77fa0b39388a313f3da35c68d971661a65475b56fb2162c8e63}"
          ;;
        *)
          known_value_error "VOICE_ASR_MODEL" "$VOICE_ASR_MODEL" "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
          ;;
      esac
      ;;
    *)
      known_value_error "VOICE_ASR_ENGINE" "$VOICE_ASR_ENGINE" "sensevoice"
      ;;
  esac
}

resolve_tts_model() {
  VOICE_TTS_ENGINE="$(normalize_key "${VOICE_TTS_ENGINE:-melotts}")"
  case "$VOICE_TTS_ENGINE" in
    melotts)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-MeloTTS-Chinese}"
      case "$VOICE_TTS_MODEL" in
        MeloTTS-Chinese)
          MELOTTS_MODEL_SOURCES="${MELOTTS_MODEL_SOURCES:-modelscope:myshell-ai/MeloTTS-Chinese:master}"
          MELOTTS_REQUIRED_FILES="config.json,checkpoint.pth"
          MELOTTS_MODEL_SHA256S="${MELOTTS_MODEL_SHA256S:-config.json=d58b5acdab89ad2bbd65325affab309ae3cb964834b02f9a60587474e81c8bb9,checkpoint.pth=a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093}"
          ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "MeloTTS-Chinese"
          ;;
      esac
      ;;
    online)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-edge-tts}"
      case "$VOICE_TTS_MODEL" in
        edge-tts) ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "edge-tts"
          ;;
      esac
      ;;
    *)
      known_value_error "VOICE_TTS_ENGINE" "$VOICE_TTS_ENGINE" "melotts, online"
      ;;
  esac
}

resolve_homophone_replacer() {
  ASR_HOMOPHONE_REPLACER_ENABLED="${ASR_HOMOPHONE_REPLACER_ENABLED:-1}"
  ASR_HOMOPHONE_GENERATE_ON_START="${ASR_HOMOPHONE_GENERATE_ON_START:-1}"
  ASR_HOMOPHONE_CONFIG_PATH="${ASR_HOMOPHONE_CONFIG_PATH:-$CONFIG_DIR/homophones.yaml}"
  ASR_HOMOPHONE_DIR="${ASR_HOMOPHONE_DIR:-$MODELS_DIR/homophone}"
  ASR_HOMOPHONE_LEXICON="${ASR_HOMOPHONE_LEXICON:-$ASR_HOMOPHONE_DIR/lexicon.txt}"
  ASR_HOMOPHONE_RULE_FSTS="${ASR_HOMOPHONE_RULE_FSTS:-$ASR_HOMOPHONE_DIR/replace.fst}"
  ASR_HOMOPHONE_LEXICON_URL="${ASR_HOMOPHONE_LEXICON_URL:-}"
  ASR_HOMOPHONE_LEXICON_SHA256="${ASR_HOMOPHONE_LEXICON_SHA256:-978900e511bc481b8630cb6e4a573c12566fa092c366d5396e2c3823dec9dcb9}"
  ASR_HOMOPHONE_GENERATOR_PYTHON="${ASR_HOMOPHONE_GENERATOR_PYTHON:-/opt/homophone-fst/bin/python}"
  export ASR_HOMOPHONE_REPLACER_ENABLED
  export ASR_HOMOPHONE_GENERATE_ON_START
  export ASR_HOMOPHONE_CONFIG_PATH
  export ASR_HOMOPHONE_DIR
  export ASR_HOMOPHONE_LEXICON
  export ASR_HOMOPHONE_RULE_FSTS
  export ASR_HOMOPHONE_LEXICON_URL
  export ASR_HOMOPHONE_LEXICON_SHA256
  export ASR_HOMOPHONE_GENERATOR_PYTHON
}

find_pulse_cookie() {
  uid="$1"
  for cookie in /host-home/*/.config/pulse/cookie; do
    [ -r "$cookie" ] || continue
    cookie_uid="$(stat -c %u "$cookie" 2>/dev/null || true)"
    [ "$cookie_uid" = "$uid" ] || continue
    printf '%s' "$cookie"
    return 0
  done
  return 1
}

copy_pulse_cookie() {
  source_cookie="$1"
  pulse_dir="${XDG_RUNTIME_DIR:-/tmp}/chat2me-pulse"
  mkdir -p "$pulse_dir"
  chmod 700 "$pulse_dir" 2>/dev/null || true
  target_cookie="$pulse_dir/cookie"
  cp "$source_cookie" "$target_cookie"
  chmod 600 "$target_cookie"
  printf '%s' "$target_cookie"
}

configure_speech_audio_output() {
  [ "${VOICE_ROLE:-}" = "chat2me-speech" ] || return 0

  if ! aplay -L 2>/dev/null | grep -Fxq pulse; then
    echo "PulseAudio ALSA plugin is missing in the speech image" >&2
    exit 1
  fi

  for socket in /host-run/user/*/pulse/native /run/user/*/pulse/native; do
    [ -S "$socket" ] || continue
    uid="$(basename "$(dirname "$(dirname "$socket")")")"
    cookie="$(find_pulse_cookie "$uid" || true)"
    [ -n "$cookie" ] || continue
    cookie_copy="$(copy_pulse_cookie "$cookie" || true)"
    [ -n "$cookie_copy" ] || continue

    PULSE_SERVER="unix:$socket"
    PULSE_COOKIE="$cookie_copy"
    AUDIO_OUTPUT_DEVICE="pulse"
    export PULSE_SERVER PULSE_COOKIE AUDIO_OUTPUT_DEVICE
    if pactl info >/dev/null 2>&1; then
      chat2me_log info "pulse audio output enabled: $socket"
      return
    fi
  done

  echo "PulseAudio output is required but no host PulseAudio socket/cookie was found" >&2
  exit 1
}

required_files_ok() {
  for required_file in "$@"; do
    if [ ! -s "$required_file" ]; then
      echo "Missing or empty model file: $required_file"
      return 1
    fi
  done
}

kws_runtime_ok() {
  python3 - "$KWS_MODEL" "$WAKE_WORDS" <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path

import sherpa_onnx

model_dir = sys.argv[1]
wake_words = [word.strip() for word in sys.argv[2].split(",") if word.strip()]
if not wake_words:
    print("WAKE_WORDS must contain at least one wake word", file=sys.stderr)
    sys.exit(1)


def create_keyword_spotter(keywords):
    return sherpa_onnx.KeywordSpotter(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
        decoder=f"{model_dir}/decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
        joiner=f"{model_dir}/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
        num_threads=1,
        keywords_file=str(keywords),
        provider="cpu",
    )

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_keywords = tmp_path / "keywords_raw.txt"
        keywords = tmp_path / "keywords.txt"
        raw_keywords.write_text("".join(f"{word} @{word}\n" for word in wake_words), encoding="utf-8")
        subprocess.run(
            [
                "sherpa-onnx-cli",
                "text2token",
                "--tokens",
                f"{model_dir}/tokens.txt",
                "--tokens-type",
                "phone+ppinyin",
                "--lexicon",
                f"{model_dir}/en.phone",
                str(raw_keywords),
                str(keywords),
            ],
            check=True,
        )

        create_keyword_spotter(keywords)
except Exception as exc:
    print(f"Invalid KWS model: {model_dir}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

asr_runtime_ok() {
  python3 - "$ASR_MODEL" "$AUDIO_SAMPLE_RATE" "${SENSEVOICE_LANGUAGE:-auto}" "${SENSEVOICE_USE_ITN:-1}" <<'PY'
import sys
from pathlib import Path
import sherpa_onnx

model_dir = sys.argv[1]
sample_rate = int(sys.argv[2])
language = sys.argv[3].strip().lower() or "auto"
use_itn_value = sys.argv[4].strip().lower()
if language not in {"auto", "zh", "en", "ja", "ko", "yue"}:
    print(f"Invalid SENSEVOICE_LANGUAGE: {language}", file=sys.stderr)
    sys.exit(1)
if use_itn_value in {"1", "true", "yes", "on"}:
    use_itn = True
elif use_itn_value in {"0", "false", "no", "off"}:
    use_itn = False
else:
    print(f"Invalid SENSEVOICE_USE_ITN: {use_itn_value}", file=sys.stderr)
    sys.exit(1)

model_path = Path(model_dir) / "model.int8.onnx"
if not model_path.is_file():
    model_path = Path(model_dir) / "model.onnx"


def create_recognizer():
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        tokens=f"{model_dir}/tokens.txt",
        model=str(model_path),
        num_threads=1,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        language=language,
        use_itn=use_itn,
        provider="cpu",
    )

try:
    create_recognizer()
except Exception as exc:
    print(f"Invalid ASR model: {model_dir}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

kws_model_ok() {
  required_files_ok \
    "$KWS_MODEL/tokens.txt" \
    "$KWS_MODEL/en.phone" \
    "$KWS_MODEL/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx" \
    "$KWS_MODEL/decoder-epoch-13-avg-2-chunk-8-left-64.onnx" \
    "$KWS_MODEL/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx" \
    && kws_runtime_ok
}

asr_model_ok() {
  required_files_ok \
    "$ASR_MODEL/tokens.txt" || return 1
  if [ ! -s "$ASR_MODEL/model.int8.onnx" ] && [ ! -s "$ASR_MODEL/model.onnx" ]; then
    echo "Missing or empty model file: $ASR_MODEL/model.int8.onnx" >&2
    return 1
  fi
  asr_runtime_ok
}

melotts_runtime_ok() {
  python_module_ok torch \
    && python_module_ok melo.api \
    && python_module_ok soundfile \
    && python_module_ok pypinyin \
    && python_module_ok cn2an \
    && python_module_ok jieba
}

melotts_model_ok() {
  required_relative_files_ok "$TTS_MODEL_DIR" "$MELOTTS_REQUIRED_FILES" \
    && melotts_runtime_ok
}

dir_has_files() {
  dir="$1"
  [ -d "$dir" ] || return 1
  find "$dir" -type f -print -quit | grep -q .
}

required_relative_files_ok() {
  base_dir="$1"
  required_files="$2"

  if [ -z "$required_files" ]; then
    dir_has_files "$base_dir"
    return
  fi

  old_ifs="$IFS"
  IFS=","
  set -- $required_files
  IFS="$old_ifs"
  for relative_file in "$@"; do
    relative_file="$(printf '%s' "$relative_file" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -n "$relative_file" ] || continue
    required_files_ok "$base_dir/$relative_file" || return 1
  done
}

python_module_ok() {
  python3 - "$1" <<'PY'
import importlib
import sys

try:
    importlib.import_module(sys.argv[1])
except Exception as exc:
    print(f"[runtime] Python module '{sys.argv[1]}' import failed: {exc}", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PY
}

missing_image_dependency() {
  echo "$1 is missing from the image. Rebuild the Docker image with the required dependency baked in." >&2
  exit 1
}

require_python_module() {
  module="$1"
  python_module_ok "$module" || missing_image_dependency "Python module '$module'"
}

require_command() {
  command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || missing_image_dependency "Command '$command_name'"
}

ensure_kws_runtime() {
  require_python_module sherpa_onnx
  require_python_module pypinyin
  require_command sherpa-onnx-cli
}

ensure_sensevoice_asr_runtime() {
  require_python_module sherpa_onnx
}

ensure_melotts_runtime() {
  if melotts_runtime_ok; then
    return
  fi
  missing_image_dependency "MeloTTS runtime"
}

ensure_online_tts_runtime() {
  require_python_module edge_tts
  require_command ffmpeg
}

ensure_selected_runtimes() {
  if model_selected kws; then
    ensure_kws_runtime
  fi

  if model_selected speech || model_selected asr-service; then
    case "$VOICE_ASR_ENGINE" in
      sensevoice) ensure_sensevoice_asr_runtime ;;
    esac
  fi

  if model_selected speech || model_selected tts-service; then
    case "$VOICE_TTS_ENGINE" in
      melotts) ensure_melotts_runtime ;;
      online) ensure_online_tts_runtime ;;
    esac
    if [ "$VOICE_TTS_ENGINE" = "online" ]; then
      ensure_melotts_runtime
    fi
  fi
}

content_length() {
  curl -fsSIL --retry 5 --connect-timeout 20 "$1" \
    | awk 'tolower($1) == "content-length:" { size = $2 } END { gsub("\r", "", size); print size }'
}

sha256_ok() {
  file="$1"
  expected="$2"
  [ -n "$expected" ] || return 0
  actual="$(sha256sum "$file" | awk '{print $1}')"
  if [ "$actual" = "$expected" ]; then
    return 0
  fi
  echo "SHA256 mismatch for $file: expected $expected, got $actual" >&2
  return 1
}

relative_file_sha256() {
  relative_file="$1"
  checksums="$2"
  old_ifs="$IFS"
  IFS=","
  set -- $checksums
  IFS="$old_ifs"
  for item in "$@"; do
    key="${item%%=*}"
    value="${item#*=}"
    if [ "$key" = "$relative_file" ] && [ "$key" != "$value" ]; then
      echo "$value"
      return 0
    fi
  done
}

copy_default_path() {
  source="$1"
  target="$2"
  label="$3"

  [ -e "$source" ] || return 1
  rm -rf "$target"
  mkdir -p "$(dirname "$target")"
  cp -a "$source" "$target"
  echo "[models] copied $label from image defaults"
}

copy_default_archive_model() {
  name="$1"
  target="$2"
  default_relative_path="${3:-$name}"
  source="$DEFAULT_MODELS_DIR/$default_relative_path"

  copy_default_path "$source" "$target" "$name"
}

copy_default_file() {
  source="$1"
  target="$2"
  label="$3"
  expected_sha256="$4"

  [ -s "$source" ] || return 1
  if ! sha256_ok "$source" "$expected_sha256"; then
    return 1
  fi
  mkdir -p "$(dirname "$target")"
  cp "$source" "$target"
  echo "[models] copied $label from image defaults"
}

print_download_progress() {
  label="$1"
  completed="$2"
  total="$3"

  case "$completed" in ''|*[!0-9]*) completed=0 ;; esac
  case "$total" in ''|*[!0-9]*) total=0 ;; esac

  if [ "$total" -gt 0 ]; then
    awk -v label="$label" -v done="$completed" -v total="$total" '
      BEGIN {
        width = 24
        pct = int(done * 100 / total)
        if (pct > 100) pct = 100
        filled = int(pct * width / 100)
        bar = ""
        for (i = 0; i < width; i++) bar = bar (i < filled ? "#" : "-")
        printf("[models] %s [%s] %3d%% %.1f/%.1f MB\n", label, bar, pct, done / 1048576, total / 1048576)
      }'
  else
    awk -v label="$label" -v done="$completed" '
      BEGIN {
        printf("[models] %s %.1f MB downloaded\n", label, done / 1048576)
      }'
  fi
}

download_with_progress() {
  output="$1"
  url="$2"
  label="$3"
  tmp="$output.download"
  max_attempts="${MODEL_DOWNLOAD_RETRIES:-10}"
  attempt=1

  mkdir -p "$(dirname "$output")"
  total="$(content_length "$url" || true)"
  echo "[models] downloading $label"

  while [ "$attempt" -le "$max_attempts" ]; do
    if [ -f "$tmp" ]; then
      completed="$(wc -c < "$tmp" | tr -d ' ')"
      echo "[models] resuming $label from $(awk -v done="$completed" 'BEGIN { printf("%.1f", done / 1048576) }') MB"
    else
      completed=0
    fi
    print_download_progress "$label" "$completed" "$total"

    curl -fL --retry 10 --retry-connrefused --connect-timeout 20 --speed-limit 1024 --speed-time 120 --continue-at - --silent --show-error "$url" -o "$tmp" &
    curl_pid="$!"

    (
      while kill -0 "$curl_pid" 2>/dev/null; do
        if [ -f "$tmp" ]; then
          completed="$(wc -c < "$tmp" | tr -d ' ')"
          print_download_progress "$label" "$completed" "$total"
        fi
        sleep 5
      done
    ) &
    progress_pid="$!"

    if wait "$curl_pid"; then
      curl_ok=1
    else
      curl_ok=0
    fi

    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true

    if [ "$curl_ok" -eq 1 ]; then
      completed="$(wc -c < "$tmp" | tr -d ' ')"
      print_download_progress "$label" "$completed" "$total"
      if [ "${total:-0}" -gt 0 ] && [ "$completed" -lt "$total" ]; then
        echo "[models] incomplete download for $label: $completed/$total bytes" >&2
      else
        mv "$tmp" "$output"
        return 0
      fi
    fi

    if [ "$attempt" -ge "$max_attempts" ]; then
      rm -f "$tmp"
      return 1
    fi
    echo "[models] retrying $label download ($attempt/$max_attempts)"
    attempt=$((attempt + 1))
    sleep 5
  done

  rm -f "$tmp"
  return 1
}

download_verified_with_progress() {
  output="$1"
  url="$2"
  label="$3"
  expected_sha256="$4"

  download_with_progress "$output" "$url" "$label"
  if ! sha256_ok "$output" "$expected_sha256"; then
    rm -f "$output"
    return 1
  fi
}

download_and_extract() {
  name="$1"
  url="$2"
  target="$3"
  expected_sha256="$4"
  archive="$MODELS_DIR/$name.tar.bz2"

  if [ -z "$url" ]; then
    echo "No runtime download URL configured for $name, and image defaults are missing or invalid." >&2
    echo "Set a trusted domestic URL with the corresponding SHA256, or rebuild the image with default models." >&2
    return 1
  fi

  echo "[models] preparing $name"
  rm -rf "$target"
  download_verified_with_progress "$archive" "$url" "$name" "$expected_sha256"
  echo "[models] extracting $name"
  tmp_extract_dir="$MODELS_DIR/.extract.$name"
  rm -rf "$tmp_extract_dir"
  mkdir -p "$tmp_extract_dir"
  python3 - "$archive" "$tmp_extract_dir" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:bz2") as archive:
    archive.extractall(sys.argv[2])
PY
  extracted_dir="$tmp_extract_dir/$name"
  if [ -d "$extracted_dir" ]; then
    mkdir -p "$(dirname "$target")"
    mv "$extracted_dir" "$target"
  else
    mkdir -p "$target"
    find "$tmp_extract_dir" -mindepth 1 -maxdepth 1 -exec mv {} "$target"/ \;
  fi
  rm -rf "$tmp_extract_dir"
  rm -f "$archive"
  echo "[models] extracted $name"
}

download_model_snapshot() {
  target="$1"
  label="$2"
  required_files="$3"
  sources="$4"
  checksums="$5"

  for source in $sources; do
    provider="${source%%:*}"
    rest="${source#*:}"
    repo_id="${rest%:*}"
    revision="${rest##*:}"
    if [ "$repo_id" = "$rest" ]; then
      revision=""
    fi

    if download_model_snapshot_from_source "$target" "$provider" "$repo_id" "$revision" "$label" "$required_files" "$checksums"; then
      return 0
    fi
    echo "[models] failed downloading $label via $provider:$repo_id" >&2
  done

  echo "failed to download $label from official sources" >&2
  return 1
}

download_model_snapshot_from_source() {
  target="$1"
  provider="$2"
  repo_id="$3"
  revision="$4"
  label="$5"
  required_files="$6"
  checksums="$7"

  rm -rf "$target"
  mkdir -p "$target"
  echo "[models] downloading $label from $(source_display_name "$provider") repo $repo_id"
  old_ifs="$IFS"
  IFS=","
  set -- $required_files
  IFS="$old_ifs"
  for relative_file in "$@"; do
    relative_file="$(printf '%s' "$relative_file" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -n "$relative_file" ] || continue
    expected_sha256="$(relative_file_sha256 "$relative_file" "$checksums")"
    download_model_file "$target/$relative_file" "$provider" "$repo_id" "$revision" "$relative_file" "$expected_sha256" || return 1
  done
}

source_display_name() {
  case "$1" in
    modelscope) echo "ModelScope" ;;
    *) echo "$1" ;;
  esac
}

download_model_file() {
  output="$1"
  provider="$2"
  repo_id="$3"
  revision="$4"
  relative_file="$5"
  expected_sha256="$6"

  case "$provider" in
    modelscope)
      url="https://modelscope.cn/models/$repo_id/resolve/${revision:-master}/$relative_file"
      ;;
    *)
      echo "unsupported model source provider: $provider" >&2
      return 1
      ;;
  esac

  download_verified_with_progress "$output" "$url" "$relative_file" "$expected_sha256"
}

ensure_archive_model() {
  name="$1"
  url="$2"
  check_name="$3"
  target="$4"
  expected_sha256="$5"
  default_relative_path="${6:-$name}"

  if "$check_name"; then
    echo "$name is ready"
    return
  fi

  if copy_default_archive_model "$name" "$target" "$default_relative_path" && "$check_name"; then
    echo "$name is ready"
    return
  fi

  echo "$name is missing or invalid; re-downloading"
  download_and_extract "$name" "$url" "$target" "$expected_sha256"

  echo "[models] validating $name"
  if ! "$check_name"; then
    echo "$name is still invalid after download" >&2
    exit 1
  fi
}

ensure_snapshot_model() {
  name="$1"
  sources="$2"
  check_name="$3"
  target="$4"
  required_files="$5"
  checksums="$6"

  if "$check_name"; then
    echo "$name is ready"
    return
  fi

  if copy_default_path "$DEFAULT_MODELS_DIR/$name" "$target" "$name" && "$check_name"; then
    echo "$name is ready"
    return
  fi

  echo "$name is missing or invalid; re-downloading"
  download_model_snapshot "$target" "$name" "$required_files" "$sources" "$checksums"

  echo "[models] validating $name"
  if ! "$check_name"; then
    echo "$name is still invalid after download" >&2
    exit 1
  fi
}

ensure_melotts_model() {
  ensure_snapshot_model \
    "$VOICE_TTS_MODEL" \
    "$MELOTTS_MODEL_SOURCES" \
    melotts_model_ok \
    "$TTS_MODEL_DIR" \
    "$MELOTTS_REQUIRED_FILES" \
    "$MELOTTS_MODEL_SHA256S"
}

enabled_value() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_homophone_replacer_resources() {
  enabled_value "$ASR_HOMOPHONE_REPLACER_ENABLED" || return

  mkdir -p "$ASR_HOMOPHONE_DIR"
  if [ ! -s "$ASR_HOMOPHONE_LEXICON" ]; then
    if ! copy_default_file "$DEFAULT_MODELS_DIR/homophone/lexicon.txt" "$ASR_HOMOPHONE_LEXICON" "homophone lexicon" "$ASR_HOMOPHONE_LEXICON_SHA256"; then
      if [ -z "$ASR_HOMOPHONE_LEXICON_URL" ]; then
        echo "No runtime download URL configured for homophone lexicon, and image defaults are missing or invalid." >&2
        exit 1
      fi
      download_verified_with_progress "$ASR_HOMOPHONE_LEXICON" "$ASR_HOMOPHONE_LEXICON_URL" "homophone lexicon" "$ASR_HOMOPHONE_LEXICON_SHA256"
    fi
  elif ! sha256_ok "$ASR_HOMOPHONE_LEXICON" "$ASR_HOMOPHONE_LEXICON_SHA256"; then
    echo "Homophone lexicon checksum validation failed: $ASR_HOMOPHONE_LEXICON" >&2
    exit 1
  fi

  if enabled_value "$ASR_HOMOPHONE_GENERATE_ON_START"; then
    if [ -s "$ASR_HOMOPHONE_CONFIG_PATH" ]; then
      case "$ASR_HOMOPHONE_RULE_FSTS" in
        *,*)
          echo "Cannot auto-generate multiple homophone rule FSTs: $ASR_HOMOPHONE_RULE_FSTS" >&2
          exit 1
          ;;
        *)
          if [ ! -x "$ASR_HOMOPHONE_GENERATOR_PYTHON" ]; then
            echo "Homophone FST generator Python is missing or not executable: $ASR_HOMOPHONE_GENERATOR_PYTHON" >&2
            exit 1
          fi
          "$ASR_HOMOPHONE_GENERATOR_PYTHON" /app/runtime/tools/homophone.py \
            "$ASR_HOMOPHONE_CONFIG_PATH" \
            "$ASR_HOMOPHONE_RULE_FSTS"
          chat2me_log info "Generated homophone rule FST: $ASR_HOMOPHONE_RULE_FSTS"
          ;;
      esac
    else
      echo "Homophone config is missing: $ASR_HOMOPHONE_CONFIG_PATH" >&2
      exit 1
    fi
  fi

  if [ ! -s "$ASR_HOMOPHONE_RULE_FSTS" ]; then
    echo "Homophone rule FST is missing: $ASR_HOMOPHONE_RULE_FSTS" >&2
    exit 1
  fi
}

init_config
load_runtime_env

VOICE_MODELS_REQUIRED="${VOICE_MODELS_REQUIRED:-1}"
VOICE_ROLE="${VOICE_ROLE:-}"
configure_speech_audio_output
resolve_kws_model
resolve_asr_model
resolve_tts_model
resolve_homophone_replacer
: "${WAKE_WORDS:?WAKE_WORDS must be set in runtime.env}"
: "${AUDIO_SAMPLE_RATE:?AUDIO_SAMPLE_RATE must be set in runtime.env}"
KWS_MODEL="$MODELS_DIR/$KWS_MODEL_NAME"
ASR_MODEL="$MODELS_DIR/$VOICE_ASR_ENGINE/$VOICE_ASR_MODEL"
TTS_MODEL_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
export VOICE_ASR_ENGINE
export VOICE_ASR_MODEL
export VOICE_TTS_ENGINE
export VOICE_TTS_MODEL
ORIGINAL_VOICE_ASR_ENGINE="$VOICE_ASR_ENGINE"
ORIGINAL_VOICE_ASR_MODEL="$VOICE_ASR_MODEL"
ORIGINAL_VOICE_TTS_ENGINE="$VOICE_TTS_ENGINE"
ORIGINAL_VOICE_TTS_MODEL="$VOICE_TTS_MODEL"

prepare_asr_download_target() {
  VOICE_ASR_ENGINE="$ORIGINAL_VOICE_ASR_ENGINE"
  VOICE_ASR_MODEL="$ORIGINAL_VOICE_ASR_MODEL"
  resolve_asr_model
  ASR_MODEL="$MODELS_DIR/$VOICE_ASR_ENGINE/$VOICE_ASR_MODEL"
}

prepare_tts_download_target() {
  if [ "$ORIGINAL_VOICE_TTS_ENGINE" = "online" ]; then
    VOICE_TTS_ENGINE="melotts"
    VOICE_TTS_MODEL="MeloTTS-Chinese"
  else
    VOICE_TTS_ENGINE="$ORIGINAL_VOICE_TTS_ENGINE"
    VOICE_TTS_MODEL="$ORIGINAL_VOICE_TTS_MODEL"
  fi
  resolve_tts_model
  TTS_MODEL_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
}

restore_runtime_model_selection() {
  VOICE_ASR_ENGINE="$ORIGINAL_VOICE_ASR_ENGINE"
  VOICE_ASR_MODEL="$ORIGINAL_VOICE_ASR_MODEL"
  VOICE_TTS_ENGINE="$ORIGINAL_VOICE_TTS_ENGINE"
  VOICE_TTS_MODEL="$ORIGINAL_VOICE_TTS_MODEL"
  resolve_asr_model
  resolve_tts_model
  ASR_MODEL="$MODELS_DIR/$VOICE_ASR_ENGINE/$VOICE_ASR_MODEL"
  TTS_MODEL_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
  export VOICE_ASR_ENGINE
  export VOICE_ASR_MODEL
  export VOICE_TTS_ENGINE
  export VOICE_TTS_MODEL
}

MODEL_SET="$(default_voice_model_set)"
if model_selected speech || model_selected asr-service; then
  case "$VOICE_ASR_ENGINE" in
    sensevoice) MODEL_SET="$MODEL_SET,asr" ;;
  esac
fi
if model_selected speech || model_selected tts-service; then
  case "$VOICE_TTS_ENGINE" in
    melotts) MODEL_SET="$MODEL_SET,melotts" ;;
    online) ;;
  esac
  if [ "$VOICE_TTS_ENGINE" = "online" ]; then
    MODEL_SET="$MODEL_SET,melotts"
  fi
fi
: "${LOCK_WAIT_LOG_SECONDS:?LOCK_WAIT_LOG_SECONDS must be set in runtime.env}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-600}"
LOCK_HEARTBEAT_SECONDS="${LOCK_HEARTBEAT_SECONDS:-5}"

if [ "$VOICE_MODELS_REQUIRED" != "1" ]; then
  exec "$@"
fi

mkdir -p "$MODELS_DIR"
LOCK_DIR="$MODELS_DIR/.download.$(lock_key).lock"
LOCK_OWNER="$(hostname 2>/dev/null || echo unknown)-$$-$(now_seconds)"
LOCK_HEARTBEAT_PID=""
lock_waited=0
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
  sleep 2
  lock_waited=$((lock_waited + 2))
  if [ "$lock_waited" -eq 2 ]; then
    echo "[models] waiting for voice model download lock: $MODEL_SET"
  elif [ "$LOCK_WAIT_LOG_SECONDS" -gt 0 ] && [ $((lock_waited % LOCK_WAIT_LOG_SECONDS)) -eq 0 ]; then
    echo "[models] still waiting for voice model download lock: $MODEL_SET (${lock_waited}s)"
  fi
  if model_lock_is_stale; then
    stale_owner="$(model_lock_value owner)"
    echo "[models] removing stale voice model download lock: $LOCK_DIR (${lock_stale_reason:-unknown})" >&2
    if [ -n "$stale_owner" ]; then
      current_owner="$(model_lock_value owner)"
      if [ "$current_owner" = "$stale_owner" ]; then
        rm -rf "$LOCK_DIR"
      fi
    else
      rm -rf "$LOCK_DIR"
    fi
    lock_waited=0
  fi
done
chat2me_log info "voice model download lock acquired: $MODEL_SET"
trap 'cleanup_model_lock' EXIT
trap 'cleanup_model_lock; exit 129' HUP
trap 'cleanup_model_lock; exit 130' INT
trap 'cleanup_model_lock; exit 143' TERM
write_model_lock_metadata
start_model_lock_heartbeat

ensure_selected_runtimes

if model_selected kws; then
  ensure_archive_model \
    "$KWS_MODEL_NAME" \
    "$KWS_MODEL_URL" \
    kws_model_ok \
    "$KWS_MODEL" \
    "$KWS_MODEL_SHA256" \
    "$KWS_MODEL_NAME"
fi

if model_selected asr; then
  prepare_asr_download_target
  ensure_archive_model \
    "$VOICE_ASR_MODEL" \
    "$ASR_MODEL_URL" \
    asr_model_ok \
    "$ASR_MODEL" \
    "$ASR_MODEL_SHA256" \
    "$VOICE_ASR_ENGINE/$VOICE_ASR_MODEL"
  resolve_homophone_replacer
  ensure_homophone_replacer_resources
fi

if model_selected melotts; then
  prepare_tts_download_target
  ensure_melotts_model
fi

trap - EXIT HUP INT TERM
cleanup_model_lock

restore_runtime_model_selection
exec "$@"

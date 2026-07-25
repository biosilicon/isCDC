#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${ISCDC_DEPLOY_SESSION:-iscdc}"
PORT="${ISCDC_DEPLOY_PORT:-5000}"
HOST="0.0.0.0"
PYTHON_BIN="${ISCDC_PYTHON:-/home1/shezixi/miniconda3/envs/iscdc/bin/python}"
DATABASE_PATH="${ISCDC_DATABASE_PATH:-${PROJECT_ROOT}/data/catalog.db}"
DATA_ROOT="${ISCDC_DATA_ROOT:-${PROJECT_ROOT}/data/datasets}"
LOG_PATH="${ISCDC_DEPLOY_LOG:-${PROJECT_ROOT}/data/iscdc-server.log}"
LOCAL_URL="http://127.0.0.1:${PORT}/"
PUBLIC_URL="http://10.138.46.171:${PORT}"

if [[ "${DATABASE_PATH}" != /* ]]; then
    DATABASE_PATH="${PROJECT_ROOT}/${DATABASE_PATH}"
fi
if [[ "${DATA_ROOT}" != /* ]]; then
    DATA_ROOT="${PROJECT_ROOT}/${DATA_ROOT}"
fi
if [[ "${LOG_PATH}" != /* ]]; then
    LOG_PATH="${PROJECT_ROOT}/${LOG_PATH}"
fi

usage() {
    cat <<'EOF'
Usage: ./deploy_test.sh {start|stop|restart|status|logs}

Commands:
  start    Start isCDC in a detached tmux session.
  stop     Stop the managed tmux session.
  restart  Stop and start the service.
  status   Check the tmux session, listening port, and HTTP endpoint.
  logs     Follow the latest server log output.

Optional environment variables:
  ISCDC_PYTHON          Python executable from the iscdc Conda environment.
  ISCDC_DEPLOY_PORT     Listening port (default: 5000).
  ISCDC_DEPLOY_SESSION  tmux session name (default: iscdc).
  ISCDC_DEPLOY_LOG      Server log path (default: data/iscdc-server.log).
  ISCDC_DATABASE_PATH   Catalogue database path (default: data/catalog.db).
  ISCDC_DATA_ROOT       Dataset directory (default: data/datasets).
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

validate_configuration() {
    [[ "${SESSION_NAME}" =~ ^[A-Za-z0-9_-]+$ ]] || \
        die "ISCDC_DEPLOY_SESSION may contain only letters, numbers, underscores, and hyphens"
    [[ "${PORT}" =~ ^[0-9]+$ ]] || die "ISCDC_DEPLOY_PORT must be a number"
    local port_number=$((10#${PORT}))
    ((port_number >= 1 && port_number <= 65535)) || \
        die "ISCDC_DEPLOY_PORT must be between 1 and 65535"
}

session_exists() {
    tmux has-session -t "=${SESSION_NAME}" 2>/dev/null
}

listening_sockets() {
    ss -H -ltn "sport = :${PORT}" 2>/dev/null
}

http_is_healthy() {
    curl --noproxy '*' --fail --silent --show-error --max-time 2 --output /dev/null \
        "${LOCAL_URL}" >/dev/null 2>&1
}

show_recent_log() {
    if [[ -f "${LOG_PATH}" ]]; then
        printf '%s\n' "--- Recent server log ---" >&2
        tail -n 30 "${LOG_PATH}" >&2
    fi
}

validate_start_requirements() {
    [[ -x "${PYTHON_BIN}" ]] || die "iscdc Python executable is not available: ${PYTHON_BIN}"
    [[ -f "${DATABASE_PATH}" ]] || die "catalogue database is not available: ${DATABASE_PATH}"
    [[ -d "${DATA_ROOT}" ]] || die "dataset directory is not available: ${DATA_ROOT}"
    require_command tmux
    require_command ss
    require_command curl
    mkdir -p "$(dirname -- "${LOG_PATH}")"
}

start_service() {
    validate_start_requirements

    if session_exists; then
        if http_is_healthy; then
            printf 'isCDC is already running at %s (tmux session: %s).\n' \
                "${PUBLIC_URL}" "${SESSION_NAME}"
            return 0
        fi
        die "tmux session '${SESSION_NAME}' exists, but the HTTP health check failed; use restart"
    fi

    local listeners
    if ! listeners="$(listening_sockets)"; then
        die "could not inspect TCP port ${PORT}"
    fi
    [[ -z "${listeners}" ]] || die "TCP port ${PORT} is already in use"

    printf '\n[%s] Starting isCDC on %s:%s\n' \
        "$(date --iso-8601=seconds)" "${HOST}" "${PORT}" >>"${LOG_PATH}"

    local launch_command quoted_log
    printf -v launch_command '%q ' \
        exec env \
        "PYTHONPATH=${PROJECT_ROOT}/src" \
        "ISCDC_DATABASE_PATH=${DATABASE_PATH}" \
        "ISCDC_DATA_ROOT=${DATA_ROOT}" \
        "${PYTHON_BIN}" -m uvicorn iscdc.app:app \
        --app-dir "${PROJECT_ROOT}/src" \
        --host "${HOST}" \
        --port "${PORT}" \
        --workers 1
    printf -v quoted_log '%q' "${LOG_PATH}"
    launch_command+=">>${quoted_log} 2>&1"

    if ! tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_ROOT}" "${launch_command}"; then
        die "failed to create tmux session '${SESSION_NAME}'"
    fi

    local attempt
    for attempt in {1..40}; do
        if http_is_healthy; then
            printf 'isCDC started successfully: %s\n' "${PUBLIC_URL}"
            printf 'Logs: %s\n' "${LOG_PATH}"
            return 0
        fi
        if ! session_exists; then
            show_recent_log
            die "isCDC exited before becoming ready"
        fi
        sleep 0.25
    done

    tmux kill-session -t "=${SESSION_NAME}" 2>/dev/null || true
    show_recent_log
    die "isCDC did not become ready within 10 seconds"
}

stop_service() {
    require_command tmux

    if ! session_exists; then
        printf 'isCDC is not running (tmux session: %s).\n' "${SESSION_NAME}"
        return 0
    fi

    if ! tmux send-keys -t "${SESSION_NAME}" C-c; then
        tmux kill-session -t "=${SESSION_NAME}" 2>/dev/null || true
        die "could not send the graceful shutdown signal to tmux session '${SESSION_NAME}'"
    fi
    local attempt
    for attempt in {1..20}; do
        if ! session_exists; then
            printf 'isCDC stopped.\n'
            return 0
        fi
        sleep 0.25
    done

    tmux kill-session -t "=${SESSION_NAME}"
    printf 'isCDC tmux session was forcefully stopped after the graceful shutdown timed out.\n'
}

status_service() {
    require_command tmux
    require_command ss
    require_command curl

    if ! session_exists; then
        printf 'isCDC is not running (tmux session: %s).\n' "${SESSION_NAME}"
        return 1
    fi

    local listeners
    if ! listeners="$(listening_sockets)"; then
        printf 'isCDC session exists, but TCP port %s could not be inspected.\n' "${PORT}" >&2
        return 1
    fi
    if [[ -z "${listeners}" ]]; then
        printf 'isCDC session exists, but TCP port %s is not listening.\n' "${PORT}" >&2
        return 1
    fi
    if ! http_is_healthy; then
        printf 'isCDC session and port exist, but the HTTP health check failed.\n' >&2
        return 1
    fi

    printf 'isCDC is healthy: %s (tmux session: %s).\n' "${PUBLIC_URL}" "${SESSION_NAME}"
}

follow_logs() {
    require_command tail
    [[ -f "${LOG_PATH}" ]] || die "server log does not exist yet: ${LOG_PATH}"
    tail -n 100 -f "${LOG_PATH}"
}

validate_configuration

case "${1:-}" in
    start)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        start_service
        ;;
    stop)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        stop_service
        ;;
    restart)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        stop_service
        start_service
        ;;
    status)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        status_service
        ;;
    logs)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        follow_logs
        ;;
    -h|--help|help)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

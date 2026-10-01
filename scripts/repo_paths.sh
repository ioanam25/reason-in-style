#!/usr/bin/env bash
# Shared repo path helpers. Prefer sourcing scripts/env.sh in new scripts.
_repo_paths_init() {
  local _src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  SCRIPT_DIR="$(cd "$(dirname "${_src}")" && pwd)"
  # shellcheck source=env.sh
  source "${SCRIPT_DIR}/env.sh"
  CONFIG_SCAS="${REPO_ROOT}/configs/scas"
}

#!/usr/bin/env bash
# scripts/trigger_mix.sh — Manually trigger a mix generation via the API Gateway
# Usage: ./scripts/trigger_mix.sh [duration_minutes] [style_hint]

set -euo pipefail

DURATION=${1:-45}
STYLE=${2:-""}
API_URL=${NEBULA_API_URL:-"http://localhost:8000"}

PAYLOAD=$(cat <<EOF
{
  "requested_duration_minutes": ${DURATION},
  "style_hint": $([ -n "$STYLE" ] && echo "\"$STYLE\"" || echo "null")
}
EOF
)

echo "Triggering mix: duration=${DURATION}min style='${STYLE}'"
curl -s -X POST \
  "${API_URL}/mixes/generate" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" | python3 -m json.tool

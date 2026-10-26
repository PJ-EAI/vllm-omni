#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Check a deployed gateway from outside: health, auth, the model route, robots.
# Run it from a machine on the robots' side of the network to prove the public path.
#
#   URL=https://maas.example.com OPERATOR_KEY=... MODEL=pi05_piper bash verify_server.sh
set -euo pipefail

: "${URL:?set URL, e.g. https://maas.example.com}"
: "${OPERATOR_KEY:?set OPERATOR_KEY (keys/operator.env)}"
MODEL="${MODEL:-pi05_piper}"
URL="${URL%/}"
H="Authorization: Api-Key $OPERATOR_KEY"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "ok   $*"; }

curl -sf "$URL/healthz" >/dev/null || fail "$URL/healthz unreachable (DNS, TLS proxy or gateway down)"
ok "healthz"
code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Api-Key wrong" "$URL/v1/models")"
[ "$code" = 401 ] || fail "a wrong key got $code, expected 401"
ok "wrong key -> 401"
info="$(curl -sf -H "$H" "$URL/v1/models/$MODEL")" || fail "GET /v1/models/$MODEL failed (operator key, model name, adapter block)"
ok "model $MODEL: $info"
robots="$(curl -sf -H "$H" "$URL/v1/robots")" || fail "GET /v1/robots failed"
ok "connected robots: $robots"

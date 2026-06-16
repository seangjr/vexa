#!/bin/bash
# =============================================================================
# Vexa Lite - OpenTelemetry launch shim (LITE-ONLY)
# =============================================================================
# Usage:  otel-exec <service-name> <command> [args...]
#
# When OTEL_EXPORTER_OTLP_ENDPOINT is set (and OTEL is not explicitly disabled),
# the target command is launched under `opentelemetry-instrument` so FastAPI
# routes, outbound HTTP, Redis, SQLAlchemy/psycopg2, and logs are exported to
# the configured OTLP collector with no application code changes.
#
# When the endpoint is unset, the command runs UNCHANGED — identical to the
# pre-instrumentation behaviour. This makes OTLP a pure opt-in: clearing the
# endpoint env var is an instant kill-switch with no rebuild.
# =============================================================================
set -e

svc="$1"
shift

if [ -n "$OTEL_EXPORTER_OTLP_ENDPOINT" ] && [ "${OTEL_SDK_DISABLED:-false}" != "true" ]; then
    export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-$svc}"
    exec opentelemetry-instrument "$@"
fi

exec "$@"

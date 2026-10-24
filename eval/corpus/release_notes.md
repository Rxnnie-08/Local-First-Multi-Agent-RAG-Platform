# Orbit Release Notes

## 3.4.0 (2026-06-12)

Added `orbit doctor --json` for machine-readable health output. Added priority lanes: five lanes,
numbered 1 (highest) to 5.

## 3.3.0 (2026-03-02)

Webhook signing with the `X-Orbit-Signature` header. The dashboard gained a dark mode.

## 3.2.0 (2025-11-18)

TLS 1.2 was disabled; TLS 1.3 is now required. The default job timeout changed from 30 seconds to
45 seconds.

## 3.1.0 (2025-08-05)

Introduced the dead letter queue for jobs that exhaust their retries.

## 3.0.0 (2025-04-21)

First public release, with REST API v2.

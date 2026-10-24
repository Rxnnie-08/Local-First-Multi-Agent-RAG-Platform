# Orbit Security Guide

## Authentication

API tokens are 48 characters long and must be rotated every 30 days. Dashboard passwords must be
at least 14 characters long. An account is locked for 10 minutes after 5 failed login attempts.

## Transport and storage

Orbit requires TLS 1.3. TLS 1.2 has been disabled since release 3.2. Data at rest is encrypted
with AES-256-GCM.

## Audit and incidents

Audit logs are retained for 400 days. Security incidents are handled by severity. A severity 1
incident must be acknowledged within 15 minutes, severity 2 within 2 hours, and severity 3 by
the next business day.

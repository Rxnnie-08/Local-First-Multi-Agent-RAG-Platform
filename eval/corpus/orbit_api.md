# Orbit REST API Reference

All endpoints live under the base path `/v2`.

## Endpoints

`POST /v2/jobs` creates a job. `GET /v2/jobs/{id}` returns its status. `DELETE /v2/jobs/{id}`
cancels it. The maximum request body is 512 KB.

## Limits

Each token may make 120 requests per minute. Beyond that the API answers HTTP 429 and sets the
`Retry-After` header to the number of seconds to wait.

List endpoints return 50 items per page by default. The maximum page size is 250.

## Idempotency and webhooks

Send the `X-Orbit-Idempotency` header with a unique key when creating a job, so a retried request
does not create a duplicate. Idempotency keys are valid for 24 hours.

Webhook deliveries are signed. The `X-Orbit-Signature` header contains an HMAC-SHA256 of the
request body, computed with your webhook secret.

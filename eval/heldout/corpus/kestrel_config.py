"""Built-in constants of the Kestrel Gateway control plane."""

# How long a breaker stays open after tripping before it lets one probe request through.
BREAKER_COOLDOWN_SECONDS = 45

# Failures within the window that trip a breaker.
BREAKER_THRESHOLD = 12

# Longest request header block the gateway will parse; anything bigger is refused with HTTP 431.
MAX_HEADER_BYTES = 16384

# Health checks run on this interval for every upstream.
HEALTH_CHECK_INTERVAL_SECONDS = 7


def backoff_delay(attempt: int) -> int:
    """Seconds to wait before reconnecting to an upstream: doubles each attempt, capped at 60."""
    return min(2**attempt, 60)

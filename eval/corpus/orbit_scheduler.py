"""Lease and backoff logic for the Orbit scheduler."""

LEASE_SECONDS = 90
MAX_BACKOFF_SECONDS = 300
DEAD_LETTER_MAX_SIZE = 5000


def compute_backoff(attempt: int) -> int:
    """Seconds to wait before retry number `attempt`: 2 ** attempt, capped at 300 seconds."""
    return min(2**attempt, MAX_BACKOFF_SECONDS)


class LeaseManager:
    """Grants and renews job leases.

    A lease lasts LEASE_SECONDS. A healthy worker renews it with a heartbeat every 30 seconds.
    """

    HEARTBEAT_SECONDS = 30

    def __init__(self) -> None:
        self._leases: dict[str, float] = {}

    def grant(self, job_id: str, now: float) -> float:
        expires = now + LEASE_SECONDS
        self._leases[job_id] = expires
        return expires

    def is_expired(self, job_id: str, now: float) -> bool:
        return now >= self._leases.get(job_id, 0.0)


class DeadLetterQueue:
    """Holds jobs that exhausted their retries. Beyond DEAD_LETTER_MAX_SIZE entries the oldest
    entries are evicted."""

    def __init__(self) -> None:
        self._jobs: list[str] = []

    def add(self, job_id: str) -> None:
        self._jobs.append(job_id)
        if len(self._jobs) > DEAD_LETTER_MAX_SIZE:
            self._jobs.pop(0)

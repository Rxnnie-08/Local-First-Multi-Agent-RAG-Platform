# Orbit Handbook

Orbit is a job queue and scheduler for background work. This handbook covers the defaults you
will meet first. All values below apply to release 3.4.

## Defaults

The Orbit server listens on port 8421 by default. The port can be changed with the `port` key in
`orbit.toml` or with the environment variable ORBIT_PORT.

The default retry limit is 7 attempts. After the last attempt a failed job moves to the dead
letter queue.

The default job timeout is 45 seconds. A job that runs longer is cancelled and counted as a failed
attempt.

Each worker process runs 6 jobs at the same time by default (the concurrency setting).

The configuration file is `orbit.toml`, searched in `~/.config/orbit/` first and then in the
current directory. The default log level is warn; set ORBIT_LOG=debug for more detail.

## Command line

`orbit run` starts a worker. `orbit drain` stops the worker from accepting new jobs and waits up
to 120 seconds for running jobs to finish before it exits. `orbit doctor` checks connectivity and
prints a health score from 0 to 100. `orbit migrate` upgrades the database schema.

## Troubleshooting

Error ORB-1042 means a lease expired: a worker held a job longer than its lease. Raise
`lease_seconds` in `orbit.toml`, or make the job call the heartbeat function regularly.

Error ORB-2210 means the queue is full: it reached its maximum depth of 20000 jobs. Add workers
or raise `max_depth`.

Error ORB-3307 means a schema mismatch after an upgrade. Run `orbit migrate` and restart the
workers.

# Kestrel Gateway release notes

## 2.8

The default upstream timeout was lowered from 5000 to 2500 milliseconds. The response cache default is unchanged at 90 seconds. Plugins now have a time budget of 50 milliseconds.

## 2.7

Added `kestrel top`. Admin bearer tokens now expire after 12 hours instead of 24. TLS 1.1 support was removed in this release.

## 2.6

Added the circuit breaker. The default idle connection pool was raised from 40 to 80 connections per route.

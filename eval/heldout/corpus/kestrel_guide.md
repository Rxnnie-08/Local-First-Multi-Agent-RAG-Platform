# Kestrel Gateway Guide

Kestrel is an API gateway that sits in front of internal services. This guide describes release 2.8.

## Defaults

The gateway accepts client traffic on port 9310 by default. Change it with the `listen` key in `kestrel.yaml` or with the environment variable KESTREL_LISTEN.

The default upstream timeout is 2500 milliseconds. A request that waits longer for its upstream is answered with HTTP 504.

Each route may keep up to 80 idle connections to its upstream by default.

The response cache keeps entries for 90 seconds unless the upstream sends its own Cache-Control header.

The configuration file is `kestrel.yaml`. Kestrel looks in `/etc/kestrel/` first and then in the working directory. The default log level is info; set KESTREL_LOG=trace for packet-level detail.

## Command line

`kestrel serve` starts the gateway. `kestrel reload` applies a changed configuration without dropping connections. `kestrel lint` validates a configuration file and exits with status 2 if it finds a problem. `kestrel top` shows live per-route request rates.

## Troubleshooting

Error KG-4107 means the circuit breaker is open: too many upstream failures in a short time. The gateway rejects requests to that route until the cooldown ends. Check the upstream and wait, or raise `breaker.threshold`.

Error KG-5520 means a certificate has expired. Renew the certificate and run `kestrel reload`.

Error KG-6003 means the configuration has a route with no upstream. Run `kestrel lint` to find it.

## Authentication

Gateway admin calls need a bearer token. Tokens expire after 12 hours. Service-to-service callers use mutual TLS instead and never receive a bearer token.

## Plugins

Plugins run in a sandbox with a 50 millisecond time budget per request. A plugin that exceeds its budget is skipped for that request and the event is logged.

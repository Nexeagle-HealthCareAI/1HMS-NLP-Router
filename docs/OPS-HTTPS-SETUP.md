# Mapping nlp-api.nexeagle.com to the Prod VM (HTTPS)

Scope: prod only. The prod app VM (`151.185.45.67`) already runs the NLP Router
container on host port 5003 (see `.github/workflows/deploy-nlp.yml`'s `deploy-prod`
job). This maps `https://nlp-api.nexeagle.com` to that same container via the
reverse proxy that already handles TLS for `www.nexeagle.com` — the app itself
doesn't change, and this is the SAME Caddy instance, not a new one.

Not in scope here: dev environment (`151.185.45.77:5003` keeps using its raw
`http://IP:port` address, matching how NexEagleWebsite's dev environment does too
— see `NexEagleWebsite/docs/OPS-HTTPS-SETUP.md`).

## Why this matters beyond just "HTTPS"

NexEagleWebsite's prod deploy (`.github/workflows/deploy.yml`, `deploy-prod` job)
already hardcodes `NLP_ROUTER_BASE_URL='https://nlp-api.nexeagle.com'` into the
website container's environment — someone wired this in on 2026-07-21, ahead of
this actually being reachable. Until this Caddyfile change ships, that URL just
times out, `callNlpRouter()` in `app/api/search/parse/route.ts` silently swallows
the failure, and every search falls through to the Anthropic path — not broken,
but the NLP router's fast-path/feedback-loop is dead weight until this is done.

**DNS is already set up** (`nlp-api.nexeagle.com` already resolves to
`151.185.45.67` — verified `2026-08-04`, nothing to do for step 1 this time,
unlike the original website HTTPS setup this doc is modeled on). Port 443 is also
already open and answering (Caddy is running) — it just doesn't have a site block
for this hostname yet, so TLS handshakes for it currently fail.

## 1. Add a site block to the EXISTING Caddyfile

Run via SSH into `151.185.45.67`. Do **not** start a second Caddy container —
`/opt/caddy/Caddyfile` already exists and is bind-mounted into the running
`caddy` container (see `NexEagleWebsite/docs/OPS-HTTPS-SETUP.md` for how it got
there). Adding a second site block to the same file is all that's needed; Caddy
handles multiple certs/hostnames from one instance fine.

```bash
cat /opt/caddy/Caddyfile
# Expect to see the existing block:
#   www.nexeagle.com, nexeagle.com {
#       reverse_proxy localhost:8080
#   }
```

Append the new block (don't overwrite the file — add to it):

```bash
cat >> /opt/caddy/Caddyfile <<'EOF'

nlp-api.nexeagle.com {
    reverse_proxy localhost:5003
}
EOF
```

`localhost:5003` reaches the NLP Router container because both Caddy and it run
with `--network host` (same pattern as the website's `localhost:8080`).

## 2. Reload Caddy (no downtime for the existing site)

```bash
docker exec caddy caddy reload --config /etc/caddy/Caddyfile
```

`caddy reload` re-reads the file and requests a new Let's Encrypt cert for the
newly-added hostname without dropping the existing `www.nexeagle.com` connections
or touching that cert.

Check it worked:

```bash
docker logs caddy --tail 50
curl -I https://nlp-api.nexeagle.com/health
```

The first request may take a few seconds while Caddy completes the ACME
handshake for the new hostname. If it fails, `docker logs caddy` will say why —
almost always either the NLP Router container not actually running on :5003 yet
(check `docker ps --filter name=nlp-router-prod` and `curl localhost:5003/health`
on the VM itself), or port 80/443 blocked upstream of the VM (unlikely — the
website's HTTPS already proves those are open).

## 3. Redeploy NexEagleWebsite's prod container

The currently-running `nexeagle-website` prod container predates the
`NLP_ROUTER_BASE_URL` env var (last successful prod deploy: 2026-07-20; the var
was added 2026-07-21; the one deploy attempt since, 2026-07-24, failed before
reaching `docker run` — a `docker pull` layer-commit error on the VM, unrelated
to this change). So the running container doesn't have the var set at all yet —
a fresh successful deploy of `main` is needed to pick it up. Re-run
`deploy-nlp.yml`... er, `deploy.yml`'s `deploy-prod` job (workflow_dispatch, or
push to `main`) once steps 1–2 above are confirmed working. Worth confirming
Docker/containerd on the VM is healthy first, given the prior failure mode
(`df -h`, `docker system df`, maybe `docker system prune` if disk pressure is
the cause) — that failure had nothing to do with the NLP router, but it'll bite
this redeploy too if it's still unresolved.

## Notes

- This Caddyfile edit is a one-time, manually-managed change — it isn't part of
  `deploy-nlp.yml`'s per-push pipeline, and doesn't need to be. The `caddy`
  container keeps running (`--restart unless-stopped`) across NLP Router
  redeploys underneath it; nothing about the existing deploy pipeline changes.
- Caddy auto-renews both certs indefinitely; no further action needed after
  setup.
- The NLP Router remains reachable at `http://151.185.45.67:5003` directly too
  (unchanged) — this just adds a second, HTTPS-terminated way to reach the same
  container, same as the website's setup.

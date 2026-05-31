# ImmunoSense — Security & Threat Model

Living document. Phase 1 = wellness app (not clinical), HIPAA-aware in design.
Tested controls link to `server/tests/test_security.py`.

## Security layers — status

| Layer | Control | Status |
|---|---|---|
| Transport | TLS everywhere (Supabase + host) | built |
| Auth | Supabase JWT (ES256), JWKS-verified, user_id from token only | built + tested |
| API authorization | every query filters by authed user_id; no id in path/body trusted | built + tested |
| Audit | every health read/write logged with actor + trace id | built + tested |
| Consent / AI boundary | consent per-type; user_id + PHI stripped before TFM prompt | built |
| Data isolation | 4 schemas; email isolated from user_id | built |
| CORS | strict origin allowlist | built + tested |
| Rate limiting | per-IP/user, esp. auth + evaluate | built + tested |
| Security headers | CSP, HSTS, X-Frame-Options, etc. | built + tested |
| Row-Level Security | DB-level per-user guard (defense in depth) | designed, not enabled |
| Encryption at rest | EncryptedString seam (passthrough Phase 1); disk encryption via Supabase | partial |

## Threat model by entry point

### Authentication (the front door)
- Threats: missing/forged/expired tokens, blank-user auth.
- Controls: JWKS signature verification; `sub` claim → user_id; 401 on any
  failure. Dev-bypass (`X-Dev-User`) ONLY when `DEV_AUTH=1` — production sets
  `DEV_AUTH=0`.
- Tested: `TestAuthentication`.

### Authorization / IDOR / BOLA
- Threats: user A reading/writing user B's PHI; smuggling a different user_id
  in the request body or path.
- Controls: no endpoint accepts a user_id/record id from the client; identity
  comes only from the verified token; every query is scoped to that user_id.
  Body fields like `user_id`/`bucket_id` are not in the schemas and are ignored.
- Tested: `TestCrossUserIsolation` (including body-override attempt).

### Injection (free text → storage / model / UI)
- Threats: SQL injection via notes/meal text; stored XSS via text rendered in
  the UI; prompt injection in the free-text symptom note steering the TFM to
  leak identifiers or follow embedded instructions.
- Controls:
  - SQL: SQLAlchemy parameterizes all queries; user text is stored literally.
  - XSS: API returns text as JSON data (no transform). UI MUST render as text,
    never HTML; no `dangerouslySetInnerHTML`, no markdown-rendering of model
    output without sanitization.
  - Prompt injection: user_id + PHI stripped before the prompt leaves infra;
    model OUTPUT is display text only — never executed, never a DB command.
- Tested: `TestInjection` (SQL-ish stored verbatim, XSS stored verbatim, model
  explanation carries no identifiers, no cross-user id leak).

### Input abuse / DoS / denial-of-wallet
- Threats: oversized payloads; rapid evaluate calls running up Claude API cost.
- Controls (current): server doesn't crash on large input; rate limiting now
  caps request volume per client (tighter on evaluate/flare/consent).
- TODO: explicit length cap on free-text fields; a per-user daily evaluation
  budget once the live TFM is wired.
- Partially tested: `TestInputBounds` (no 500 on large input — placeholder).

### Photo upload
- Threats: malicious file content; EXIF location leakage; oversized uploads.
- Controls: signed-URL upload (bytes never transit the API); EXIF strip in the
  storage edge function; content-type pinned. (Live verification needs Supabase
  Storage — see api/README.md.)

## Known hardening items (tracked, not yet done)

1. **Opaque bucket_id.** `bucket_id` is `{user_id}_{date}_{slot}`, so the opaque
   user_id appears in every returned bucket_id, logs, and the UI. Not a
   cross-user leak (the caller owns it), but it couples the ID format to the
   identifier we isolated and risks incidental exposure if bucket_ids are shared
   or logged externally. Fix = hash/opaque bucket_id, decoupled from user_id.
   Deferred: touches the library bucketing core (+752 tests), data layer, API.

2. **Enable RLS** on the `health` schema (and `identity`/`audit`). The API
   enforces per-user access today; RLS adds DB-level defense in depth so an API
   bug can't cause cross-user reads. A Supabase SQL step.

3. **Rotate shared secrets.** The dev DB password and a test-user password were
   shared during setup. Rotate both before any real data. Use a secrets manager
   / env vars; never commit secrets (`token.txt`, `*.db` are gitignored).

4. **CORS + rate limit + security headers** — DONE (increment 2). Configure
   `CORS_ORIGINS` with your front-end origin(s) before the UI connects; tune
   `RATE_LIMIT_*` env vars as needed. Rate limiter is in-memory (per-process);
   for multi-instance deployment move counters to a shared store (e.g. Redis).

5. **Field-level encryption** (activate the EncryptedString seam) — Phase 2 /
   clinical upgrade.

## LLM red-teaming

- Conventional API: OWASP ZAP / Burp against a SANDBOX instance only (disposable
  DB, synthetic data, never production).
- LLM layer: a `promptfoo` or pytest injection suite run when the live TFM is
  wired (it's MockTFM by default). Assert: no identifier in output, no
  instruction-following, output is inert display text.
- Golden rule: only ever test against infrastructure you own, with synthetic
  data.

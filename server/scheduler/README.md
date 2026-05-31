# Scheduler — timezone-aware evaluation trigger

Triggers the normal-path bucket evaluation per user, at their local time. The
flare button still gives immediate evaluation anytime; this handles the routine
"reflect on your day" evaluation.

## Design principle (locked)

Events are stored and bucketed in **UTC** (stable, unambiguous, DST-proof — what
the whole library uses). Only **scheduling** and **presentation** are
timezone-aware. The scheduler computes each user's local target time and
converts it to the correct UTC instant (DST-correct via `zoneinfo`).

This means the verified core library is untouched; timezone logic lives only in
`server/scheduler/`.

## Cadence (configurable)

- **`daily`** (default): once per local day at a configured hour (default 21:00
  local — the "reflect on your day" moment).
- **`6h_block`**: at the end of each local 6h block (00/06/12/18 local). Finer
  granularity, available now via config.

## How it runs

`run_scheduler.py` does ONE pass of due evaluations and exits. Invoke it
periodically (e.g. hourly) from an external scheduler:

```cmd
:: daily cadence (default)
venv\Scripts\python.exe run_scheduler.py

:: 6h-block cadence
venv\Scripts\python.exe run_scheduler.py --cadence 6h_block

:: different local hour
venv\Scripts\python.exe run_scheduler.py --local-hour 20
```

Wire it to: Windows Task Scheduler, cron, or a cloud scheduled job (Render Cron,
Railway cron, etc.). Run it hourly — the runner decides per-user who is actually
due based on their timezone, so an hourly poll catches each user at their local
slot. Idempotent: a user is evaluated at most once per local slot.

Requires `DATABASE_URL` set (Supabase in prod).

## Verify

```cmd
venv\Scripts\python.exe verify_scheduler.py
venv\Scripts\python.exe -m pytest server\tests\test_scheduler.py -q
```
Expect `RESULT: 9/9` and `14 passed`.

## Phase 1 limitation (documented, not a bug)

Last-run tracking is **in-memory** in `SchedulerRunner`. This is correct for a
single long-lived polling process. If you deploy the scheduler as separate
short-lived cron invocations (each a fresh process), promote last-run tracking
to a small `scheduler_runs` table so idempotency survives across processes. For
Phase 1 single-worker deployment, in-memory is fine. The timezone math and
due-logic are unaffected either way.
```

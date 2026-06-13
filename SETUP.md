# ImmunoSense automation scripts

Six small `.bat` files that replace the repetitive manual commands. Put them in
your project root: `C:\Projects\immunosense\`.

## One-time setup

1. Copy all the `.bat` files + `SETUP.md` into `C:\Projects\immunosense\`.

2. Edit **`env.local.bat`** — paste your real `SUPABASE_SERVICE_ROLE_KEY`
   (from Supabase → Settings → API → service_role → Reveal). The other values
   are already filled in. THIS FILE IS GIT-IGNORED — never commit it.

3. Add this line to your `.gitignore` (so secrets never get committed):

       env.local.bat

   (commit.bat also refuses to commit it as a safety net.)

4. Settle ONE canonical web folder at `C:\Projects\immunosense\web`. If your web
   app currently lives elsewhere (Downloads, C:\temp), copy the latest version
   there once:

       robocopy "<current web folder>" "C:\Projects\immunosense\web" /E /XD node_modules dist

## Daily use

| Command | What it does |
|---|---|
| `run-api.bat` | Frees port 8000, loads all env+secrets, starts the API. |
| `run-web.bat` | Starts the web dev server (npm install if needed). |
| `test.bat` | Runs the full suite with a clean env (no false failures). |
| `test.bat server` | Runs just the server tests. |
| `apply.bat C:\temp\SomePackage` | Finds server/ and web/src inside an extracted package and copies them in, then tests. |
| `commit.bat "message"` | Secret-scans, stages server/web/SECURITY.md, commits, pushes. |

## Typical workflow when I hand you a new package

1. Extract the tarball somewhere (e.g. `C:\temp\NewPkg`).
2. `apply.bat C:\temp\NewPkg`   → copies it in + runs tests.
3. `run-api.bat`   (in one terminal)
4. `run-web.bat`   (in another terminal)
5. Test in the browser.
6. `commit.bat "what changed"`

## What these DON'T do (still manual / interactive)
- Supabase dashboard steps (buckets, migrations, RLS policies)
- Verifying the app looks right in the browser
- The migration command itself (run alembic manually — it's rare and needs care)
- Building the mobile app (that's new code, not a process to script)

## Notes
- Run `.bat` files by typing their name in cmd from the project root, or
  double-clicking in Explorer.
- `run-api.bat` and `run-web.bat` each occupy their own terminal (they run
  servers). Open two terminals.
- If `apply.bat` says "nothing copied," the package had an unexpected layout —
  fall back to manual xcopy and tell me the structure.

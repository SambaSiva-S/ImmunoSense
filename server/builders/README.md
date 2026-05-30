# Builders

Translate raw UI input (what the user taps / types / says) into the domain
objects the agents consume. This is the seam between the app and the agents.

| Builder | Input | Output | External data |
|---|---|---|---|
| `build_symptom_summary` | SymptomLog rows | `DailySymptomMoodSummary` | none |
| `build_biomarker_input` | reading + profile | `{demographics, reading}` | none |
| `DietaryPipeline.build_rollup` | meal logs | `DailyRollup` | NHANES caches |

## Symptom & biomarker (no setup)

```python
from server.builders import build_symptom_summary, build_biomarker_input

summary = build_symptom_summary(symptom_log_rows, date="2026-05-27",
                                user_id="u1", disease="SLE")

bio_input = build_biomarker_input(reading_row, profile)
```

Symptom: when several logs exist for one day, each field takes the most severe
(max) non-null value — a flare day is defined by its worst point.

Biomarker: `sex` uses the NHANES convention (1 = male, 2 = female). Missing
demographics default to age 40 / sex 1 / bmi 25.

## Dietary — one-time cache build

The dietary builder runs the real NHANES nutrition pipeline (food matching +
nutrient density + DII / glycemic load / triggers). It needs two cache files
built once from your NHANES dietary .XPT files. **The caches are multi-MB and
are NOT committed to the repo.**

### Build the caches (once per machine)

```cmd
venv\Scripts\python.exe build_dietary_caches.py ^
    --dr1iff "C:\path\to\P_DR1IFF.XPT" ^
    --drxfcd "C:\path\to\P_DRXFCD.XPT"
```

This writes `data\nhanes\density.pkl` and `data\nhanes\food_index.pkl` (override
the location with `--out-dir`). Expect ~5,254 food codes. Takes ~30 seconds.

`data\nhanes\*.pkl` should be in `.gitignore` (the package adds this).

### Use the dietary builder

```python
from server.builders import DietaryPipeline

pipeline = DietaryPipeline(
    "data/nhanes/density.pkl",
    "data/nhanes/food_index.pkl",
)
rollup = pipeline.build_rollup(meal_log_rows, date="2026-05-27",
                               age=45, sex=2, bmi=27.0)
```

### Extractor choice (text -> food list)

`DietaryPipeline` defaults to `MockExtractor` (deterministic, no API, lower
extraction quality). For higher-quality food extraction, pass a ClaudeHaiku
extractor:

```python
from immunosense.agents.dietary.sources.claude_haiku import ClaudeHaikuExtractor
pipeline = DietaryPipeline(density_path, index_path,
                           extractor=ClaudeHaikuExtractor(...))
```

The NHANES food-matching and all nutrition math are real either way; only the
food-name extraction quality differs. This is a runtime call to a third-party
API (cost + the same privacy boundary as the TFM) — opt in deliberately.

## Tests

`server/tests/test_builders.py` (15 tests). Dietary tests use tiny synthetic
cache fixtures in `server/tests/fixtures/`, so they pass without the real
NHANES files. For real-data verification, build the caches and run a manual
rollup.

# Backend Structure — Read This First

**The real app is `app/` — that's the only one that matters.**

Run it with:
```
cd backend
uvicorn app.main:app --reload
```

## Files prefixed with `_archive_` are dead — do not run or import them
These are kept only so nothing was silently deleted. Safe to manually
delete the whole backend folder's `_archive_*` files and the
`_stub_unused_app/` and `backend/` (nested) folders once you've verified
`app/` works correctly.

- `_archive_agent_superseded.py` — old bare Gemini wrapper, no retrieval logic
- `_archive_kimi_main_broken_import.py` — old entrypoint, imported the file above
- `_stub_unused_app/` — empty scaffold folder, never had real code in it
- `backend/` (the nested one) — leftover from a zip that got extracted one level too deep

## If you ever bring another AI tool (Kimi, etc.) into this project again
Point it at this file first, and at `app/`, so it doesn't create yet another
competing `main.py` / `agent.py` pair like last time.

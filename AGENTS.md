# Repository Instructions

These instructions apply to the entire repository and must be followed by Codex and other coding agents.

## Required layout

- Put production code only in `src/问题一/`, `src/问题二/`, `src/问题三/`, or `src/可视化/`.
- Put all tests and test outputs in `temp/tests/`.
- Put intermediate data in `temp/interim/`.
- Put logs, caches, previews, debugging files, one-off scripts, and exploratory work under `temp/`.
- Put all analysis, validation, modeling, and process reports under `docs/`, including reviewed reports.
- Treat a "delivery document" or "handoff document" as a Markdown context handoff for continuing the work in a new Codex conversation. Store it under `docs/` and include the current objective, completed work, important decisions, file locations, known issues, and recommended next steps.
- Do not create DOCX or PDF versions of delivery/handoff documents unless the user explicitly requests that format.
- Put only reviewed final figures, tables, and data outputs under `results/`; do not place narrative reports there.
- Keep paper figures and tables in `paper/figures/` and `paper/tables/`.
- Do not create `notebooks/`, `tests/`, `paper/final/`, or `paper/sections/`.

## Working rules

- Read `CONTRIBUTING.md` before adding or relocating files.
- Treat `data/raw/` as read-only unless the user explicitly requests a change.
- Do not force-add files ignored under `temp/`.
- Do not place generated artifacts in the repository root.
- Preserve user changes and avoid unrelated rewrites.
- Update documentation and dependencies when production behavior changes.

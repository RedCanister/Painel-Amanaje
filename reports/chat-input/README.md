# Chat Transcript Intake

This folder is the private local intake area for exported Painel Amanaje chat history used by the next-release completeness audit.

## Accepted Files

Place exported or copied project chat files here using one of these formats:

- `.md`
- `.txt`
- `.json`

Keep each file focused on project/product planning, implementation decisions, or deferred ideas. Descriptive filenames help later audit passes, for example `2026-05-visualization-roadmap.md` or `assistant-slm-planning.json`.

## Privacy Rule

Raw transcript files in this folder are ignored by git. Only this README is intended to be tracked.

The roadmap must cite transcript-backed findings only when a transcript file is present in this folder. If the folder has no transcript files, the audit must say that chat-history recovery is pending and rely only on repo-local evidence.

## Continual Use

For each release pass:

1. Add new transcript exports or design notes here.
2. Update `reports/next-release-completeness-roadmap.md`.
3. Record which ideas were promoted, deferred, or closed.
4. For deferred ideas, capture the reason and the trigger for revisiting them.

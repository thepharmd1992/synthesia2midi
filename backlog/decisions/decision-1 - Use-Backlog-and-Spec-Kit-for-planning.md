---
id: decision-1
title: Use Backlog and Spec Kit for planning
date: '2026-06-08 23:53'
status: Active
---

## Context

Synthesia2MIDI development is now moving through frequent UI, workflow, detection, and MIDI changes. Root docs and chat history are not enough planning memory.

## Decision

Use Backlog as the repo's planning memory and status surface. Use Spec Kit for non-trivial feature execution planning. Keep existing root docs canonical for architecture, setup, testing, and current state until specific content is deliberately migrated.

## Consequences

- Backlog tasks carry acceptance criteria, status, and final summaries.
- Backlog docs and decisions carry durable planning context.
- Spec Kit artifacts under `specs/` carry detailed feature specs, plans, and task breakdowns.
- `AGENTS.md` remains the top-level agent contract.
- Product tasks from other repos must not be copied into this repo.

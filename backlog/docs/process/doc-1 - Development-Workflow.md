---
id: doc-1
title: Development Workflow
type: guide
created_date: '2026-06-08 23:52'
---

# Development Workflow

Backlog is the planning memory. Spec Kit is the planning tool for non-trivial feature implementation. Existing root docs remain canonical for architecture, setup, testing, and current state until specific material is deliberately migrated.

## Source Of Truth Roles

| Surface | Role |
|---|---|
| `backlog/tasks/` | Current and historical work items with status and acceptance criteria |
| `backlog/docs/` | Durable workflow, product, and planning docs |
| `backlog/decisions/` | One durable decision per file |
| `specs/` | Spec Kit feature artifacts for non-trivial implementation work |
| `.specify/` | Spec Kit templates, scripts, integration metadata, and active feature pointer |
| `.agents/skills/` | Repo-installed Spec Kit skills |
| `.lab/` | Optional experiments and generated evidence that should not become product truth |
| `scratch/` | Ignored one-off working files |
| `AGENTS.md` | Top-level agent contract |
| `PROJECT_LOG.md` | Concise state handoff during transition to Backlog |

Do not maintain duplicate product truth in root docs and Backlog docs. Migrate deliberately.

## Normal Flow

1. Start with `git status --short --branch`.
2. Read `AGENTS.md`, the relevant Backlog task/docs/decision files, and `.specify/feature.json`.
3. If the request is ambiguous, discuss first and record only accepted decisions or parked questions.
4. Create or update one Backlog task for non-trivial work. Keep acceptance criteria outcome-based.
5. Use Spec Kit only when implementation needs a formal spec, plan, and task breakdown.
6. Implement against the Backlog task or active Spec Kit plan.
7. Verify with the gates in `docs/testing.md`.
8. Update Backlog docs, decisions, task status, and final summary when durable state changed.
9. Commit coherent slices after verification.

## Backlog And Spec Kit Relationship

Backlog tracks project status. Spec Kit tracks implementation detail.

Preferred shape:

- one meaningful Spec Kit feature maps to one Backlog parent task;
- the Backlog task links to the `specs/` directory;
- Spec Kit owns implementation substeps;
- Backlog owns status, acceptance criteria, and final result.

Do not mirror every Spec Kit checklist item into Backlog.

## Task Granularity

Backlog tasks are outcome units. Create a separate task only when the work can be paused, resumed, accepted, blocked, delegated, or parked independently.

Use task notes or Spec Kit tasks for phase checklists.

## Parking Lot

Use `backlog/docs/process/doc-3 - Parking-Lot.md` for parked ideas that are not accepted current work. Convert parked items into Backlog tasks only when they become real work.

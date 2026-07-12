# ADR 0001: Use a modular monorepo during the 2.0 migration

- Status: accepted
- Date: 2026-07-11

Bridge, MCP, Python, schemas, generated types, fixtures, and migrations need
atomic changes while the v1/v2 compatibility window is open. They remain in a
single Git repository with enforced dependency boundaries. Repository splits
may be reconsidered only after v1 removal and stable published contracts.

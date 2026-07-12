# ADR 0003: Separate player-control, training, and catalog capabilities

- Status: accepted
- Date: 2026-07-11

Player-visible control and privileged training/debug operations do not share an
implicit trust domain. A normal MCP process receives only player-control
capability. Training uses a separately advertised scoped capability. Catalog
export is removed from the game HTTP mutation surface.

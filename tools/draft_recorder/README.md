# Draft event recorder

This external recorder replaces the duplicate `sts2-draft-tracker` Harmony mod.
It subscribes to the bridge revision stream, reads canonical state, and records
deck/draft transitions under `STS2_ARTIFACT_ROOT/draft-events`.

The recorder never patches the game and never receives training capability.
It uses only player-control state. If the bridge cannot expose an offered-card
set as player-visible state, the event records the observed deck transition
without inventing an offer.

```powershell
python tools/draft_recorder/record.py
```

# Offline catalog export

The in-game `POST /static/export` endpoint has been retired. It performed
arbitrary filesystem writes and expensive reflection/serialization on the game
main thread. The Bridge now returns HTTP `410 Gone` for that legacy route.

`export_catalog.py` publishes the canonical repository `game-data` package from
an ordinary process. It:

1. requires an explicit output directory below `STS2_ARTIFACT_ROOT` (or the
   default `~/.sts2-artifacts` root);
2. rejects output inside, equal to, or above the source checkout;
3. validates every input byte count and SHA-256 against `game-data/manifest.json`;
4. copies only manifest-listed relative paths;
5. verifies copied hashes;
6. writes a provenance manifest; and
7. atomically renames a same-parent staging directory into place.

```powershell
$env:STS2_ARTIFACT_ROOT = '<ABSOLUTE_ARTIFACT_ROOT_OUTSIDE_CHECKOUT>'
python .\tools\catalog-export\export_catalog.py `
  --output 'catalogs\verified-game-data'
```

Use `--source` only to point at another canonical game-data package with the
same manifest format. Absolute destinations must still be below the selected
artifact root. The destination must be absent or empty; this tool never
overwrites a populated directory.

This command **packages** canonical data. It does not reflect a running game.
To refresh source data, run the offline `tools/game_data` generation and
manifest workflow first, review provenance, then export the verified package.

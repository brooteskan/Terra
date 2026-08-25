# Authored sculpt compatibility fixtures

These payloads are frozen outputs of the production `SculptStrokeParams` Serde
implementation at commit `3a6ca7c86881a5cc9d1f03795c8bf7f46748202e`.
They were emitted with `serde_json::to_string_pretty` from production model
values and copied unchanged. They must not be regenerated or hand-normalized as
the authored model moves into this crate.

| Fixture | SHA-256 | Contract |
| --- | --- | --- |
| `legacy_bounded_sculpt_params.json` | `177B942B702E8748E5BD1AEF6A7741107A9ECF06E3CFEFC1C2A5E62B2EAB5D7B` | Legacy untagged bounded-UV `strokes`; `world_metres_v1` is absent. |
| `world_metres_v1_sculpt_params.json` | `04F36C7858A3DF074DAC4C95422E8DF55D8BCB2E9BA1E60D949C9E7F91F470D0` | Empty bounded `strokes` plus stable UUID and fixed-origin `f64` world records under `world_metres_v1`. |

The structural tests in `../compatibility_fixtures.rs` guard the discriminating
fields until the typed round-trip and migration tests are added with the new
authored stores.

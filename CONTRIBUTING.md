# Contributing to Terra

Thanks for helping improve Terra. This document covers the basics for code contributions.

## Development setup

```bash
cargo build -p terra-app
cargo test --workspace
cargo run -p terra-app
```

Use a recent stable Rust toolchain. GPU features need a working wgpu backend (DX12 on Windows by default).

## Crate rules

| Crate | Must not depend on |
|-------|--------------------|
| `terra-world` | `terra-core`, evaluation, GPU, renderer, IO, or app crates |
| `terra-core` | `wgpu`, any UI crate |
| `terra-gui` | `terra-core` or other domain crates |

Keep domain content (layer kinds, presets, archetypes) in `terra-core` when practical. UI crates present and apply; they should not become a second catalog of truth.

## Pull request guidelines

- Prefer small, reviewable PRs (one concern: split a module, fix a bug, add a feature).
- Do not mix large refactors with feature work.
- Prefer modules under ~800–1000 lines; treat files over ~1500 lines as split candidates unless they are pure static data.
- Avoid decorative banner comments; document invariants briefly at module level.
- Prefer `Result` at fallible boundaries; reserve `expect` for true invariants with a clear reason.
- When adding a layer type, update the layer registry (and inspector family if needed) rather than hand-syncing multiple catalogs.

## Clone ratchet

Run the same pinned production clone gate used by CI with:

```bash
node tools/clone-audit/clone-audit.mjs check
```

The command scans `crates/*/src/**/*.rs` with jscpd 5.0.4 in mild mode at
12 non-comment lines and 100 tokens. It preserves source line numbers while
removing inline `#[cfg(test)]` items from production input. Tests, examples,
source test modules, and the removed inline items receive a separate non-blocking
report under `target/clone-audit/`.

The gate compares normalized clone fingerprints and absolute duplicated lines
with `tools/clone-audit/production-baseline.json`, and requires every accepted
group to have a complete entry in `tools/clone-audit/exceptions.json`. It fails
for a new or grown group, increased duplicated production LOC, a missing reason,
or a stale baseline/exception. Percentage is reported but is never a gate.

After consolidating a clone, run `update` and review both checked-in JSON files.
New debt should not be accepted unless its exception documents the boundary,
equivalence expectation, protecting invariant, and current audit cycle. Run the
detector's add/remove and unrelated-growth proof with:

```bash
node tools/clone-audit/clone-audit.mjs self-test
```

## Docs

- User-facing guides live under `docs/` (workflow, creating terrain, editor overview); the root [README](README.md) lists them.
- Prefer updating those guides when authoring UX changes; keep algorithm internals in code comments or module docs rather than new algorithm guides.

## Logs and diagnostics

Terra writes the same filtered records to an attached console and to a persistent
per-user log. On Windows, logs are under `%LOCALAPPDATA%\Terra\logs`; other
platforms use the local data directory reported by the operating system. Each
launch writes a new `log-YYYY-MM-DD_HH-MM-SS-mmm.log` file.

Terra retains the six most recent launch logs. If the directory or file cannot
be created, Terra reports the problem to the console and continues with
console-only logging.

The default filter records Terra messages at `info` and third-party dependencies
at `warn`. Set `RUST_LOG` before launch to change it without recompiling, for
example in PowerShell:

```powershell
$env:RUST_LOG = "terra_core=debug,terra_render=warn,terra_app=debug"
cargo run -p terra-app
```

For a useful bug report, include the Terra version, OS and GPU, reproduction
steps, and the timestamped log from the affected launch.
Include the evaluation token, quality, layer, and project involved when known.
Logs can contain local project paths, so review them before sharing publicly.

## License

By contributing, you agree that your contributions are licensed under the MIT License, as described in [LICENSE](LICENSE).

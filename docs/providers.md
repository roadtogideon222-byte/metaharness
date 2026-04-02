# Providers

## Provider Model

`metaharness` separates the optimization loop from the system that actually edits files.
That editing system is called a proposer backend.

Current status:

- `CodexExecBackend` is real and exercised in benchmark runs
- `FakeBackend` is deterministic and used for tests and smoke runs
- `GeminiCliBackend` exists as a scaffold and is not yet at parity with Codex

The current package is Codex-first with an extensible backend interface.
All real provider benchmark runs currently documented in this repository were executed through the Codex CLI path.
That includes hosted Codex and local Ollama models used through Codex.

## Hosted Codex

Hosted Codex is supported today.

Requirements:

- `codex` CLI installed
- authenticated session or API key
- network access to the provider

Important command:

```bash
uv run metaharness run examples/python_fixture_benchmark --backend codex --hosted --budget 1 --run-name hosted-codex
```

Why `--hosted` matters:

- some benchmark configs default to local Ollama
- `--hosted` clears the local-provider settings for that run

Current conclusion:

- hosted Codex is supported in the library today
- the remaining requirement is environment access and authentication

## Local Codex Over Ollama

This path is supported and has been exercised with:

- `gpt-oss:20b`
- `gpt-oss:120b`

Probe first:

```bash
uv run metaharness smoke codex examples/python_fixture_benchmark --probe-only --oss --local-provider ollama --model gpt-oss:20b
```

Run:

```bash
uv run metaharness run examples/python_fixture_benchmark --backend codex --oss --local-provider ollama --model gpt-oss:20b --proposal-timeout 240 --budget 1
```

## Current Provider Takeaways

Based on the recorded benchmark runs in this repository:

| Provider | Benchmark Result | Observed Pattern |
| --- | --- | --- |
| Hosted Codex | solved both real benchmarks in one proposal iteration | fastest high quality path so far |
| Ollama `gpt-oss:20b` | timed out on both real benchmarks at `240s` | useful for very small smoke runs, not reliable enough for the current real benchmarks |
| Ollama `gpt-oss:120b` | solved `python_fixture_benchmark` | slower than hosted Codex, but capable |

This means the project's current public benchmark evidence is centered on Codex.
Other coding-agent benchmark writeups may emphasize Claude Code or Opus, but those are not the provider paths currently documented in this repository.

## What To Use In Practice

If you want the most reliable current path:

- use hosted Codex for serious benchmark or project runs

If you want a local-only workflow:

- use `gpt-oss:20b` for quick scaffold smoke checks
- use `gpt-oss:120b` for more capable local proposal runs
- increase proposal timeout for the larger model

## Next Provider Work

The next provider milestone after the current Codex path is:

- Gemini parity
- Pi backend integration

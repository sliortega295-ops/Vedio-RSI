# Vedio-RSI development

The native Codex agent performs research, proposals and code changes. The
`video-fast-rollout` Skill plus `fastroll` CLI is the execution harness.

- Read `docs/FASTROLLOUT.md` before runtime changes.
- Existing experiment/report branches are immutable evidence, not generated test
  fixtures. Never overwrite a historical timing or quality contract.
- Tests: `python -m unittest discover -s tests -v`; no GPU, API key or network.
- Real execution requires time-bounded external GPU ownership authorization.
  The agent must not create an approved grant on its own. Simulation accepts
  only built-in demo/compare adapters and cannot execute user code or use CUDA.
- Do not change primary cold measurement into resident timing. Reference evidence
  can be reused; final candidate decisions cannot.
- Do not call subprocess isolation a security sandbox. External optimization
  repositories require review or a separately administered container boundary.
- Keep credentials, model assets, authorization files and generated artifacts out
  of Git. Do not print environment variables wholesale.

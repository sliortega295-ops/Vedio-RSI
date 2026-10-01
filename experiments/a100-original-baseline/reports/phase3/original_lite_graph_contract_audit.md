# Archived Cache visual-contract conflict: actual orchestration-lite path

The archived repository contains two incompatible Cache visual-review
instructions. This run used the archived `orchestration/bin/spawn_executor.py`
path: it creates the Cache worktree, then concatenates `seed_goal`,
`workflow/cache_ca/nodes/codex_executor/cache_scope.md`,
`orchestration/prompts/loop_and_gate_contract.md`, and the frozen baseline
into the Executor goal (`spawn_executor.py:212–227`). The Master follows
`orchestration/prompts/master.md:25–31,44–55` and starts/polls/resumes
Executors through the lite CLI. This is the **archived orchestration-lite**
path with a project-local Codex transport adaptation, not execution of the
separate workflow graph.

The shared lite loop (`loop_and_gate_contract.md:32–53`) explicitly directs
a lossy Cache Executor to run `plan_eval --no-gemini`, use *its own built-in
multimodal vision* to inspect aligned candidate/dense frames, and write
`visual_verdict.json` before retaining/delivering. The archived Master then
independently runs `verify_delivery` and visually inspects each delivered
lossy point (`master.md:44–55`). A genuine image input is still required
for each role; a montage command or an unsupported `view_image` self-report
does not satisfy either gate.

Conversely, the appended graph-specific Cache scope says after a full run
the Executor should exit and the explicit graph should run an independent
blind Codex visual reviewer; the Executor must not author that reviewer's
verdict (`cache_scope.md:33–44`, also
`workflow/cache_ca/nodes/codex_executor/callable_nodes.md:13–21`). That
graph reviewer is not invoked by the archived lite Master/Executor runtime
used here, and no graph-owned `codex_visual_verdict.json` or merged
`assess_verdict` may be claimed. A Cache Executor's own limited verdict
must not be labeled an independent blind graph verdict. The conflicting
text is present in its prompt; the resulting ambiguity and any Agent
behavior it caused are an original-Harness contract limitation, not proof
that the graph path was completed.

Therefore a successful lite DELIVERY→Master independent verify→integration
could support a qualified **archived orchestration-lite reproduction**.
It cannot by itself support an unqualified claim of completing the entire
archived graph workflow or every Cache graph-specific gate. This report
does not change the original prompts, validator, or candidate search.

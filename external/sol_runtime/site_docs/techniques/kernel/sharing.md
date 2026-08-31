# Branch sharing (CFG / STG)

CFG/STG run several guidance branches as one batch; before the perturbed branch
diverges, much of the work is identical across branches.

- **Block-0 self-attention sharing** — in block 0 (no mask/perturbation, equal
  inputs), run self-attention once for a representative branch and expand back.
- **Guidance-prefix sharing** — drop the redundant perturbed branch for the whole
  block prefix before the first STG divergence, then expand to the full batch at the
  divergence block.

Both only share where branches are provably equivalent, so the output is unchanged.

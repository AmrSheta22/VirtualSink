# Complete-block memory teacher and supervised read gate

Frozen release: `artifacts/memory-block-gate-20261014-r1`; remote root
`/cluster/users/alex116u1/data/memory-block-gate-20261014-r1`.
Jobs: validation 217071, training array 217072, report 217073.

The preceding frozen read sweep found that boosting memory mass alone did not
reproduce direct auxiliary recall. Its auxiliary decoder omitted the query
residual and MLP. This experiment separates decoder alignment from gate learning
in a 2×2 factorial, using three seeds (12 runs): legacy or complete-block teacher,
each with or without supervised memory-read log odds.

Raw attention forms physical scores from RoPE queries and cached RoPE keys,
plus one null score and three recurrent-memory scores. A joint softmax divides
attention between these branches. Evicted raw K/V pairs update LSTM cells;
forget gates retain older cells, input gates admit new content, and output gates
expose it. Memory scores use the raw query and projected exposed states. The
supervised read-gate logit is logsumexp(memory scores) minus
logsumexp(physical scores and null score), whose sigmoid equals total memory
attention mass. Balanced BCE targets 0.9 on the final retrieval query and 0.0001
on other positions. This uses the known synthetic query schedule, not a general
language-model routing oracle; ordinary evaluation receives no target gate.

The complete-block teacher assigns all final-query attention to a softmax over
the three memory slots, then uses the existing O projection, query residual,
MLP residual, final norm and tied LM head. It has no extra decoder parameters.
For the one-layer model, stored K/V and the query depend only on embeddings,
so this teacher exactly reconstructs the native final block at memory mass one.
A numerical test compares it to the actual forward with dominating memory
scores. The teacher is used only in auxiliary training and labeled diagnostics.

All runs use the same ordered examples, cache 8, hidden size 32, four query
heads, two KV heads, initial write bias -4, retention timescales 128/512/2048,
initial maximum virtual mass 0.0001, batch 32 and AdamW LR 0.003. The backbone
is frozen for 300 teacher-only warmup steps, then all active parameters train
for 1000 steps with ordinary CE plus retained teacher CE. Gate arms additionally
use BCE weight one throughout. All arms train the same memory parameters in
warmup, including read keys and score biases. Backbone equality is asserted
before adaptation, and Adam moments carry across phases. No transfer
recalibration erases the learned gate.

Each run consumes 41,600 labeled sequences and 1,081,600 unique input tokens;
the added losses reuse those sequences. Fixed held-out sets contain 512
examples per delay (24, 48, 96). Snapshots at 0, 300 and 1300 distinguish
ordinary attention, legacy decoding and forced full-block decoding. Interventions
disable writes, reads, memory values, or retention. Query and background memory
mass are reported separately. A successful ordinary retrieval claim requires
accuracy gains together with dependence on stored memory, rather than teacher
accuracy alone. This random one-layer task cannot establish Qwen language-model
quality, multi-layer stability or general query routing.

Sources are archived and SHA256 hashed before submission; old releases remain
unchanged. Slurm runs both CUDA validation and training, using the established
runtime and excluding comp013/comp016. Local verification passed 26 tests.

All 12 runs and the strict report completed successfully on 14 October 2026.
Complete-block teachers without gate loss reached 100%/95.5%/74.4% ordinary
recall at delays 24/48/96. Adding gate supervision produced selective routing
but lower mean recall (75.9%/74.5%/58.2%), with a weak third seed. Full findings,
interventions and source audits are saved in
`artifacts/memory-block-gate-20261014-r1/FINDINGS.md`. A supplemental local
checkpoint-gradient test brought local verification to 27 tests; the frozen
cluster release passed its original 26 tests.

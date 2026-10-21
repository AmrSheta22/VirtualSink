# Optimized Qwen ordinary-memory continuation

On 16 October 2026 the user authorized replacing legacy ordinary-memory
updates after a ten-minute delay, verifying a fully saved checkpoint before
resuming and verifying successful continuation. This supersedes the earlier
read-only monitoring restriction specifically for this migration.

Handoff release: `/cluster/users/alex116u1/data/qwen05-optimized-handoff-20261016-r2`.
Training data/artifacts remain at the original long-run root
`/cluster/users/alex116u1/data/qwen05-long-uptraining-20261014-r1`.

- Optimized primary: **217217**, replacing the first attempt scheduled not
  before 12:33:23 Cairo time. The corrected job started at 12:36:21.
- Optimized backup continuation: **217218**, after successful primary exit.
- Legacy primary **217102** is asked to finish its current update and save.
- Legacy queued continuation **217103** is cancelled only after preflight passes.
- Replacement report **217219** depends on 217218. It can still report INCOMPLETE
  because the teacher was deliberately stopped; that does not invalidate the
  two remaining arms.

The controller first pins the existing atomically saved checkpoint and fully
deserializes every model, optimizer and recurrent-state tensor on CPU. It
checks stable metadata, finite tensor values, original identity, two stream
states, model/cache dimensions, data cursor, input-token count and matching
training-log step. Only then does it cancel the queued legacy continuation and
send USR1 directly to the uniquely identified Python trainer in job 217102.
The original signal handler finishes the update and saves its latest state.

The controller waits for legacy job exit COMPLETED, then pins and fully checks
the final checkpoint again. The fresh checkpoint must match the new paused
record and have a timestamp after the stop request. If these checks fail,
no optimized writer is started. The final pinned checkpoint is retained as
`handoff-checkpoint.pt` for rollback. No large checkpoint is downloaded.

The frozen original training source is reused byte-for-byte. A wrapper replaces
only its RecurrentSinkLM constructor with the benchmarked
OptimizedRecurrentSinkLM subclass. Architecture, parameters, optimizer moments,
both stream states, data cursor, RNG, LR schedule, 2048-step/1,016,397-token
budget, validation selection and final ablations all continue. The original
checkpoint identity stays compatible; separate source manifests and migration
records document the implementation change explicitly.

There is only one training writer. A new startup record must report the exact
saved step, followed by three contiguous updates with finite CE, auxiliary CE,
gradient norms and timings and increasing token counts. Those checks write
`continuation-verified.json` in the handoff root and
`optimized-continuation.json` in the training root. The training child then
continues under the same Slurm job. Failures propagate as nonzero job exits.

The authoritative original jobs.tsv retains all old job IDs and appends the
replacement training and report IDs. The original teacher remains stopped; FIFO already completed. The
six-hour monitor was updated to follow 217217/217218. Its rate analysis excludes
pre-migration steps from optimized throughput estimates. Use
`python hpc/collect_qwen05_handoff.py` for read-only small-artifact collection.

## Handoff execution

The first attempt (217215, backup 217216) fully verified checkpoint 1152 but
failed before signalling the legacy trainer. Its nested srun inherited a 64 GB
memory request, exceeding the legacy job's 24 GB allocation. Legacy training
continued without lost state. The corrected r2 helper explicitly requests 256 MB.
The first backup was dependency-cancelled; original report 217106 ran early and
reported incomplete ordinary training. All records remain preserved, and
report 217219 now waits for the corrected optimized continuation.

The corrected stop request was delivered at 12:36:35 Cairo time. Legacy 217102
finished its current update, saved checkpoint 1157 and exited with code 0 at
12:37:49. Checkpoint 1157 was saved at 12:37:48 and fully verified at 12:38:07:
576,920 input tokens, cursor [129,2816], both memory streams at position 2816,
5,956,377,931 serialized bytes, 2193 finite tensors and 482 optimizer states.
Its immutable rollback link is retained in the r2 release.

The optimized trainer emitted startup at exactly step 1157 under job 217217.

At 12:40:09 the supervisor verified three consecutive optimized updates
1158–1160: CE 2.1574166 / 2.1055059 / 2.2624176, finite gradient norms
27.14446 / 26.57790 / 27.60125, and times 32.2181 / 31.7123 / 31.7014 seconds.
Their mean was 31.8773 seconds/update (about 16.06 input tokens/second), versus
roughly 158 seconds for legacy full updates. Input tokens increased by 512
each update; both stream/data cursors continued without resetting.

A subsequent check at 12:40:49 confirmed job 217217 RUNNING with zero restarts,
an empty current error log, and a fourth successful update at step 1161
(578,968 input tokens, context position 4864, CE 2.0587578, gradient norm
26.08276, 31.7086 seconds). Backup 217218 and report 217219 remain dependency
queued. `continuation-verified.json` preserves the three-update proof.

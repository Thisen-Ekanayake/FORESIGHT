# FORESIGHT — Plan for the next phase

Written 2026-11-02 after a full pass over the repo, logs and `Project_FORESIGHT.pdf`. Milestone status lives in
`PROGRESS.md`; this file is the *why and in what order*.

## Decisions this plan rests on (user, 2026-11-02)

| Question | Answer |
|---|---|
| What to optimise for | **Close the spec's research deliverables** (A → C → B), aiming at a report/paper. Side threads parked. |
| Deadline | None fixed: planned by milestones, each with a definition of done. |
| Task for the depth-error study (A) and stress benchmark (C) | **Point-goal** ("reach the doorway"): spec-native, short episodes, clean metrics, usable on all 10 minival scenes. Object search stays as a secondary, already-collected ablation. |
| Compute / data | Local RTX 4060, **rented cloud GPU** (RunPod; price quoted before anything is created), **Matterport HM3D token** (more semantic homes), GCP VM `finsight` (downloads, preprocessing, scoring). |
| Literature review | Done **with Athena** (`/ml/Athena`): research maps, field evolution, open problems, idea checks, plus a written feedback report on Athena itself. |

## Lit review outcome (2026-11-02, `docs/lit_review/`)

Done with Athena (six maps, gap themes, idea checks) and cross-checked on the web. What it changed:

- **A is the clearest gap. Kept and sharpened.** No work plots closed-loop navigation error against monocular
  depth error. NavTrust (IROS 2026) corrupts *sensor* depth at fixed severities, and LG-VLN (2026) claims navigation
  is insensitive to uniform scale error. New hypothesis H-A: uniform scale/noise errors degrade gracefully, while
  structured errors on reflective/textureless regions cause a disproportionate share of collisions. Curves are per
  error *type*, and NavTrust's four depth corruptions are added for comparability.
- **C narrowed.** Low light and motion blur are already benchmarked for navigation (RobustNav, NavTrust), so they
  become a secondary check using those benchmarks' corruption definitions. Mirror/glass fraction and texture energy,
  measured per frame, are C's primary axes (persisting open problems in the depth, mirror and ObjNav maps; nobody
  covers them for camera-only navigation).
- **B is framed against 2026 work** (Slow Brain/Fast Planner, AsyncVLA, TIC-VLA): reported per stratum, with
  latency-honest evaluation as a methodological contribution.
- The v2 mirror guard has no camera-only counterpart (GlassGuard and LiDAR mirror detection are LiDAR-based).
- Phase 1 below is done.

## Where the project stands

- Engineering is ahead of the research. Milestones 1–7 are done; 10–12 (Laya, map-free object search, v2/v2.1 mirror
  guard, Isaac replay) go well beyond the spec.
- Spec deliverables: **A ⬜** (the spec calls the depth-error curve "the spine of the project"), **C ⬜** (needs
  milestone #8), **B 🔵** (three object-search sets; the point-goal VLM-vs-heuristic ablation is not run).
- The spec's **depth-sensor baseline** (same stack on simulator GT depth) does not exist yet. It is cheap:
  `sim/timed_episode.run_timed_episode` already takes a pluggable `depth_estimator` and already reads `obs["depth"]`.
- Statistics are thin: 15–16 episodes per set, large run-to-run variance (same episode: sofa at 84 s vs 250 s),
  ~3 h per set for two conditions on one 8 GB GPU. Only 4 homes have semantics; only 00808 has usable mirrors.
- Known simulator limits that shape C: HM3D textures have the scan's lighting baked in (no physical light sweep in
  Habitat), and Habitat mirrors are baked photos of the reflection, not physical mirrors.

## Phase 0 — Housekeeping (½ day)

- Commit the pending Go2 ground-friction fix (`tools/isaac/render_replay.py`).
- Fill the team names / owners in `PROGRESS.md` (needs the user).
- Freeze the object-search results as they are (three sets): they go into the write-up as the secondary ablation,
  pooled with confidence intervals. No more tuning on them.

**Done when:** clean `git status`, PROGRESS.md points here.

## Phase 1 — Literature map with Athena ✅ (2026-11-02; `docs/lit_review/`, Athena feedback in `/ml/Athena/docs/feedback/`)

Goal: a related-work map that says how the field evolved, where its open problems are, and exactly which gap each
FORESIGHT deliverable fills. Athena runs on the same 8 GB GPU (llama-server ~7 GB), so map builds run when no
FORESIGHT job is on the GPU, or against an OpenAI-compatible API / a rented GPU.

1. **Start Athena** (`./run_native.sh`, or Docker + host `llama-server`), create the admin account.
2. **Build five maps** (`scripts/build_evolution_topic.py <slug>` after seeding the topics):
   1. VLM / foundation-model navigation: zero-shot ObjectNav and PointNav, VLN, mapless monocular VLM navigation.
   2. Monocular metric depth estimation (zero-shot, indoor).
   3. Transparent and reflective surfaces in robot perception (glass/mirror detection, depth on glass, mirror-aware
      mapping).
   4. Robustness and perception-error propagation in embodied navigation (sensor noise, visual corruptions,
      sim benchmarks of failure).
   5. Fast–slow / dual-system architectures and latency in VLM-driven control.
3. **From each map:** lineage timeline (how the field evolved), milestones and reading path, open-problem themes
   (persisting / new / fix claimed / quiet).
4. **Idea checks:** the FORESIGHT abstract, then each contribution separately: (A) depth-error → navigation-error
   curve, (B) "does a compact VLM earn its cost over a depth heuristic", (C) difficulty-stratified failure profile,
   (v2) fast/slow mirror guard. Record closest prior work and what Athena says is new.
5. **Recall check, used as Athena feedback:** a hand-made seed list of papers the field knows well (the spec's
   VLM-Nav, AutoFly, MonoDream, NaVid; plus e.g. CoW, ESC, VLFM, SG-Nav, Uni-NaVid; Depth Anything V1/V2, Metric3D v2,
   UniDepth, Depth Pro, ZoeDepth; RobustNav; glass/mirror detection and ClearGrasp-style depth on transparent
   objects), each verified by a web search. Measure which ones each map surfaces, how it ranks them, and whether
   the gap themes match what the papers actually state.

**Outputs:** `docs/lit_review/` with one note per map (evolution, milestones, gaps), `positioning.md` (FORESIGHT's
three contributions against the gaps, the draft of the paper's related-work section), and
`athena_feedback.md`: recall per map, wrong or missing gaps, idea-check quality, failures, build times, UX friction,
concrete feature requests.

**Done when:** related-work draft exists and every claimed gap cites the papers that state it.

## Phase 2 — Point-goal evaluation infrastructure (the base for A, B and C)

1. **Depth sources** (`perception/depth/`): a `DepthSource` protocol that sees the sim observation, not only RGB:
   - `GroundTruthDepth`: the spec's depth-sensor baseline;
   - `InjectedErrorDepth(GT, …)`: global scale, additive bias, depth-proportional Gaussian noise, holes, and
     structured error on masked pixels (reflective: replace with the "see-through" depth; textureless: smooth or
     swap in Depth Anything's output there); plus NavTrust's four sensor corruptions (Gaussian noise, missing
     data, multipath, quantization) so our results line up with theirs;
   - `ZeroShotDepthEstimator` (unchanged, wrapped).
   Option for latency matching: GT/injected sources can wait the measured Depth Anything time, so a comparison
   changes only the error, not the loop rate.
2. **Point-goal conditions** sharing one loop (`run_timed_episode`, latency-honest):
   `heuristic` (milestone #7 potential field), `map` (`MapNavigator`, map-free occupancy mapping, depth only),
   `foresight` (`MapNavigator` + `FastSlowMirrorGuard`). The guard uses `scans`, `last_path` and
   `reflection_segments`, which live on `ObjectSearchNavigator`: lift them into `MapNavigator` so point-goal gets
   the same guard. Goal given as GPS/compass bearing, as in the spec.
3. **Episode sets**, seeded and committed before any run (as for object search): reuse
   `tools/generate_pointnav_episodes.py`; all 10 minival scenes; geodesic-distance bins; a separate dev set for any
   tuning.
4. **Batch runner + scorer** `tools/run_pointnav_eval.py`: `--condition`, `--depth-source`, `--episodes`,
   `--shard i/n` (for parallel workers); metrics per episode: collision rate, success, SPL, progress to goal, time,
   achieved Hz, and per-frame difficulty (reflective-pixel fraction from semantics, texture energy) so A and C
   read from the same runs. Stats: bootstrap 95% CIs, paired tests on shared episodes (McNemar for success, Wilcoxon for SPL).
5. **Throughput:** measure episodes/hour locally. Non-VLM conditions need only the renderer + depth (~1.5 GB),
   so 3–4 can run in parallel on the 4060. Then a container image (habitat-sim 0.3.3 + models) for RunPod. Price
   is quoted before any pod is created.
6. **Pilot** (~30 episodes × 3 conditions × GT/DA depth): check the plumbing, then set N from the observed variance
   (power analysis), not a guess.

**Decisions to settle in the pilot:** success radius (standard PointNav 0.2 m with stop vs. 0.5–1.0 m for continuous
control), time budget per episode, N per cell.

**Done when:** one command reproduces a scored, CI-annotated table for any (condition × depth source × episode set).

## Phase 3 — Deliverable A: depth-error propagation curve

- **Hypothesis H-A** (from the lit review): uniform errors (scale, bias, noise) degrade navigation gracefully;
  structured errors on reflective/textureless pixels cause a disproportionate share of collisions. Curves are
  plotted per error type against RMSE, with the share of collisions coming from structured error.
- **Sweeps on GT depth** (each axis separately, then a few combinations): scale 0.5–2.5× (Depth Anything V2 was
  measured at 1.66–2.5× on HM3D), bias ±0.5 m, noise 0–20%, holes, structured reflective/textureless error.
  Conditions: `heuristic` and `map` (cheap, no VLM) at every level; `foresight` at a few levels around the
  reflective-error axis, to test the spec's hypothesis that the VLM brings a semantic signal *independent* of depth.
- **Measured on the frames the agent actually saw:** depth RMSE, AbsRel, δ1 per run → plot collision / success /
  SPL / progress against depth RMSE, with CIs.
- **Place the real estimator on the curve:** Depth Anything V2 with and without the `FloorScale` correction, so the
  curve says how much of the real system's failure its depth error explains.
- **External calibration (optional):** Depth Anything V2 on NYUv2 (download via `finsight`) to show the
  injected ranges match real-world error.

**Done when:** the curve figure + table, with the real estimator marked on it.

## Phase 3 result (2026-11-04, `docs/results/depth_error_A.md`)

Sweep complete (5,600 runs). Error *type* predicts navigation error, magnitude does not. Depth Anything V2's error is
mostly global scale, which the floor calibration removes (map navigator 79% vs GT 83% on set A, n.s.; 71% vs 89% on
set S, p = 0.006). Zero-mean noise collapses the map navigator (phantom obstacles from per-column nearest points).
**Injected reflective error does not move aggregate success on random episodes**, even mirror-exposed ones: reflective
failure is a tail event, so C's mirror/glass stratum must use targeted encounters (below).

## Phase 4 — Milestone #8 + Deliverable C: difficulty-stratified benchmark

**Change after Phase 3:** random point-goal episodes dilute reflective failures to nothing. The mirror/glass stratum is
built from *encounter* episodes instead: start 2–4 m from a mirror or glass pane, facing it (± 30°), goal behind or
beside it, scouted from the HM3D-Semantics annotations of the 4 annotated homes (the 00808 hallway mirror is one);
reported per encounter type, with GT / DA / masked-swap depth, and the map navigator vs FORESIGHT.

Each axis gets a *measured* difficulty level, as the spec asks. After the lit review, **mirror/glass and textureless
are the primary axes**. Low light and motion blur are already benchmarked (RobustNav, NavTrust), so they become a
secondary check that uses those benchmarks' corruption definitions for comparability.

| Axis | How it's measured | How it's made in Habitat | Caveat to report |
|---|---|---|---|
| Glass / mirror | fraction of pixels labelled mirror/glass/window along the run | natural scenes, stratified; more homes from **HM3D-Semantics** (token), scouted with the existing `glass_scout` scripts | Habitat mirrors are baked photos; optionally a physical-mirror subset in the existing Isaac pipeline |
| Textureless | texture energy / gradient density per frame, binned | natural stratification; controlled version on semantic homes (flatten wall pixels) | flattening is an image edit, not geometry |
| Low light | image mean luminance / exposure level (pseudo-lux) | photometric sensor model: exposure gain, gamma, Poisson–Gaussian noise | baked lighting: no physical lux sweep, say so |
| Motion blur | exposure × angular velocity | average sub-frames rendered along the motion during the exposure (real geometric blur) | — |

- Conditions per stratum: GT-depth stack (upper bound), `heuristic`, `map`, `foresight`, all on Depth Anything
  depth. Output: per-stratum collision / success / SPL / progress with CIs: the failure profile.
- HM3D-Semantics: download on `finsight` (local link ~40 KB/s), copy back only what's scouted as useful, and
  symlink into `data/processed/` (CLAUDE.md data rules).

**Done when:** one failure-profile figure across the four axes + per-stratum tables.

## Phase 5 — Deliverable B: does the VLM earn its cost?

Mostly falls out of Phases 3–4 (same episodes, paired). Framed against Slow Brain/Fast Planner, AsyncVLA and
TIC-VLA (architecture-side latency work), with latency-honest *evaluation* as our methodological contribution:
- point-goal `foresight` vs `map` vs `heuristic`, overall and per stratum;
- the operating point table the spec asks for: model sizes, device, achieved Hz of each loop, latency per look;
- object-search results from the three sets, pooled, as the secondary ablation.

A negative result (the VLM doesn't pay off outside the mirror stratum) is a reportable finding, as the spec says.

## Phase 6 — Write-up (W1, W2)

Results analysis, figures, related work from Phase 1, limitations (baked lighting/mirrors, one simulator, minival
scale, sim time ignoring GPU contention). Venue/format still open.

## Parked (revisit after Phase 5)

- v2.2 fixes (no re-check of a mirror walled from another angle; stop only for candidates within ~1.5 m) and a
  fourth object-search set.
- PPO PointNav retrain (milestone #9, ~5% success). Could be a trained-baseline row in B if retrained on HM3D
  train scenes once the token is in.
- Learned "when to check / where to explore / when to stop" policy.
- Isaac Sim polish and new narrated videos (beyond committing the friction fix).

## Risks

| Risk | Mitigation |
|---|---|
| GPU contention: Athena (~7 GB) vs evaluation jobs on one 8 GB card | Athena when the GPU is idle, against an API, or on a rented GPU |
| Cloud cost creep | per-batch price quote before any pod; spot-check throughput locally first |
| Not enough mirror/glass episodes even with more homes | scout first; report stratum sizes honestly; physical-mirror subset in Isaac as a stretch |
| Low light / mirrors not physical in Habitat | named as limitations; image-space models are documented and parameterised |
| Tuning on evaluation episodes | dev set separate; every evaluation set committed before its first run (existing practice) |

## Order

```
Phase 0 ─┬─ Phase 1 (Athena lit map) ─────────────────────────────┐
         └─ Phase 2 (infra + pilot) ─ Phase 3 (A) ─ Phase 4 (#8, C) ─ Phase 5 (B) ─ Phase 6 (write-up)
                         └─ HM3D-Semantics download on finsight (parallel, needed by Phase 4)
```

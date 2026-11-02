# Literature review: the field map (2026-11-02)

Built with Athena (`/ml/Athena`: research maps, milestones, gap themes from papers' own limitation sections,
idea checks, Ask a map) and cross-checked by web search. Positioning and what it changes in the plan:
[`positioning.md`](positioning.md). Athena's milestone and reading-path exports: [`athena_exports/`](athena_exports/).
Feedback on Athena itself: `/ml/Athena/docs/feedback/2026-11-02-foresight-lit-review.md`.

## Method

- Six maps, each from a search proposal whose anchors and keywords were then curated (as a user would in the UI),
  then full text (Tier 2: milestones, main path, strongest citers; 63–87 PDFs per map, GROBID) and gap themes.
- Coverage checked against a hand-made list of 57 papers each field knows well (all web-verified): 49 present.
  Missing are mostly very recent UAV systems (AutoFly, VLM-Nav) and dual-system VLAs (RT-2, Hi Robot, RoboDual,
  HiRT, GR00T N1).
- Two maps drifted (robustness → generic adversarial ML; dual-system → VLA post-training). Their gap themes are used
  only where a theme's example sentences are about navigation or VLAs.
- Six idea checks (the FORESIGHT abstract, deliverables A/B/C, the v2 mirror guard, latency-honest evaluation) and
  two Ask-a-map questions.

| Map | Papers | Milestones | Gap themes | Usable? |
|---|---|---|---|---|
| Vision-and-Language Navigation | 941 | 17 | 35 | yes |
| Zero-Shot Object Goal Navigation | 524 | 13 | 28 | yes |
| Monocular Metric Depth Estimation | 641 | 18 | 41 | yes |
| Mirror, Glass and Transparent Surface Perception | 469 | 24 | 15 | yes |
| Robustness of Embodied Navigation | 1191 | 22 | 61 | partly (drifted) |
| Dual-System Vision-Language-Action Models | 356 | 9 | 46 | partly (narrow) |

## How the fields evolved

**Vision-and-language navigation** (941 papers; 4 in 2017 → 191 in 2025). Four eras:
1. 2017–2019, the benchmark and seq2seq era: R2R (Anderson et al. 2018), Speaker-Follower, RCM. "On Evaluation of
   Embodied Navigation Agents" (2018) fixes SPL as the metric.
2. 2020–2022, pretraining and memory: RxR, VLN-BERT, PREVALENT, HAMT, DUET, and continuous environments (VLN-CE 2020).
3. 2023–2024, LLM/VLM planners: NavGPT, MapGPT, NavCoT, then video VLMs that plan the next step (NaVid, Uni-NaVid,
   NaVILA).
4. 2025–2026, streaming and dual-system VLN (StreamVLN's SlowFast context; "Ground Slow, Move Fast"), monocular-only
   agents (MonoDream, monoVLN, HAVEN), UAV VLN.

The milestone trail moves from panoramic RGB-D in discrete graphs towards one forward RGB camera in continuous
space: FORESIGHT's setting.

**Zero-shot object-goal navigation** (524 papers). It starts from learned semantic exploration (SemExp 2020, PONI
2022, "ObjectNav Revisited"). Zero-shot arrived in 2022 (ZSON with CLIP goal embeddings; CoW with CLIP + frontier
exploration). 2023 brought frontier maps valued by a VLM/LLM (VLFM, ESC, L3MVN, LGX). 2024 added scene-graph
prompting (SG-Nav, OpenFMNav, VoroNav), and 2025 adaptive exploration (ApexNav). FORESIGHT's object search
(frontiers + VLM grounding + object memory) sits on the VLFM/ESC line, but without a depth sensor.

**Monocular metric depth estimation** (641 papers). It runs from self-supervised stereo/video (Monodepth 2016,
Monodepth2 2018) and supervised regression (DORN, BTS, AdaBins/BinsFormer, NeWCRFs) to zero-shot transfer (MiDaS
2019, DPT), then zero-shot *metric* depth (ZoeDepth, Metric3D 2023). 2024 is the foundation-model year: Depth
Anything V1/V2, UniDepth, Metric3D v2, Depth Pro, plus Marigold's diffusion route. 2025 adds video consistency and
camera-agnostic models (UniDepthV2, Video Depth Anything, Depth Any Camera). Output from 2025 on is 2× any earlier year.

**Mirror, glass and transparent surfaces** (469 papers). Physics-based reconstruction (2005–2016: "Finding Glass",
light-path triangulation, tomography) was followed by learned detection in 2019–2021: mirror detection (MirrorNet
2019, PMD 2020), glass detection (GDNet 2020, reflection priors 2021) and transparent segmentation (Trans10K).
Depth completion for grasping ran alongside (ClearGrasp 2019, local implicit functions 2021, TODE-Trans,
TCRNet). Since 2022 the work has gone to video and multi-view mirror/glass detection and SAM-based detectors. The
whole field is about manipulation (depth completion for grasping) or detection benchmarks. **None of its milestones
is about navigating past a mirror**, and the robot-navigation work it touches is LiDAR-based (laser mirror detection;
GlassGuard 2026, LiDAR + RGB).

**Robustness of embodied navigation** (drifted map; navigation-relevant thread only). Habitat (2019) and DD-PPO set
up the evaluation, "Sim2Real Predictivity" asked whether sim results transfer, and RobustNav (2021) introduced
visual and dynamics corruptions for PointNav/ObjectNav. "Is Mapping Necessary for Realistic PointGoal Navigation?"
(2022) followed, then NavTrust (2026), which adds depth corruptions and VLN/OGN agents. In between, the map is
mostly adversarial-example literature.

**Dual-system VLAs** (narrow map). RT-H (2024) introduced language action hierarchies. 2025 brought dual-system
designs with asynchronous frequencies (Fast-in-Slow, OpenHelix, "Ground Slow, Move Fast", NaVILA's VLA + locomotion
policy), and 2026 latency-aware and edge variants (AsyncVLA, TIC-VLA). Its open problems are deployment on
resource-limited hardware and real-time control.

## Open problems that bear on FORESIGHT

From Athena's gap themes (status: persisting / new / quiet), with a representative stated limitation.

| Theme (map) | Status | What papers say |
|---|---|---|
| Limited spatial reasoning of VLMs (ZS-ObjNav, 65 statements; VLN "spatial · vlms", 17) | persisting | "given VLMs' limited 3D spatial understanding, involving VLMs in fine-grained, in-room exploration fails to leverage their semantic…" |
| Sensor noise and control errors (ZS-ObjNav, 5); depth-sensor corruptions (robustness, 9) | persisting | "Current benchmarks also neglect depth-sensor corruptions and lack a unified framework for systematically evaluating robustness" (NavTrust) |
| Detection of small reflective objects (ZS-ObjNav, 7) | persisting | "SAM3 struggles with transparent, reflective, and small objects" |
| Limited real-time reasoning; LLM call costs (ZS-ObjNav, 6 + 3) | persisting | "frequent API calls lead to high costs and slow execution"; "repeated foundation-model calls limit efficiency" |
| Monocular / limited field of view (VLN, 5) | persisting | monocular agents lag panoramic ones (MonoDream, monoVLN) |
| Transparent and reflective surfaces in depth (depth map, 9; mirror map, 69) | persisting | "accurately reconstructing transparent objects and reflective surfaces remains challenging"; UniDepthV2 "interprets mirrors as cavities" |
| Weak texture and flat surfaces (depth map, 5) | persisting | "these models still struggle on weakly textured distant regions and on geometrically flat surfaces" |
| Hard regions behind good averages (depth map) | persisting | "Beyond average accuracy, persistent failure cases cluster in several hard regions" |
| Benchmarks report clean performance (robustness: MDE, 7) | persisting | depth-model "vulnerabilities are largely overlooked by existing benchmarks, which typically report performance under clean…" |
| Real-time control on limited platforms (dual-system, 7 + 8) | new / persisting | "running such a model on … Jetson AGX Orin can result in inference latency…"; "limited empirical characterization of how deployed systems behave on specific hardware" |
| Sim-to-real (VLN 15, dual-system 16) | persisting | real-robot validation left as future work (out of FORESIGHT's scope too) |

Ask a map, on the ZS-ObjNav map: *no paper in the map reports how much success drops when depth is estimated from
RGB instead of read from a sensor* (ZSON compares RGB-only with CoW's RGB-D only indirectly; PixNav and MVP-Nav
are RGB-only but report no such comparison). On the depth map, the only robotics measurement it found is a static
one ("Assessing Depth Anything V2 as a LiDAR alternative", 2025: 0.32–0.66 m mean error indoors, "not suitable for
… obstacle avoidance"); nothing measures what those errors do in closed-loop navigation.

## Closest prior work (Athena + web)

| Work | What it does | Overlap with FORESIGHT | What it doesn't do |
|---|---|---|---|
| VLM-Nav (UAV) | Depth Anything V2 + VLM, mapless obstacle avoidance | the same perception idea | cloud VLM, no controlled depth-error study, no stratification |
| AutoFly (ICLR 2026) | end-to-end UAV VLA with a pseudo-depth encoder | monocular pseudo-depth | trained, outdoor, no failure analysis |
| MonoDream (AAAI 2026), monoVLN, HAVEN | monocular VLN agents | monocular RGB only | trained VLN; no depth-error / reflective analysis |
| RANGER (ICRA 2026), MVP-Nav, PixNav | monocular zero-shot semantic navigation | RGB-only ObjectNav | no GT-depth comparison, no mirrors |
| LG-VLN (2026) | monocular zero-shot VLN-CE with estimated depth | estimated depth vs GT | reports only a small drop and argues navigation is insensitive to uniform scale error; no controlled curve, no structured error |
| APF + UniDepth path planning (2025) | monocular metric depth → occupancy → potential field | ≈ FORESIGHT's depth-only baseline | no VLM, no failure study |
| RobustNav (ICCV 2021) | visual + dynamics corruptions, PointNav/ObjectNav | low light, motion blur axes | trained RGB(-D) agents, sensor depth, no reflective axis |
| NavTrust (IROS 2026) | RGB corruptions (motion blur, low light ± noise, spatter, flare, defocus, …) + depth corruptions (Gaussian, missing, multipath, quantization), VLN + OGN incl. VLFM | corruption benchmark | fixed severities, sensor-depth corruptions only, no performance-vs-depth-error curve, glass only as "missing data" |
| GlassGuard (2026), LiDAR mirror detection (2024) | glass/mirror mapping for navigation | reflective surfaces on the map | LiDAR-based, glass only / laser intensity |
| Slow Brain, Fast Planner (2026) | async VLM picks among planner trajectories, VLM-vs-planner ablation | B's ablation + latency | sidewalk, trained planner, remote VLM |
| AsyncVLA, TIC-VLA (2026) | asynchronous / delay-aware VLAs for navigation | fast/slow split, latency | trained VLAs; latency handled in the model, not in the evaluation |
| "Can Vision Foundation Models Navigate?" (2026) | real-world eval of GNM/ViNT/NoMaD etc. with perturbations | failure-focused evaluation | trained visual navigation models, no depth study |
| Depth on mirrors/glass (2307.15052; GIFT 2026) | DA-V2 drops sharply on Booster / Mirror3D | quantifies reflective depth error | perception only, no navigation |

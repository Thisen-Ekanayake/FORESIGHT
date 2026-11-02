# Positioning: what the literature says about each FORESIGHT contribution

Companion to [`README.md`](README.md) (the field map). Each contribution is checked against the closest work and the
stated open problems, then the consequence for `PLAN.md`.

## A. Depth-error propagation (the spine): **the clearest gap. Keep it, and sharpen it.**

- Nobody plots closed-loop navigation error against monocular depth error. NavTrust corrupts *sensor* depth
  (Gaussian noise, missing data, multipath, quantization) at fixed severities and reports per corruption; RobustNav
  has no depth axis. The depth-estimation literature measures the error statically (DA-V2 vs LiDAR: 0.32–0.66 m)
  and stops there. Athena's Ask a map found no paper in the ObjectNav map reporting the success cost of estimated
  vs sensor depth.
- There is a claim to test: LG-VLN reports a small drop when GT depth is swapped for estimated depth and argues
  navigation is insensitive to uniform scale error. Our own data say the tail is not benign: Depth Anything V2 reads
  1.9–2.5× too far on HM3D homes, and a mirror is read as open space 4.5–7.9 m deep at 2.5 m true distance.
  Depth papers themselves say failures "cluster in hard regions" behind good averages.
- **Sharpened hypothesis (H-A):** uniform scale and noise errors degrade navigation gracefully (so averages look
  fine, as LG-VLN found), while *structured* errors on reflective and textureless regions cause a
  disproportionate share of collisions. The curve should separate the two: error *magnitude* (RMSE) on the x-axis,
  one line per error *type*.
- **Plan change:** include NavTrust's four depth corruptions as extra error types (comparability), add the
  structured reflective / textureless types, and report per-type curves plus the share of collisions from
  structured error. Place the real Depth Anything V2 (raw and FloorScale-corrected) on the curve.

## B. Does a compact VLM earn its cost?: **still open, but it must be framed against 2026 work.**

- Slow Brain, Fast Planner (2026) runs a VLM-vs-planner ablation (sidewalk, remote VLM, trained planner), and
  AsyncVLA / TIC-VLA handle VLM latency inside trained VLAs. Athena's ZS-ObjNav map lists "limited real-time
  reasoning" and "LLM call costs" as persisting problems; the dual-system map lists real-time control on limited
  hardware as new.
- FORESIGHT's distinct angle: a zero-shot, *on-board* 2B VLM sharing one 8 GB GPU with depth and rendering, evaluated
  under **latency-honest** simulation (sim time advances by measured compute), against a depth-only planner on
  paired episodes, with the hypothesis that the VLM's value is a semantic signal *independent of degenerate depth*
  (the mirror case). The object-search results already show the shape: fewer mirror crashes, paid for in time.
- **Plan change:** report latency-honest evaluation as a methodological contribution (Athena's idea check and the
  web found no evaluation-side treatment). Report B per stratum, because the VLM is expected to pay only where depth
  is degenerate. Cite Slow Brain / AsyncVLA / TIC-VLA as the architecture-side work.

## C. Difficulty-stratified benchmark: **partly covered. Narrow it to what is monocular-specific.**

- Low light and motion blur are already benchmarked for navigation (RobustNav: < 10% SPL drop for PointNav-RGB;
  NavTrust: low light + noise costs RGB-only agents 29–31% SR). Re-doing them is replication, not a contribution.
- Reflective and textureless surfaces are the monocular-specific failures. The depth map (transparent/reflective,
  weak texture: persisting), the mirror map (69 statements on transparent depth, all manipulation or detection) and
  the ObjNav map ("detection of small reflective objects": persisting) all leave them open for navigation, and no
  benchmark *measures* reflective-pixel fraction or texture energy per episode. GlassGuard and LiDAR mirror work
  cover LiDAR, not a single camera.
- **Plan change:** C's main axes are **mirror/glass fraction** and **texture energy**, measured per frame along the
  run. Low light and motion blur stay as a smaller secondary axis using RobustNav/NavTrust's corruption definitions,
  so our numbers are comparable to theirs (a check, not a headline). This also reduces work, because the photometric
  low-light model and the sub-frame blur renderer drop down the priority list.

## v2 mirror guard (method behind the mirror stratum): **novel in its sensor setting.**

- Mirror detection exists (MirrorNet, PMD, SAMirror); mirror/glass mapping for navigation exists with LiDAR
  (GlassGuard; laser-intensity mirror detection). A camera-only robot that uses a VLM's yes/no check plus a depth
  "phantom room" gap to wall off a mirror on a self-built map, with a slow description verified only when the mirror
  is in the way, has no direct counterpart in either map or on the web.
- **Plan change:** none structurally. It is FORESIGHT's method for the mirror stratum in C and the "semantic signal"
  in B, and should be evaluated there on point-goal episodes, not only object search.

## Positioning statement (draft for the paper)

> Recent systems show that a single RGB camera plus foundation models can drive zero-shot navigation (VLM-Nav,
> RANGER, MonoDream, LG-VLN), and robustness benchmarks show that navigation agents degrade under visual and sensor
> corruptions (RobustNav, NavTrust). What is missing is a causal account of *how monocular depth error becomes
> navigation error*, and of whether it is the error's size or its structure that matters. FORESIGHT measures this
> directly in simulation by injecting controlled depth error against ground truth. It then tests whether a compact
> on-board VLM, evaluated with its real latency, buys back the failures depth cannot see, which are concentrated on
> mirrors, glass and textureless walls.

## Effect on `PLAN.md`

| Phase | Before | After |
|---|---|---|
| 2 (infra) | GT / injected / DA depth sources | + NavTrust's four depth corruptions as error types; per-frame reflective fraction and texture energy logged by the runner (needed by A and C) |
| 3 (A) | sweeps vs RMSE | per-type curves (uniform vs structured), H-A tested explicitly, share of collisions from structured error |
| 4 (C) | four equal axes | mirror/glass + textureless primary; low light + blur secondary, using RobustNav/NavTrust definitions |
| 5 (B) | VLM vs heuristic | per stratum; latency-honest evaluation reported as a method; cite Slow Brain / AsyncVLA / TIC-VLA |
| 6 (write-up) | related work TBD | drafted here |

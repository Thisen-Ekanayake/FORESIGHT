# `foresight.planning.fast_slow` — FORESIGHT v2

Fast/slow reflection handling for [object search](object_search.md). v1 (`laya_fusion.ForesightNavigator`) put a
~2 s Qwen3-VL scene description on every look so Laya could check each one for mirrors: on the first evaluation set
it crashed less than the baseline (1 vs 5 runs, every baseline crash at the 00808 hallway mirror) but its looks took
2.6 s against 0.34 s, and it ran out of time 7 times against 2. v2 keeps the same two-signal rule for turning a
mirror into a wall, but pays for the slow signal only when a mirror is in the robot's way.

| Stage | Runs | Cost (RTX 4060) | What it gives |
|---|---|---|---|
| Fast check | every look | one forward pass (`option_probs`, "Is there a mirror in this image?"), plus grounding only if P(yes) > 0.5 | a mirror box |
| Candidate wall | every look with a mirror box | none (depth scan of that frame) | `ObjectSearchNavigator.reflection_segments`: the boxed region grown sideways while it reads as deep; a candidate only if it reads ≥ 1.3× deeper than the real surfaces framing it (a phantom room) |
| In the way? | every tick, per candidate (kept 15 s) | none | the planned route within 5 m, or a 2.5 m ray along the heading, crosses the candidate |
| Slow check | only then | stop forward motion, face the view, the next look is the reflection-aware description alone (~2.5 s); Laya reads it (~25 ms) | P(mirror) > 0.6 → the candidate becomes a wall; else dismissed and not raised again; 14 s timeout |

The rest is shared with the baseline and v1: explorer, object grounding and memory, close-in arrival, safety layer.
Objects seen through a wall are ignored (`ObjectSearchNavigator.through_wall`).

## Why these choices

- The yes/no question is far more precise than the 2B model's boxes (presence probe: ≈1 on 11/12 frames with the
  object, ≈0 on 19/20 without), so it gates the mirror grounding as it gates object grounding.
- "In the way" uses the candidate's geometry, not the view's angle: v1's angular in-path test (route points within
  ±25° of the flagged view) judged a P = 0.94 flag off-route while the robot was turning, and the robot drove into
  that mirror 1.2 s later.
- The note look skips the object queries: it exists only to be read by Laya, and the robot is standing still.

## Results

See [object_search.md](object_search.md#results-fresh-set-2026-11-01).

## v2.1 (2026-11-01, commit `ffa11ce`)

v2 (commit `b38b4ed`) was evaluated on the second set. Its mirror crash there (`s2-00808-tv-3`) and its crash on the
demo episode were into the same 00808 closet mirror (sliding mirror doors). Replaying those looks offline (re-rendered
frames, the same depth model and `reflection_segments`) showed:

1. **A real candidate dropped as "known".** v2 treated a candidate as already handled if its midpoint was within
   0.8 m of any wall's midpoint; an earlier wall drawn at the wrong angle nearby hid it. v2.1: known only if the
   candidate lies *along* an existing wall or a dismissed candidate (both ends and its middle within 0.4 m of it).
2. **No candidate at all.** Qwen's "mirror" boxes landed on objects inside the reflection (a TV, two pictures), which
   read about as near as their frame, so there was no phantom-room gap. v2.1: a look with P(mirror) ≥ 0.9 also turns
   each mirror box into a *wedge* (its bearing span from that pose, out to 3 m) when it has no gap geometry; if the
   route or heading enters the wedge, the slow check runs. If Laya confirms, the wedge is walled across its span at the
   nearest depth read inside it (0.6–2.0 m).

Post-hoc check on those two episodes (not evidence by itself): no crash on either; tv found at 92 s; on the demo,
the sofa found at 122 s, the bed rejected by the close-up check, timeout at 300 s. The test is the third set,
drawn and committed (`8d638e5`) before any run on it.

## v2.2 (2026-11-10): against over-checking

On point-goal (dev set `A_3001`, 30 episodes, Depth Anything depth) v2.1 succeeded 33% against 63% for the same
navigator without it (p = 0.006): 4–18 slow checks and 1–6 confirmed walls per episode, even in homes with no annotated
mirror; object-search set 3 showed the same over-checking in 00808. v2.2 adds three options (off by default, so object
search keeps its evaluated v2.1 behaviour; on in `tools/run_pointnav_eval.py --guard v22`, the default there):

| Option | v2.2 value | What it changes |
|---|---|---|
| `near_m` | 1.5 m | a candidate is in the way only where the route or heading meets it within 1.5 m (v2.1: 5 m of route, 2.5 m of heading, 3 m wedges) |
| `walled_m` | 0.8 m | a candidate within 0.8 m of an existing wall is already handled, whatever its angle |
| `gap_s` | 5 s | at least 5 s between the end of one slow check and the start of the next |

Dev results (Depth Anything depth; paired episodes, scan holes voided):

| Dev set | Map navigator only | v2.1 | v2.2 |
|---|---|---|---|
| `A_3001` (26 paired): success / crash | 58% / 4% | 38% / 0% | **58% / 0%** |
| slow checks / confirmed walls per episode | – | 9.3 / 1.8 | 5.2 / 1.0 |
| v2.2 vs map: −3/+3, p = 1; v2.2 vs v2.1: −0/+5, p = 0.06 | | | |
| `S_3002` (16 paired, annotated homes): success / crash | 44% / 6% | – | 38% / 0% (−1/+0) |

v2.2 no longer costs the navigator anything on dev, and it removes its crashes. **Open issue: precision.** On the
annotated dev homes only 3 of 19 confirmed walls lie within 0.6 m of a real mirror/glass object (`walls_on_reflective`),
and the slow check confirms 45% of what it checks (72 of 159). The next dev step, before the evaluation-set run (B), is a
stricter confirmation (Laya threshold 0.6 → 0.8, or requiring both a mirror box and a phantom-room gap).

## v2.3 (2026-11-11): wedge candidates only

v2.2 now logs every confirmed wall (`wall_log`: time, Laya P, mirror-box P, candidate kind; on annotated homes also the
distance to the nearest mirror/glass object, `on_reflective` if < 0.6 m). Diagnostic: `tools/guard_precision.py`.
On the annotated dev episodes (`S_3002` first 4 per stratum + 12 dev encounters `E_9001`), v2.2 confirmed 45 walls, 20
(44%) on a real mirror/glass object. Neither confidence separates them: keeping walls with Laya P ≥ 0.8 or box P ≥ 0.97
leaves precision at 41–60% and loses real walls. What separates them is the candidate kind: walls from **depth gaps**
(phantom-room segments, `reflection_segments`) were 6/26 on real mirrors, walls from **wedges** (the mirror box's
bearing) 14/19. v2.3 = v2.2 + `gap_candidates=False` (`run_pointnav_eval.py --guard v23`).

Dev results (Depth Anything depth; scan holes voided):

| Dev set | Map navigator only | v2.2 | v2.3 |
|---|---|---|---|
| Confirmed walls on a real mirror/glass (annotated homes) | – | 20/45 (44%) | **21/33 (64%)** |
| `A_3001` (28 paired): success / crash / SPL | 61% / 4% / 0.50 | 54% / 0% / 0.49 | 57% / 0% / 0.53 |
| slow checks / confirmed walls per episode | – | 5.8 / 1.25 | 4.2 / 0.82 |
| v2.3 vs map: −3/+2, p = 1; v2.3 vs v2.2: −2/+3, p = 1 | | | |
| `S_3002` exposed (16): success / crash | 44% / 6% | 38% / 0% | 44% / 0% |
| Dev encounters `E_9001`: success / crash | 40% / 40% (10) | 33% / 11% (9) | **50% / 10% (10)** |
| slow checks per encounter | – | 8.4 | 6.6 |

v2.3 keeps the real walls (21 vs 20) and drops most false ones (12 vs 25), costs nothing on plain point-goal, and on
encounters turns the map navigator's 4 crashes in 10 into 1 while succeeding more often. Left over: 3 of 10 encounters
still time out after 10–12 slow checks (a real mirror walled from several angles, the route keeps re-approaching it), and
one crash at the 00808 closet mirror wall although 3 walls were confirmed on it. The dev sets are small (10–28
episodes), so these are directions, not effects; v2.3 is the configuration for the evaluation run (B).

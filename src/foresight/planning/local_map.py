"""Map-free local mapping + planning from zero-shot monocular depth.

No prior map: each robot builds a 2D occupancy grid from its own depth estimates as it drives (pose from
odometry, the same GPS+compass a PointGoal agent already gets for its goal bearing), and plans a route
to the goal on that grid every tick. Unmapped cells are traversable but cost more, so the robot explores
when its known free space doesn't reach the goal.

- `FloorScale`: zero-shot metric depth is off by a scene-dependent factor (~2.3x too far in HM3D 00808).
  The camera's height above the floor is known, so floor pixels have a known true depth; the running
  median of (geometric floor depth / predicted depth) over frames whose bottom band looks like flat floor
  rescales the prediction. No ground truth is used.
- `depth_scan`: per image column, the nearest point whose height above the floor is in [h_min, h_max]
  becomes a laser-scan-like range (inf if none within r_max).
- `OccupancyMap.add_wall`: a virtual obstacle segment that free-space carving can't erase — FORESIGHT
  draws one across a view Laya flags as a reflection, so a mirror's phantom hallway can't be planned through.
- `plan`: Dijkstra from the goal over the inflated grid (obstacles very expensive, not impassable), then
  steepest descent from the robot to a carrot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numba
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


class FloorScale:
    """Ratio of the floor's geometric depth to the predicted depth in the image's bottom band, kept for every
    frame where that band looks like one flat horizontal surface (same ratio on every row).

    Any flat surface passes that test, and a raised one (bed, table top) at height h reads as floor with a
    ratio 1.5 / (1.5 - h) too high, e.g. ~1.6x for a bed; next to a bed those frames can be the majority.
    Raised planes only ever bias the ratio up, so the floor is the lowest cluster of ratios that holds a
    fair share of the samples (not the median)."""

    def __init__(self, hfov_deg: float = 90.0, cam_height_m: float = 1.5, band=(0.85, 1.0), cols=(0.3, 0.7),
                 max_spread: float = 0.15, history: int = 200, cluster_width: float = 0.12, min_share: float = 0.2):
        self.hfov, self.cam_h, self.band, self.cols = np.deg2rad(hfov_deg), cam_height_m, band, cols
        self.max_spread, self.history = max_spread, history
        self.cluster_width, self.min_share = cluster_width, min_share
        self.samples: list = []
        self.value = 1.0

    def update(self, depth: np.ndarray) -> float:
        h, w = depth.shape
        f = (w / 2) / np.tan(self.hfov / 2)
        r0, r1 = int(self.band[0] * h), int(self.band[1] * h)
        c0, c1 = int(self.cols[0] * w), int(self.cols[1] * w)
        v = np.arange(r0, r1) - h / 2 + 0.5
        floor_z = self.cam_h * f / v  # true optical-axis depth of the floor at each row
        ratio = floor_z[:, None] / np.maximum(depth[r0:r1, c0:c1], 1e-3)
        per_row = np.median(ratio, axis=1)
        med = float(np.median(per_row))
        if med > 0 and (per_row.max() - per_row.min()) / med < self.max_spread:  # one flat surface
            self.samples = (self.samples + [med])[-self.history:]
            self.value = self._lowest_cluster()
        return self.value

    def _lowest_cluster(self) -> float:
        x = np.log(np.sort(self.samples))
        n = len(x)
        # count of samples within cluster_width (in log space) above each sample
        counts = np.searchsorted(x, x + np.log1p(self.cluster_width), side="right") - np.arange(n)
        need = max(1, int(np.ceil(self.min_share * n)))
        k = int(np.argmax(counts >= need)) if (counts >= need).any() else int(np.argmax(counts))
        return float(np.exp(np.median(x[k:k + counts[k]])))


def depth_scan(depth: np.ndarray, scale: float, hfov_deg: float = 90.0, cam_height_m: float = 1.5,
               stride: int = 4, h_min: float = 0.2, h_max: float = 1.7, r_max: float = 3.5):
    """-> (bearings rad, positive = right; ranges m, inf where nothing within r_max)."""
    d = depth[::stride, ::stride].astype(np.float32) * scale
    h, w = d.shape
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    u = np.arange(w) - w / 2 + 0.5
    v = np.arange(h) - h / 2 + 0.5
    x = u[None, :] * d / f
    height = cam_height_m - v[:, None] * d / f
    rng = np.sqrt(x ** 2 + d ** 2)
    obstacle = (height > h_min) & (height < h_max) & (d > 0.05)
    ranges = np.where(obstacle, rng, np.inf).min(axis=0)
    ranges[ranges > r_max] = np.inf
    return np.arctan(u / f), ranges


def _rotate(fwd_xz, bearing):
    f = np.asarray(fwd_xz, float) / max(np.linalg.norm(fwd_xz), 1e-9)
    r = np.array([-f[1], f[0]])  # habitat: forward -Z at yaw 0 -> right = +X
    return np.cos(bearing)[..., None] * f + np.sin(bearing)[..., None] * r


@dataclass
class OccupancyMap:
    center_xz: np.ndarray
    size_m: float = 16.0
    res: float = 0.1
    score: np.ndarray = field(init=False)
    virtual: np.ndarray = field(init=False)

    def __post_init__(self):
        self.n = int(round(self.size_m / self.res))
        self.origin = np.asarray(self.center_xz, float) - self.size_m / 2
        self.score = np.zeros((self.n, self.n), np.float32)  # >0 occupied evidence, <0 free evidence
        self.virtual = np.zeros((self.n, self.n), bool)
        self.walls: list = []  # virtual wall segments (world xz endpoints), for drawing

    def cell(self, xz):
        xz = np.asarray(xz, float)
        ij = np.floor((xz - self.origin) / self.res).astype(int)  # [..., 0] = col (x), [..., 1] = row (z)
        return np.clip(ij[..., 1], 0, self.n - 1), np.clip(ij[..., 0], 0, self.n - 1)

    def world(self, row, col):
        return self.origin + (np.array([col, row], float) + 0.5) * self.res

    def integrate(self, pos_xz, fwd_xz, bearings, ranges, r_free_max: float = 4.0):
        dirs = _rotate(fwd_xz, bearings)
        reach = np.minimum(np.where(np.isfinite(ranges), ranges - self.res, r_free_max), r_free_max)
        steps = np.arange(0.25, r_free_max, self.res / 2)
        pts = pos_xz[None, None, :] + dirs[:, None, :] * steps[None, :, None]
        free = steps[None, :] < reach[:, None]
        # Evidence is counted once per cell per frame (not per ray), so a cell needs hits in two frames to
        # read as occupied, and a one-frame speckle of depth noise is carved away by the next views through it.
        free_idx = np.unique(np.ravel_multi_index(self.cell(pts[free]), self.score.shape))
        hit = np.isfinite(ranges)
        hit_idx = np.unique(np.ravel_multi_index(self.cell(pos_xz[None, :] + dirs[hit] * ranges[hit, None]),
                                                 self.score.shape)) if hit.any() else np.zeros(0, int)
        flat = self.score.reshape(-1)
        flat[np.setdiff1d(free_idx, hit_idx, assume_unique=True)] -= 0.5
        flat[hit_idx] += 1.0
        np.clip(self.score, -4.0, 4.0, out=self.score)

    def add_wall(self, pos_xz, fwd_xz, dist: float, half_width: float):
        f = np.asarray(fwd_xz, float) / max(np.linalg.norm(fwd_xz), 1e-9)
        r = np.array([-f[1], f[0]])
        c = np.asarray(pos_xz, float) + f * dist
        self.add_segment(c - r * half_width, c + r * half_width, f)

    def add_segment(self, a, b, away_xz):
        """Virtual wall from a to b (world xz), thickened one cell in the `away_xz` direction."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        f = np.asarray(away_xz, float) / max(np.linalg.norm(away_xz), 1e-9)
        for s in np.linspace(0, 1, int(np.linalg.norm(b - a) / (self.res / 2)) + 2):
            p = a + (b - a) * s
            for off in (0.0, self.res):  # two cells thick so diagonals can't slip through
                self.virtual[self.cell(p + f * off)] = True
        self.walls.append((a.tolist(), b.tolist()))

    def occupied(self):
        return (self.score > 1.5) | self.virtual

    def known_free(self):
        return self.score < -0.5


@numba.njit(cache=True)
def _dijkstra(cost, gr, gc):
    n, m = cost.shape
    dist = np.full((n, m), np.inf)
    dist[gr, gc] = 0.0
    heap = [(0.0, gr, gc)]
    s2 = np.sqrt(2.0)
    while len(heap) > 0:
        d, r, c = heap[0]
        # pop
        last = heap.pop()
        if len(heap) > 0:
            heap[0] = last
            i = 0
            while True:
                lch, rch, small = 2 * i + 1, 2 * i + 2, i
                if lch < len(heap) and heap[lch][0] < heap[small][0]:
                    small = lch
                if rch < len(heap) and heap[rch][0] < heap[small][0]:
                    small = rch
                if small == i:
                    break
                heap[i], heap[small] = heap[small], heap[i]
                i = small
        if d > dist[r, c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if rr < 0 or cc < 0 or rr >= n or cc >= m or not np.isfinite(cost[rr, cc]):
                    continue
                nd = d + cost[rr, cc] * (s2 if dr != 0 and dc != 0 else 1.0)
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    heap.append((nd, rr, cc))
                    j = len(heap) - 1
                    while j > 0:
                        p = (j - 1) // 2
                        if heap[p][0] <= heap[j][0]:
                            break
                        heap[p], heap[j] = heap[j], heap[p]
                        j = p
    return dist


def _disk(radius_cells):
    r = int(np.ceil(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x ** 2 + y ** 2 <= radius_cells ** 2


@dataclass
class Plan:
    carrot_xz: Optional[np.ndarray]
    path_xz: list
    cost_to_goal: float


def cost_grid(m: OccupancyMap, pos_xz, robot_radius: float = 0.25, unknown_cost: float = 1.6,
              obstacle_cost: float = 60.0, clearance_m: float = 0.6, clearance_cost: float = 4.0):
    """Per-cell traversal cost shared by `plan` and frontier selection; also returns the inflated obstacles."""
    occ = m.occupied()
    blocked = binary_dilation(occ, structure=_disk(robot_radius / m.res))
    pr, pc = m.cell(pos_xz)
    blocked[max(pr - 2, 0):pr + 3, max(pc - 2, 0):pc + 3] = False  # never plan from inside the inflation
    near = np.clip(1.0 - distance_transform_edt(~occ) * m.res / clearance_m, 0.0, 1.0)
    cost = np.where(blocked, obstacle_cost, np.where(m.known_free(), 1.0, unknown_cost) + clearance_cost * near)
    return cost.astype(np.float64), blocked


def frontiers(m: OccupancyMap, blocked: np.ndarray, min_cells: int = 4):
    """Exploration frontiers: known-free, unblocked cells next to never-observed ones, grouped into
    8-connected clusters of at least `min_cells`. -> list of (rows, cols) index arrays."""
    from scipy.ndimage import label
    unknown = (np.abs(m.score) < 0.25) & ~m.virtual
    edge = m.known_free() & ~blocked & binary_dilation(unknown, structure=np.ones((3, 3), bool))
    lab, n = label(edge, structure=np.ones((3, 3), bool))
    out = []
    for k in range(1, n + 1):
        rows, cols = np.nonzero(lab == k)
        if rows.size >= min_cells:
            out.append((rows, cols))
    return out


def distances_from(m: OccupancyMap, cost: np.ndarray, pos_xz) -> np.ndarray:
    """Path cost from the robot to every cell (Dijkstra over `cost`)."""
    pr, pc = m.cell(pos_xz)
    return _dijkstra(cost, int(pr), int(pc))


def plan(m: OccupancyMap, pos_xz, goal_xz, robot_radius: float = 0.25, unknown_cost: float = 1.6,
         obstacle_cost: float = 60.0, clearance_m: float = 0.6, clearance_cost: float = 4.0,
         lookahead_m: float = 0.8, cost: np.ndarray = None) -> Plan:
    """Mapped obstacles (inflated by the robot radius) are very expensive rather than impassable: a goal
    the map has wrongly boxed in (e.g. by furniture seen *in* a mirror) stays reachable, while any real
    detour is far cheaper than crossing a wall. Cells within `clearance_m` of an obstacle cost up to
    `clearance_cost` extra, so routes run down the middle of doorways and corridors instead of hugging walls.
    cost: a precomputed `cost_grid` for this map and position (it is copied, not modified)."""
    if cost is None:
        cost, _ = cost_grid(m, pos_xz, robot_radius, unknown_cost, obstacle_cost, clearance_m, clearance_cost)
    else:
        cost = cost.copy()
    pr, pc = m.cell(pos_xz)
    gr, gc = m.cell(goal_xz)
    cost[gr, gc] = 1.0
    dist = _dijkstra(cost, int(gr), int(gc))
    if not np.isfinite(dist[pr, pc]):
        return Plan(None, [], float("inf"))
    path, r, c = [], int(pr), int(pc)
    for _ in range(400):
        best = (dist[r, c], r, c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m.n and 0 <= cc < m.n and dist[rr, cc] < best[0]:
                    best = (dist[rr, cc], rr, cc)
        if best[1] == r and best[2] == c:
            break
        r, c = best[1], best[2]
        path.append(m.world(r, c))
        if (r, c) == (gr, gc):
            break
    carrot = None
    acc, prev = 0.0, np.asarray(pos_xz, float)
    for p in path:
        acc += float(np.linalg.norm(p - prev))
        prev = p
        carrot = p
        if acc >= lookahead_m:
            break
    return Plan(carrot, path, float(dist[pr, pc]))

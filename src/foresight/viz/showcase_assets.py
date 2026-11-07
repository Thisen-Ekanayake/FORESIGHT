"""Assets for the showcase video (tools/make_showcase_video.py): photographic cutaways of the HM3D homes, tilted 3D
"dollhouse" floor plans built from them, and smooth fly-through camera paths.

Cutaway: an orthographic camera 2 m above the main floor looks straight down; everything above it (ceilings, upper
walls) is culled, so the image is a plan of the floor with its furniture. Walls are thin surfaces seen edge-on, so
they show up as the dark gaps between rooms; the dollhouse extrudes those gaps (and a rim around the house) into
low walls and splats the cutaway's pixels as a textured floor, viewed from an oblique orthographic camera.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MINIVAL = Path("data/processed/scene_datasets/hm3d/minival")
EXAMPLE = Path("data/processed/scene_datasets/hm3d/example")
ALL10_CFG = MINIVAL / "hm3d_minival_basis_pointnav.scene_dataset_config.json"
EXAMPLE_CFG = EXAMPLE / "hm3d_example_glb.scene_dataset_config.json"
CUT_H = 2.0  # camera height over the floor; geometry above it is culled


def homes():
    """(name, dataset config, scene ref) for the 10 minival homes and the 3 full-quality example homes."""
    out = []
    for d in sorted(p for p in MINIVAL.iterdir() if p.is_dir() and list(p.glob("*.basis.glb"))):
        out.append((d.name, str(ALL10_CFG), str(next(d.glob("*.basis.glb")).resolve())))
    for d in sorted(p for p in EXAMPLE.iterdir() if p.is_dir() and list(p.glob("*.glb"))):
        out.append((d.name, str(EXAMPLE_CFG), str(next(p for p in d.glob("*.glb") if not p.name.endswith(".semantic.glb")).absolute())))
    return out


def _sim(dataset, scene, sensors):
    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file, cfg.scene_id = dataset, scene
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    if not sim.pathfinder.is_loaded:  # the example config lists no navmeshes: load the one next to the mesh
        sim.pathfinder.load_nav_mesh(str(Path(scene).with_suffix(".navmesh")))
    return sim


def main_floor(sim, n=3000, seed=0):
    """Height of the floor most of the navmesh is on, and the x/z extent of that floor."""
    sim.pathfinder.seed(seed)
    pts = np.array([sim.pathfinder.get_random_navigable_point() for _ in range(n)])
    pts = pts[np.isfinite(pts).all(1)]
    ys = np.round(pts[:, 1] / 0.2) * 0.2
    vals, counts = np.unique(ys, return_counts=True)
    y0 = vals[counts.argmax()]
    on = pts[np.abs(pts[:, 1] - y0) < 0.3]
    return float(np.median(on[:, 1])), on[:, [0, 2]].min(0), on[:, [0, 2]].max(0)


def cutaway(dataset, scene, res=1400, margin_m=1.0, floor_y=None):
    """-> rgb (res, res, 3), meta {center [x, y, z], width_m, res}. Image right = +x, image down = +z."""
    import habitat_sim
    probe = _sim(dataset, scene, [])
    fy, lo, hi = main_floor(probe)
    probe.close()
    fy = fy if floor_y is None else floor_y
    width = float((hi - lo).max() + 2 * margin_m)
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid, spec.sensor_type, spec.resolution = "rgb", habitat_sim.SensorType.COLOR, [res, res]
    spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    spec.position, spec.orientation, spec.ortho_scale = [0.0, CUT_H, 0.0], [-math.pi / 2, 0.0, 0.0], 1.0 / width
    sim = _sim(dataset, scene, [spec])
    c = np.array([(lo[0] + hi[0]) / 2, fy, (lo[1] + hi[1]) / 2])
    sim.get_agent(0).set_state(habitat_sim.AgentState(position=c))
    rgb = np.asarray(sim.get_sensor_observations()["rgb"][..., :3])
    sim.close()
    return rgb, {"center": c.tolist(), "width_m": width, "res": res}


# ---------------------------------------------------------------- dollhouse


@dataclass
class DollView:
    """Oblique orthographic view: world (x, y, z) -> screen (u, v) pixels."""
    cx: float
    cz: float
    az: float  # rad, rotation of the plan about the vertical axis
    el: float  # rad, elevation of the viewing direction (90 deg = straight down)
    s: float  # pixels per metre
    u0: float
    v0: float

    def _rot(self, x, z):
        ca, sa = math.cos(self.az), math.sin(self.az)
        dx, dz = np.asarray(x) - self.cx, np.asarray(z) - self.cz
        return dx * ca - dz * sa, dx * sa + dz * ca

    def project(self, x, y, z):
        xr, zr = self._rot(x, z)
        return self.u0 + xr * self.s, self.v0 + (zr * math.sin(self.el) - np.asarray(y) * math.cos(self.el)) * self.s

    def depth(self, x, y, z):  # larger = nearer the viewer
        _, zr = self._rot(x, z)
        return zr * math.cos(self.el) + np.asarray(y) * math.sin(self.el)


WALL_TOP, WALL_SIDE, SLAB = np.array([246, 247, 250]), np.array([196, 201, 212]), np.array([34, 40, 58])


def dollhouse(rgb, meta, size=(900, 900), el_deg=52.0, az_deg=0.0, wall_h=0.9, slab_h=0.18, pad=30, fit_xz=None):
    """Render the cutaway as a tilted 3D plan. -> RGBA image (transparent background), DollView.
    fit_xz: optional (lo, hi) world x/z box to frame instead of the whole house."""
    from scipy import ndimage

    R = rgb.shape[0]
    W, H = size
    mpp = meta["width_m"] / R
    cx, cz = meta["center"][0], meta["center"][2]
    shown = rgb.max(-1) > 14
    k = max(1, int(round(0.018 / mpp)))  # work grid ~1.8 cm
    if k > 1:
        rgb = rgb[: R // k * k, : R // k * k].reshape(R // k, k, R // k, k, 3).mean((1, 3)).astype(np.uint8)
        shown = shown[: R // k * k, : R // k * k].reshape(R // k, k, R // k, k).mean((1, 3)) > 0.5
    n = shown.shape[0]
    g = mpp * k
    # Walls: gaps between rendered regions narrower than ~0.5 m, plus a ~10 cm rim around the house.
    close_r = max(1, int(round(0.25 / g)))
    rim_r = max(1, int(round(0.10 / g)))
    st = ndimage.generate_binary_structure(2, 1)
    filled = ndimage.binary_closing(shown, st, iterations=close_r)
    rim = ndimage.binary_dilation(shown, st, iterations=rim_r)
    walls = (filled | rim) & ~shown
    walls = ndimage.binary_opening(walls, st, iterations=1) | (walls & ndimage.binary_erosion(rim, st, 1) & ~filled)
    house = shown | walls
    ii, jj = np.nonzero(house)
    x = cx + (jj + 0.5 - n / 2) * g
    z = cz + (ii + 0.5 - n / 2) * g

    view = DollView(cx, cz, math.radians(az_deg), math.radians(el_deg), 1.0, 0.0, 0.0)
    if fit_xz is not None:
        (lx, lz), (hx, hz) = fit_xz
        bx, bz = np.array([lx, hx, hx, lx]), np.array([lz, lz, hz, hz])
        us, vs = view.project(np.r_[bx, bx], np.r_[np.zeros(4), np.full(4, wall_h)], np.r_[bz, bz])
    else:
        us, vs = view.project(np.r_[x, x], np.r_[np.zeros(len(x)), np.full(len(x), wall_h)], np.r_[z, z])
    vs_lo = vs.min() - 0.0
    vs_hi = vs.max() + slab_h * math.cos(view.el)
    s = min((W - 2 * pad) / max(us.max() - us.min(), 1e-6), (H - 2 * pad) / max(vs_hi - vs_lo, 1e-6))
    view.s = s
    view.u0 = W / 2 - (us.max() + us.min()) / 2 * s
    view.v0 = H / 2 - (vs_hi + vs_lo) / 2 * s

    pts, cols = [], []
    fl = shown[ii, jj]
    # slab under the floor (its front faces show along the near edges)
    for y in np.linspace(-slab_h, -0.01, max(2, int(slab_h * s * math.cos(view.el) * 1.5))):
        pts.append((x, np.full(len(x), y), z))
        cols.append(np.broadcast_to(SLAB, (len(x), 3)))
    pts.append((x[fl], np.zeros(fl.sum()), z[fl]))
    cols.append(rgb[ii[fl], jj[fl]])
    wx, wz = x[~fl], z[~fl]
    levels = np.linspace(0.0, wall_h, max(2, int(wall_h * s * math.cos(view.el) * 1.6)))
    for k_, y in enumerate(levels):
        pts.append((wx, np.full(len(wx), y), wz))
        shade = 0.82 + 0.18 * (y / wall_h)  # a little darker toward the floor
        cols.append(np.broadcast_to((WALL_SIDE * shade).astype(np.uint8) if k_ < len(levels) - 1 else WALL_TOP, (len(wx), 3)))
    X = np.concatenate([p[0] for p in pts])
    Y = np.concatenate([p[1] for p in pts])
    Z = np.concatenate([p[2] for p in pts])
    C = np.concatenate(cols).astype(np.uint8)
    u, v = view.project(X, Y, Z)
    d = view.depth(X, Y, Z)
    order = np.argsort(d, kind="stable")
    u, v, C = u[order], v[order], C[order]
    img = np.zeros((H, W, 4), np.uint8)
    for du in (0, 1):
        for dv in (0, 1):
            ui, vi = np.floor(u).astype(int) + du, np.floor(v).astype(int) + dv
            ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            img[vi[ok], ui[ok], :3] = C[ok]
            img[vi[ok], ui[ok], 3] = 255
    return Image.fromarray(img, "RGBA"), view


# ---------------------------------------------------------------- fly-throughs


def flythrough(dataset, scene, seconds=4.5, fps=30, speed=0.65, seed=0, n_try=400):
    """A smooth camera path along the navmesh through open space: of a few hundred random 5-12 m shortest paths on
    the main floor, the nearly straight one with the most clearance from obstacles (so the camera looks down a room,
    not at a wall), resampled at constant speed, yaw following the path ~1.5 m ahead (smoothed).
    -> (positions (N,3), yaws (N,))."""
    import habitat_sim
    from scipy.ndimage import gaussian_filter1d
    sim = _sim(dataset, scene, [])
    fy, _, _ = main_floor(sim)
    rng = np.random.default_rng(seed)
    pf = sim.pathfinder
    pf.seed(int(rng.integers(1 << 30)))
    best, best_score = None, -np.inf
    for _ in range(n_try):
        a, b = pf.get_random_navigable_point(), pf.get_random_navigable_point()
        if abs(a[1] - fy) > 0.3 or abs(b[1] - fy) > 0.3:
            continue
        p = habitat_sim.ShortestPath()
        p.requested_start, p.requested_end = a, b
        if not pf.find_path(p) or not 5.0 < p.geodesic_distance < 12.0:
            continue
        pts = np.array(p.points, float)
        straight = np.linalg.norm(pts[-1] - pts[0]) / p.geodesic_distance
        if straight < 0.85:
            continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s_ = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s_[-1], 0.25)
        samp = np.stack([np.interp(q, s_, pts[:, k]) for k in range(3)], 1)
        clear = np.array([pf.distance_to_closest_obstacle(x.astype(np.float32), 3.0) for x in samp])
        score = clear.mean() + 0.5 * clear.min() + 0.1 * p.geodesic_distance
        if score > best_score:
            best, best_score = pts, score
    sim.close()
    pts = best
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    dense = np.stack([np.interp(np.arange(0, s[-1], 0.02), s, pts[:, k]) for k in range(3)], 1)
    sm = np.stack([gaussian_filter1d(dense[:, k], 30, mode="nearest") for k in range(3)], 1)
    seg = np.linalg.norm(np.diff(sm, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = int(seconds * fps)
    total = min(s[-1] - 1.5, speed * seconds)
    q = np.linspace(0, total, n)  # from the start, so the camera looks along the whole path
    pos = np.stack([np.interp(q, s, sm[:, k]) for k in range(3)], 1)
    ahead = np.stack([np.interp(np.minimum(q + 1.5, s[-1]), s, sm[:, k]) for k in range(3)], 1)
    v = ahead - pos
    yaw = np.unwrap(np.arctan2(-v[:, 0], -v[:, 2]))
    yaw = gaussian_filter1d(yaw, 12, mode="nearest")
    return pos, yaw


FLY = [("00800-TEEsavR23oF", 3), ("00770-NBg5UqG3di3", 1), ("00807-rsggHU7g7dh", 4), ("00803-k1cupFYWXJ6", 4)]


def build_assets(out: Path, pointnav: Path, episode: str):
    out.mkdir(parents=True, exist_ok=True)
    by = {h[0]: h for h in homes()}
    for name, dataset, scene in by.values():
        if (out / f"cut_{name}.png").exists():
            continue
        rgb, meta = cutaway(dataset, scene)
        Image.fromarray(rgb).save(out / f"cut_{name}.png")
        (out / f"cut_{name}.json").write_text(json.dumps(meta, indent=1))
        print(f"cutaway {name}: {meta['width_m']:.1f} m", flush=True)
    for name, seed in FLY:
        f = out / f"fly_{name}.npz"
        if f.exists():
            continue
        pos, yaw = flythrough(by[name][1], by[name][2], seed=seed)
        np.savez(f, pos=pos, yaw=yaw)
        print(f"fly-through {name}: {np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.1f} m", flush=True)

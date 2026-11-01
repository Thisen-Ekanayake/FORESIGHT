#!/usr/bin/env python
"""Export an HM3D scene (Basis-compressed GLB) to plain meshes + JPEG textures for Isaac Sim.

Isaac Sim's asset converter can't read HM3D's `image/x-basis` textures (GOOGLE_texture_basis), and no basisu tool is
installed, so this runs in the habitat env and lets Magnum's BasisImporter (bundled with habitat-sim) transcode
them to RGBA8. Output: <out>/meshes.npz (per GLB mesh: positions, texcoords, indices, texture index, node name; in
the GLB's own frame, z up: Habitat (x, y, z) = GLB (x, z, -y)) and <out>/tex_NNN.jpg; `tools/isaac/render_replay.py` builds the USD stage from it.

Node names are kept because HM3D names its mirror surfaces ("mirror001_…"): the USD build gives those a real
reflective material.

  python tools/export_hm3d_for_isaac.py --scene 00808-y9hTuugGdiq --out results/isaac/00808
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from magnum import trade
from PIL import Image

ROOT = Path("data/processed/scene_datasets/hm3d/minival")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="00808-y9hTuugGdiq")
    ap.add_argument("--out", default="results/isaac/00808")
    ap.add_argument("--tex-size", type=int, default=2048)
    args = ap.parse_args()
    glb = ROOT / args.scene / f"{args.scene.split('-')[1]}.basis.glb"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(glb, "rb") as f:  # the glTF JSON, for node names and the material -> image mapping
        f.read(12)
        n, _ = struct.unpack("<II", f.read(8))
        j = json.loads(f.read(n))
    tex_of_material = []
    for mat in j["materials"]:
        t = j["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        tex_of_material.append(t.get("source", t.get("extensions", {}).get("GOOGLE_texture_basis", {}).get("source")))

    mgr = trade.ImporterManager()
    mgr.metadata("BasisImporter").configuration["format"] = "RGBA8"
    imp = mgr.load_and_instantiate("GltfImporter")
    imp.open_file(str(glb))
    for i in range(imp.image2d_count):
        img = imp.image2d(i)
        a = np.array(img.pixels, copy=False)[::-1, :, :3]  # Magnum images are bottom-up
        im = Image.fromarray(np.ascontiguousarray(a))
        if im.size[0] > args.tex_size:
            im = im.resize((args.tex_size, args.tex_size), Image.LANCZOS)
        im.save(out / f"tex_{i:03d}.jpg", quality=92)
    meshes = {}
    for node in j["nodes"]:
        k = node["mesh"]
        md = imp.mesh(k)
        prim = j["meshes"][k]["primitives"][0]
        name = node["name"]
        # Magnum flips texture coordinates to its bottom-up image convention, which is also USD's st convention
        meshes[f"{k:03d}_pos"] = np.array(md.attribute(trade.MeshAttribute.POSITION), np.float32)
        meshes[f"{k:03d}_uv"] = np.array(md.attribute(trade.MeshAttribute.TEXTURE_COORDINATES), np.float32)
        meshes[f"{k:03d}_idx"] = np.array(md.indices, np.int32)
        meshes[f"{k:03d}_tex"] = np.array(tex_of_material[prim["material"]], np.int32)
        meshes[f"{k:03d}_name"] = np.array(name)
    np.savez_compressed(out / "meshes.npz", **meshes)
    n_mirror = sum(1 for node in j["nodes"] if node["name"].startswith("mirror"))
    print(f"{len(j['nodes'])} meshes ({n_mirror} mirror surfaces), {imp.image2d_count} textures -> {out}")


if __name__ == "__main__":
    main()

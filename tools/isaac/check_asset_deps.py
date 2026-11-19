"""Find every file the downloaded NVIDIA scenes/robots reference but that isn't in the local asset mirror, and fetch it
(other Isaac folders such as shared materials or props). Repeats until nothing is missing.

  /ml/isaac-sim/python.sh tools/isaac/check_asset_deps.py <asset_root> <usd>...

Run on the workstation after copying the folder made by tools/isaac/download_isaac_assets.sh.
"""
import os
import subprocess
import sys

from pxr import Sdf, UsdUtils

root = os.path.abspath(sys.argv[1])
here = os.path.dirname(os.path.abspath(__file__))
REMOTE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/"


def to_key(path, anchor):
    """A referenced asset path -> its key under the asset root (None if it can't live there)."""
    p = str(path)
    for pre in (REMOTE, "omniverse://localhost/NVIDIA/Assets/Isaac/6.0/"):
        if p.startswith(pre):
            return p[len(pre):]
    if "://" in p or p.startswith("{"):
        return None
    full = os.path.normpath(os.path.join(os.path.dirname(anchor), p)) if not os.path.isabs(p) else p
    return os.path.relpath(full, root) if full.startswith(root) else None


for rnd in range(6):
    missing = set()
    for top in sys.argv[2:]:
        top = os.path.join(root, top)
        if not os.path.exists(top):
            print("not downloaded:", top)
            continue
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(top))
        for u in unresolved:
            k = to_key(u, top)
            if k and not os.path.exists(os.path.join(root, k)):
                missing.add(k)
        # also scan each layer's own relative references (ComputeAllDependencies reports unresolved by path only)
        for lyr in layers:
            for ref in lyr.GetCompositionAssetDependencies() + list(lyr.externalReferences):
                k = to_key(ref, lyr.realPath)
                if k and not os.path.exists(os.path.join(root, k)):
                    missing.add(k)
    print(f"round {rnd}: {len(missing)} missing files")
    if not missing:
        print("all dependencies present")
        break
    subprocess.run(["python3", "-I", os.path.join(here, "s3_assets.py"), "keys", root,
                    *sorted(missing)], check=False)

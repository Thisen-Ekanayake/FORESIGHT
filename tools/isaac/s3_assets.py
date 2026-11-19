#!/usr/bin/env python3
"""List / download NVIDIA's public Isaac Sim 6.0 assets (S3) into a local mirror of the asset root.

  python3 -I tools/isaac/s3_assets.py list  <prefix>...            # sizes only, nothing downloaded
  python3 -I tools/isaac/s3_assets.py fetch <dest> <prefix>...     # resumable (skips files already complete)
  python3 -I tools/isaac/s3_assets.py keys  <dest> <key>...        # fetch individual files (used by check_asset_deps.py)

Prefixes are relative to the asset root, e.g. "Isaac/Environments/Office/". Files land at
<dest>/Isaac/... so <dest> can be given to Isaac Sim as its asset root.
"""
import concurrent.futures as cf
import os
import re
import sys
import urllib.parse
import urllib.request

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
ROOT = "Assets/Isaac/6.0/"


def ls(prefix):
    out, tok = [], None
    while True:
        q = {"list-type": "2", "prefix": ROOT + prefix, "max-keys": "1000"}
        if tok:
            q["continuation-token"] = tok
        x = urllib.request.urlopen(BUCKET + "?" + urllib.parse.urlencode(q), timeout=120).read().decode()
        out += [(k[len(ROOT):], int(s)) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", x)]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", x)
        if not m:
            return out
        tok = m.group(1)


def get(dest, key, size=None):
    path = os.path.join(dest, key)
    if size is not None and os.path.exists(path) and os.path.getsize(path) == size:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = BUCKET + urllib.parse.quote(ROOT + key)
    tmp = path + ".part"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, path)
            return os.path.getsize(path)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                print(f"  FAILED {key}: {e}", flush=True)
                return 0


def main():
    cmd = sys.argv[1]
    if cmd == "list":
        tot = 0
        for p in sys.argv[2:]:
            items = ls(p)
            s = sum(v for _, v in items)
            tot += s
            print(f"{s / 1e6:9.1f} MB  {len(items):6d} files  {p}")
        print(f"{tot / 1e9:9.2f} GB  total")
    elif cmd == "fetch":
        dest = sys.argv[2]
        for p in sys.argv[3:]:
            items = ls(p)
            print(f"{p}: {len(items)} files, {sum(v for _, v in items) / 1e6:.0f} MB", flush=True)
            done = 0
            with cf.ThreadPoolExecutor(8) as ex:
                for n, _ in enumerate(ex.map(lambda kv: get(dest, *kv), items), 1):
                    done += 1
                    if n % 200 == 0 or n == len(items):
                        print(f"  {n}/{len(items)}", flush=True)
    elif cmd == "keys":
        dest = sys.argv[2]
        for k in sys.argv[3:]:
            print(f"  {k}: {get(dest, k)} bytes", flush=True)


if __name__ == "__main__":
    main()

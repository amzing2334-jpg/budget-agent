"""Check a downloaded snapshot against metadata retrieved from the official Hub.

This reads files and hashes bytes; it does not load or run a model.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    verified = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if not (name.endswith((".json", ".safetensors")) or name == "merges.txt"):
            continue
        path = args.snapshot / name
        if "lfs" in item:
            digest = hashlib.sha256()
            expected = item["lfs"]["sha256"]
        else:
            digest = hashlib.sha1()
            digest.update(f"blob {path.stat().st_size}\0".encode())
            expected = item["blobId"]
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"Hash mismatch: {name}")
        verified.append({"file": name, "algorithm": digest.name, "digest": actual})
    receipt = {"model": metadata["id"], "revision": metadata["sha"], "verified_files": verified}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(verified)} files against official Hub metadata")


if __name__ == "__main__":
    main()

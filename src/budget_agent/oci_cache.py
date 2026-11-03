"""Select verified files for an offline import of one pinned platform image."""
import hashlib
import json
from pathlib import Path
import re


def cache_files(layout: Path, spec: dict, frozen_manifest: bytes) -> list[str]:
    image = spec["image"]
    manifest_digest = "sha256:" + hashlib.sha256(frozen_manifest).hexdigest()
    if manifest_digest != image["manifest_digest"]:
        raise ValueError("Frozen platform manifest differs from the control spec")
    manifest = json.loads(frozen_manifest)
    root = {"digest": manifest_digest, "size": len(frozen_manifest)}
    blobs = {}
    for desc in (root, manifest["config"], *manifest["layers"]):
        digest = desc["digest"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"Unsupported blob digest: {digest}")
        blobs["blobs/sha256/" + digest[7:]] = desc["size"]

    # Locate all absent or truncated files before reading any large layer.
    problems = []
    metadata = ["oci-layout", "index.json"]
    for name in [*metadata, *blobs]:
        path = layout / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"Missing regular cache file: {name}")
        elif name in blobs and path.stat().st_size != blobs[name]:
            problems.append(f"Wrong size: {name} (expected {blobs[name]} bytes)")
    if problems:
        raise ValueError("\n".join(problems))

    marker = json.loads((layout / "oci-layout").read_bytes())
    if not isinstance(marker, dict) or marker.get("imageLayoutVersion") != "1.0.0":
        raise ValueError("Expected OCI image layout version 1.0.0")
    index = json.loads((layout / "index.json").read_bytes())
    entries = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError("Expected a single platform manifest in index.json")
    entry = entries[0]
    platform = entry.get("platform")
    if (index.get("schemaVersion") != 2 or entry.get("digest") != manifest_digest
            or entry.get("size") != len(frozen_manifest) or not isinstance(platform, dict)
            or f"{platform.get('os')}/{platform.get('architecture')}" != image["platform"]):
        raise ValueError("index.json does not select the pinned platform manifest")

    for name in blobs:
        with (layout / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != name.rsplit("/", 1)[1]:
            raise ValueError(f"Wrong SHA-256: {name}")
    return [*metadata, *blobs]

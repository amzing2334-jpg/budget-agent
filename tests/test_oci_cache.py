import hashlib
import json
import subprocess
import sys

import pytest

from budget_agent.cli import main


@pytest.fixture
def cache(tmp_path):
    layout = tmp_path / "layout"
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    members = {}

    def blob(name, data, media_type):
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        members[name] = f"blobs/sha256/{digest}"
        return {"mediaType": f"application/vnd.oci.image.{media_type}",
                "digest": f"sha256:{digest}", "size": len(data)}

    config = blob("config", b'{"os":"linux","architecture":"arm64"}', "config.v1+json")
    layers = [blob("base", b"base layer", "layer.v1.tar"),
              blob("last", b"last layer", "layer.v1.tar")]
    frozen = json.dumps({"schemaVersion": 2, "config": config, "layers": layers},
                        indent=2).encode() + b"\n"
    manifest = blob("manifest", frozen, "manifest.v1+json")
    spec = {"image": {"manifest_digest": manifest["digest"], "platform": "linux/arm64"}}
    entry = {**manifest, "platform": {"os": "linux", "architecture": "arm64"}}
    (tmp_path / "manifest.json").write_bytes(frozen)
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (layout / "oci-layout").write_bytes(b'{"imageLayoutVersion":"1.0.0"}')
    (layout / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [entry]}),
                                       encoding="utf-8")
    return layout, members


def run_cache(layout):
    return main(["swe-cache-files", "--layout", str(layout),
                 "--spec", str(layout.parent / "spec.json"),
                 "--manifest", str(layout.parent / "manifest.json")])


def assert_rejected(layout, capsys):
    assert run_cache(layout) != 0
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.strip()
    return output.err


def test_cache_lists_only_required_relative_tar_members(cache, capsys):
    layout, members = cache
    (layout / "private-grader.py").write_bytes(b"unrelated")
    (layout / "blobs" / "sha256" / ("f" * 64)).write_bytes(b"unreferenced")
    assert run_cache(layout) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert sorted(output.out.splitlines()) == sorted(["oci-layout", "index.json", *members.values()])


def test_member_file_uses_tar_line_endings(cache):
    layout, members = cache
    raw = subprocess.check_output([sys.executable, "-m", "budget_agent", "swe-cache-files",
                                   "--layout", str(layout), "--spec", str(layout.parent / "spec.json"),
                                   "--manifest", str(layout.parent / "manifest.json")])
    expected = ["", "oci-layout", "index.json", *members.values()]
    assert sorted(raw.split(b"\n")) == sorted(name.encode("ascii") for name in expected)


@pytest.mark.parametrize(("name", "damage"), [
    ("last", "missing"), ("last", "truncated"), ("last", "same_size"),
    ("config", "same_size"), ("manifest", "same_size"),
])
def test_invalid_blob_emits_no_members_and_identifies_digest(cache, capsys, name, damage):
    layout, members = cache
    path = layout / members[name]
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.write_bytes(b"!" * path.stat().st_size)
    assert path.name in assert_rejected(layout, capsys)


@pytest.mark.parametrize("change", ["digest", "size", "platform", "multiple", "schema"])
def test_index_must_select_the_single_frozen_platform(cache, capsys, change):
    layout, _ = cache
    path = layout / "index.json"
    index = json.loads(path.read_bytes())
    entry, = index["manifests"]
    if change == "multiple":
        index["manifests"].append(entry)
    elif change == "schema":
        index["schemaVersion"] = 1
    else:
        entry[change] = {"digest": "sha256:" + "0" * 64, "size": entry["size"] + 1,
                         "platform": {"architecture": "amd64", "os": "linux"}}[change]
    path.write_text(json.dumps(index), encoding="utf-8")
    assert_rejected(layout, capsys)


def test_supplied_manifest_must_match_pinned_bytes(cache, capsys):
    layout, _ = cache
    frozen = layout.parent / "manifest.json"
    frozen.write_bytes(frozen.read_bytes().rstrip())
    assert_rejected(layout, capsys)


def test_layout_version_must_match(cache, capsys):
    layout, _ = cache
    (layout / "oci-layout").write_bytes(b'{"imageLayoutVersion":"2.0.0"}')
    assert_rejected(layout, capsys)

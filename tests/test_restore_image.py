import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CASE = Path(__file__).resolve().parents[1] / "runs/iteration-20261102-14182-controls"
MEMBER = "blobs/sha256/" + "a" * 64
LEGACY_IMPORT = (
    'set -euo pipefail; tar -C "$1" -cf - oci-layout index.json blobs | '
    'sudo ctr -a /run/docker/containerd/containerd.sock -n moby '
    'images import --local --platform linux/amd64 -'
)


@unittest.skipUnless(sys.platform == "linux", "image restoration uses Linux tar and bash")
class RestoreImageTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.layout = self.root / "cache with spaces"
        self.layout.mkdir()
        (self.layout / "index.json").write_bytes((CASE / "cache-index.json").read_bytes())
        (self.layout / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')

        # Exercise the real shell and tar. The validator's digest checks are
        # independently covered in test_oci_cache; this stub supplies its file list.
        self.python = self.root / "validator"
        self.python.write_text("""#!/bin/sh
if [ "$1" = "-c" ]; then exec python3 "$@"; fi
[ "$1 $2 $3 $4 $6 $8" = "-m budget_agent swe-cache-files --spec --manifest --layout" ] || exit 31
[ -f "$9/$MEMBER" ] || exit 2
printf '%s\\n' oci-layout index.json "$MEMBER"
""")
        self.python.chmod(0o755)
        sudo = self.root / "sudo"
        sudo.write_text("""#!/bin/sh
printf '%s\\n' "$*" >> "$REPLAY/calls"
case " $* " in
    *" images import --local --platform linux/amd64 - ") tar -tf - > "$REPLAY/members";;
    *" docker image inspect "*) exit 0;;
    *) exit 32;;
esac
""")
        sudo.chmod(0o755)

    def run_shell(self, *args):
        return subprocess.run(
            ["bash", *args], text=True, capture_output=True, timeout=120,   # a hang guard, not a speed limit
            env={**os.environ, "PATH": f"{self.root}:{os.environ['PATH']}",
                 "REPLAY": str(self.root), "MEMBER": MEMBER})

    def restore(self):
        return self.run_shell(str(CASE / "restore_image.sh"), str(self.layout), str(self.python))

    def test_incomplete_cache_no_longer_dispatches_import(self):
        legacy = self.run_shell("-c", LEGACY_IMPORT, "replay", str(self.layout))
        self.assertNotEqual(legacy.returncode, 0)
        calls = self.root / "calls"
        self.assertEqual(calls.read_text().count("images import"), 1)
        calls.unlink()
        self.assertEqual(self.restore().returncode, 2)
        self.assertFalse(calls.exists())

    def test_import_contains_only_listed_files_and_requires_the_frozen_index(self):
        blob = self.layout / MEMBER
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"test layer")
        (blob.parent / "unrelated").write_bytes(b"extra file under blobs")
        result = self.restore()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.root / "calls"
        self.assertEqual(calls.read_text().count("images import"), 1)
        self.assertEqual((self.root / "members").read_text().splitlines(),
                         ["oci-layout", "index.json", MEMBER])

        calls.unlink()
        # This is still equivalent JSON, but its bytes differ from the frozen index.
        index = self.layout / "index.json"
        index.write_bytes(index.read_bytes() + b"\n")
        self.assertNotEqual(self.restore().returncode, 0)
        self.assertFalse(calls.exists())

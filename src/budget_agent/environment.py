"""Execution environments and independent grading for the bundled smoke fixtures.

Local execution is deliberately not presented as a security sandbox. Real model
commands require explicit opt-in; Docker is the preferred execution boundary.
"""
from __future__ import annotations

import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

from .schema import RunConfig, Task
from .security import execute_authorized
from .verification import VerificationResult, fixture_outcome


class EnvironmentError(RuntimeError):
    pass


def _safe_env() -> dict[str, str]:
    keep = ("PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def _process(command: str | list[str], cwd: Path, timeout: int, *, shell: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, cwd=cwd, shell=shell, stdout=output,
                                   stderr=subprocess.STDOUT, env=_safe_env(), **kwargs)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            else:
                import signal
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
        output.seek(0)
        raw = output.read(100_001)
    text = raw[:100_000].decode("utf-8", errors="replace")
    if len(raw) > 100_000:
        text += "\n[output truncated at 100000 bytes]"
    if timed_out:
        text += f"\n[command timed out after {timeout}s; process tree terminated]"
    return {"output": text, "ok": process.returncode == 0 and not timed_out,
            "returncode": process.returncode, "timed_out": timed_out,
            "output_truncated": len(raw) > 100_000,
            "duration_seconds": time.monotonic() - started}


def _files(root: Path) -> dict[str, bytes]:
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        if path.is_symlink():
            raise EnvironmentError(f"symlinks are not supported: {rel.as_posix()}")
        if path.is_file():
            result[rel.as_posix()] = path.read_bytes()
    return result


def _repository_tool(root: Path, action: dict[str, Any], editable: list[str] | None) -> str:
    """The same bounded file operations run locally and inside task containers."""
    import json
    import os
    from pathlib import Path, PurePosixPath

    root = root.resolve()
    tool = action["tool"]
    page_chars = 16_000

    def target(value: str) -> Path:
        normalized = value.replace("\\", "/")
        relative = PurePosixPath(normalized)
        if not normalized or relative.is_absolute() or ":" in normalized or ".." in relative.parts or any(part.rstrip(". ").lower() == ".git" for part in relative.parts):
            raise PermissionError("path must remain inside the repository; .git is not accessible")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
            raise PermissionError("symlink paths are not accessible")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise PermissionError("path escapes the workspace")
        return resolved

    def header(**metadata: Any) -> str:
        return tool + " " + json.dumps(metadata, ensure_ascii=False)

    def files(scope: Path):
        if scope.is_file():
            yield scope
            return
        if not scope.is_dir():
            raise FileNotFoundError(f"path does not exist: {scope.relative_to(root).as_posix()}")
        for directory, dirs, names in os.walk(scope):
            base = Path(directory)
            dirs[:] = sorted(name for name in dirs if name.rstrip(". ").lower() not in {".git", "__pycache__", ".pytest_cache"} and not (base / name).is_symlink())
            for name in sorted(names):
                path = base / name
                if not path.is_symlink() and path.is_file():
                    yield path

    if tool in {"list_files", "search"}:
        scope = target(action.get("path", "."))
        offset = action.get("offset", 0)
        limit = action.get("limit", 100 if tool == "list_files" else 50)
        maximum = 200 if tool == "list_files" else 100
        if offset < 0 or not 1 <= limit <= maximum:
            raise ValueError(f"{tool} needs offset >= 0 and limit between 1 and {maximum}")
        if tool == "search" and not action["query"]:
            raise ValueError("search query must be non-empty")

        def records():
            for path in files(scope):
                relative = path.relative_to(root).as_posix()
                if tool == "list_files":
                    yield relative
                    continue
                query = action["query"]
                with path.open(encoding="utf-8", errors="replace") as stream:
                    for number, line in enumerate(stream, 1):
                        position = line.find(query)
                        if position >= 0:
                            text = line.rstrip("\r\n")
                            start = max(0, position - 120)
                            excerpt = text[start:start + 500]
                            if start:
                                excerpt = "[line prefix omitted] " + excerpt
                            if start + 500 < len(text):
                                excerpt += " [line suffix omitted]"
                            yield f"{relative}:{number}: {excerpt}"

        rows, size, more = [], 0, False
        for index, row in enumerate(records()):
            if index < offset:
                continue
            if len(rows) == limit or size + len(row) + 1 > page_chars:
                more = True
                break
            rows.append(row)
            size += len(row) + 1
        metadata = dict(path=scope.relative_to(root).as_posix(), offset=offset,
                        returned=len(rows), has_more=more,
                        next_offset=offset + len(rows) if more else None)
        return header(**metadata) + "\n" + "\n".join(rows)

    path = target(action["path"])
    relative = path.relative_to(root).as_posix()
    if tool == "read_file":
        start = action.get("start_line", 1)
        end = action.get("end_line", start + 99)
        if start < 1 or end < start or end - start >= 200:
            raise ValueError("read_file needs 1 <= start_line <= end_line and at most 200 lines per call")
        rows, size, total, shown_end, clipped = [], 0, 0, None, False
        with path.open(encoding="utf-8") as stream:
            for total, line in enumerate(stream, 1):
                if start <= total <= end:
                    text = line.rstrip("\r\n")
                    if len(text) > 1_000:
                        text = text[:1_000] + " [line truncated after 1000 characters]"
                        clipped = True
                    row = f"{total}: {text}"
                    if size + len(row) + 1 <= page_chars and (shown_end is None or shown_end == total - 1):
                        rows.append(row)
                        size += len(row) + 1
                        shown_end = total
        next_line = shown_end + 1 if shown_end is not None and shown_end < total else None
        return header(path=relative, total_lines=total, start_line=start if rows else None,
                      end_line=shown_end, next_start_line=next_line,
                      truncated_line_content=clipped) + "\n" + "\n".join(rows)

    if tool not in {"write_file", "edit_file"}:
        raise ValueError(f"unknown environment tool: {tool}")
    parts = path.relative_to(root).parts
    permitted = relative in editable if editable is not None else not any(part in {"tests", "test", ".git"} for part in parts) and not path.name.startswith(("test_", "_private_"))
    if not permitted:
        raise PermissionError("this task only permits source edits; tests and verifier are protected")
    if tool == "write_file":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(action["content"], encoding="utf-8")
        return f"Wrote {relative}"
    content = path.read_text(encoding="utf-8")
    old = action["old"]
    if not old:
        raise ValueError("edit_file old must be non-empty")
    matches = content.count(old)
    if matches != 1:
        needle = old if matches else next((line for line in old.splitlines() if line.strip()), old)
        positions, offset = [], 0
        for _ in range(5):
            position = content.find(needle, offset)
            if position < 0:
                break
            line = content.count("\n", 0, position) + 1
            column = position - content.rfind("\n", 0, position)
            positions.append(f"line {line}, column {column}")
            offset = position + len(needle)
        hint = ("Matching block starts" if matches else "First nonempty old line occurs") + ": " + "; ".join(positions) if positions else "No anchor found; read the current file before editing."
        raise ValueError(f"edit_file old must match exactly once; found {matches}. No file changed. {hint}. Use read_file on {relative} around these lines, then include enough exact surrounding text in old to match once.")
    path.write_text(content.replace(old, action["new"], 1), encoding="utf-8")
    return f"Edited {relative}"


_FILE_TOOL_SOURCE = "from __future__ import annotations\n" + inspect.getsource(_repository_tool)


def _verification_command(python: str) -> tuple[list[str], str]:
    """Require a completed test run, not just a subprocess that exits with zero."""
    marker = "BUDGET_AGENT_GRADE_" + uuid.uuid4().hex
    code = (
        "import json,runpy,sys,types,unittest\n" + inspect.getsource(VerificationResult) + "\n"
        "ns=runpy.run_path('_private_verifier.py',run_name='private_verifier'); "
        "suite=unittest.defaultTestLoader.loadTestsFromModule(types.SimpleNamespace(**ns)); "
        "result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2,resultclass=VerificationResult).run(suite); "
        "print(" + repr(marker) + "+json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),"
        "'errors':len(result.errors),'skipped':len(result.skipped),'successful':result.wasSuccessful(),"
        "'tests_executed':result.tests_executed()}))"
    )
    return [python, "-c", code], marker


def _verified(result: dict[str, Any], marker: str) -> tuple[bool | None, dict[str, Any] | None]:
    """An incomplete verifier has no outcome, even when its process exits zero."""
    if not result["ok"]:
        return None, None
    for line in result["output"].splitlines():
        if marker in line:
            try:
                verdict = json.loads(line.partition(marker)[2])
                return fixture_outcome(verdict), verdict
            except ValueError:
                return None, None
    return None, None


class LocalEnvironment:
    def __init__(self, task: Task, config: RunConfig, run_dir: Path):
        self.task, self.config = task, config
        self.provenance = {"kind": "local"}
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        source = Path(task.repo).resolve()
        if not source.is_dir():
            raise EnvironmentError(f"task repository does not exist: {source}")
        self.baseline = _files(source)
        self.workspace = self.run_dir / "workspace"
        if self.workspace.exists():
            raise EnvironmentError(f"refusing to overwrite an existing workspace: {self.workspace}")
        shutil.copytree(source, self.workspace, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        self._git_init()

    def _git_init(self) -> None:
        for command in (["git", "init", "-q"], ["git", "add", "--", "."],
                        ["git", "-c", "user.name=Budget Agent", "-c", "user.email=local@invalid", "commit", "-qm", "Fixture baseline"]):
            result = _process(command, self.workspace, 30)
            if not result["ok"]:
                raise EnvironmentError(f"could not initialize task repository: {result['output']}")

    def _editable(self, relative: str) -> bool:
        editable = self.task.metadata.get("editable")
        if editable is not None:
            return relative in editable
        parts = PurePosixPath(relative).parts
        return not any(p in {"tests", "test", ".git"} for p in parts) and not PurePosixPath(relative).name.startswith(("test_", "_private_"))

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        return execute_authorized(action, self.config, self.task, self._execute)

    def _execute(self, action: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            tool = action.get("tool")
            if tool in {"read_file", "list_files", "search", "write_file", "edit_file"}:
                text = _repository_tool(self.workspace, action, self.task.metadata.get("editable"))
            elif tool == "run":
                return _process(str(action["command"]), self.workspace, self.config.timeout_seconds, shell=True)
            elif tool == "public_test":
                return _process(self.task.test_command, self.workspace, self.config.timeout_seconds)
            elif tool == "diff":
                text = self.patch()
            else:
                raise EnvironmentError(f"unknown environment tool: {tool}")
            return {"output": text[:100_000], "ok": True, "duration_seconds": time.monotonic() - started}
        except PermissionError as exc:
            return {"output": str(exc), "ok": False, "policy_denial": "repository_boundary",
                    "duration_seconds": time.monotonic() - started}
        except (KeyError, ValueError, OSError, EnvironmentError) as exc:
            return {"output": str(exc), "ok": False, "duration_seconds": time.monotonic() - started}

    def revision(self) -> str:
        digest = hashlib.sha256()
        for name, data in _files(self.workspace).items():
            digest.update(name.encode("utf-8") + b"\0" + data + b"\0")
        return digest.hexdigest()[:16]

    def patch(self) -> str:
        current = _files(self.workspace)
        chunks = []
        for name in sorted(self.baseline.keys() | current.keys()):
            before, after = self.baseline.get(name), current.get(name)
            if before == after:
                continue
            chunks.extend(difflib.unified_diff(
                (before or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
                (after or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
                fromfile=f"a/{name}" if before is not None else "/dev/null",
                tofile=f"b/{name}" if after is not None else "/dev/null"))
        return "".join(chunks)

    def grade(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            current = _files(self.workspace)
            changed = {name for name in self.baseline.keys() | current.keys() if self.baseline.get(name) != current.get(name)}
            forbidden = [name for name in changed if not self._editable(name)]
            if forbidden:
                return {"resolved": False, "status": "invalid_patch", "output": f"Protected files changed: {', '.join(sorted(forbidden))}", "duration_seconds": time.monotonic() - started}
            verifier = Path(self.task.verifier).resolve()
            if not verifier.is_file():
                raise EnvironmentError("task has no independent verifier")
            with tempfile.TemporaryDirectory(prefix="grade-", dir=self.run_dir) as directory:
                grading = Path(directory)
                for name, data in self.baseline.items():
                    path = grading / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                # Apply only the accepted source delta to the pristine grading tree.
                for name in changed:
                    path = grading / name
                    if name in current:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(current[name])
                    else:
                        path.unlink(missing_ok=True)
                shutil.copyfile(verifier, grading / "_private_verifier.py")
                command, marker = _verification_command(sys.executable)
                result = _process(command, grading, self.config.timeout_seconds)
            resolved, verdict = _verified(result, marker)
            status = "grading_error" if resolved is None else "resolved" if resolved else "unresolved"
            return {"resolved": resolved, "status": status, "output": result["output"], "test_results": verdict, "duration_seconds": time.monotonic() - started}
        except (OSError, EnvironmentError) as exc:
            return {"resolved": None, "status": "infrastructure_error", "output": str(exc), "duration_seconds": time.monotonic() - started}

    def close(self) -> None:
        pass


class DockerEnvironment:
    """Container execution; task solutions are graded after the agent stops."""
    def __init__(self, task: Task, config: RunConfig, run_dir: Path):
        self.task, self.config = task, config
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.container = f"budget-agent-{uuid.uuid4().hex[:12]}"
        self.local: LocalEnvironment | None = None
        self.workdir = task.workdir if task.kind == "swebench" else "/workspace"
        image = task.image if task.kind == "swebench" else config.docker_image
        if not image:
            raise EnvironmentError("Docker image is required")
        if task.kind == "swebench":
            expected = str(task.metadata.get("base_commit", ""))
            if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected):
                raise EnvironmentError("SWE-bench base_commit must be a full commit SHA")
        command = ["docker", "run", "-d", "--name", self.container, "--network", "none", "--cpus", "2", "--memory", "4g", "--pids-limit", "256", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-w", self.workdir]
        if task.kind != "swebench":
            self.local = LocalEnvironment(task, config, self.run_dir)
            command += ["--mount", f"type=bind,source={self.local.workspace},target=/workspace"]
        command += [image, "sh", "-c", "while true; do sleep 3600; done"]
        result = _process(command, self.run_dir, max(60, config.timeout_seconds))
        if not result["ok"]:
            self.close()
            raise EnvironmentError(f"Docker startup failed; start the Linux engine and pre-pull the image. {result['output']}")
        identity = _process(["docker", "inspect", "--format", "{{.Image}}", self.container], self.run_dir, 15)
        image_id = identity["output"].strip()
        if not identity["ok"] or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            self.close()
            raise EnvironmentError("Cannot record the actual Docker image identity")
        self.provenance = {"kind": "docker", "requested_image": image, "image_id": image_id}
        if task.kind == "swebench":
            # The official grader applies patches to the image's setup checkout.
            # Keep that same source tree and exclude setup changes from our patch.
            revision = self._exec(["git", "rev-parse", "HEAD"])
            self.base_commit = revision["output"].strip()
            if not revision["ok"] or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.base_commit):
                self.close()
                raise EnvironmentError("SWE-bench image returned an invalid checkout commit")
            ancestry = self._exec(["git", "merge-base", "--is-ancestor", expected, self.base_commit])
            if not ancestry["ok"]:
                self.close()
                raise EnvironmentError("SWE-bench image checkout does not descend from task base_commit")
            self.provenance.update(task_base_commit=expected, image_checkout=self.base_commit)

    def _exec(self, command: list[str]) -> dict[str, Any]:
        result = _process(["docker", "exec", "-w", self.workdir, self.container, *command], self.run_dir, self.config.timeout_seconds)
        if result.get("timed_out"):
            # Killing docker exec alone does not terminate container processes.
            _process(["docker", "kill", self.container], self.run_dir, 15)
        return result

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        return execute_authorized(action, self.config, self.task, self._execute)

    def _execute(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.local is not None and action.get("tool") not in {"run", "public_test"}:
            return self.local._execute(action)
        tool = action.get("tool")
        if tool in {"run", "public_test"}:
            # Official SWE-bench images activate testbed through root's bash profile.
            shell = "bash" if self.task.kind == "swebench" else "sh"
            command = shlex.join(self.task.test_command) if tool == "public_test" else str(action["command"])
            return self._exec([shell, "-lc", command])
        if tool == "diff":
            return {"output": self.patch(), "ok": True, "duration_seconds": 0.0}
        # Inline file tools run inside the container, not on the host. The payload
        # travels as argv JSON rather than shell interpolation.
        code = _FILE_TOOL_SOURCE + '''
import json, sys
from pathlib import Path
try:
    print(_repository_tool(Path.cwd(), json.loads(sys.argv[1]), json.loads(sys.argv[2])))
except PermissionError as exc:
    print(str(exc))
    sys.exit(77)
except (KeyError, ValueError, OSError) as exc:
    print(str(exc))
    sys.exit(1)
'''
        result = self._exec(["python", "-c", code, json.dumps(action, ensure_ascii=False), json.dumps(self.task.metadata.get("editable"))])
        if result.get("returncode") == 77:
            result["policy_denial"] = "repository_boundary"
        return result

    def revision(self) -> str:
        if self.local is not None:
            return self.local.revision()
        # Hash inside the container so the bounded tool-output capture cannot
        # make two large working trees look identical. The immutable starting
        # commit also survives an agent running `git commit` during its work.
        code = r'''import hashlib,pathlib,subprocess,sys
h=hashlib.sha256()
diff=subprocess.run(['git','diff',sys.argv[1],'--binary'],check=True,stdout=subprocess.PIPE).stdout
h.update(diff)
files=subprocess.run(['git','ls-files','--others','--exclude-standard','-z'],check=True,stdout=subprocess.PIPE).stdout
for raw in sorted(files.split(b'\0')):
 if not raw: continue
 h.update(b'\0'+raw+b'\0'); p=pathlib.Path(raw.decode('utf-8',errors='surrogateescape'))
 if p.is_symlink(): h.update(str(p.readlink()).encode('utf-8',errors='surrogateescape')); continue
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
print(h.hexdigest())'''
        result = self._exec(["python", "-c", code, self.base_commit])
        if not result["ok"]:
            raise EnvironmentError("cannot compute repository revision")
        digest = result["output"].strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EnvironmentError("repository revision returned an invalid digest")
        return digest[:16]

    def patch(self) -> str:
        if self.local is not None:
            return self.local.patch()
        add = self._exec(["git", "add", "-N", "--", "."])
        if not add["ok"]:
            raise EnvironmentError("cannot include new files in repository patch")
        result = self._exec(["git", "diff", self.base_commit, "--binary"])
        if not result["ok"]:
            raise EnvironmentError("cannot export repository patch")
        if result.get("output_truncated"):
            raise EnvironmentError("repository patch exceeds the 100000-byte capture limit; refusing to export a truncated patch")
        return result["output"]

    def grade(self) -> dict[str, Any]:
        if self.task.kind == "swebench":
            return {"resolved": None, "status": "pending_official_evaluation", "output": "Export predictions and run the official SWE-bench harness.", "duration_seconds": 0.0}
        assert self.local is not None
        # Never execute model-modified code on the host in Docker mode.
        started = time.monotonic()
        current = _files(self.local.workspace)
        changed = {name for name in self.local.baseline.keys() | current.keys() if self.local.baseline.get(name) != current.get(name)}
        forbidden = [name for name in changed if not self.local._editable(name)]
        if forbidden:
            return {"resolved": False, "status": "invalid_patch", "output": f"Protected files changed: {forbidden}", "duration_seconds": time.monotonic() - started}
        with tempfile.TemporaryDirectory(prefix="grade-", dir=self.run_dir) as directory:
            grading = Path(directory)
            for name, data in self.local.baseline.items():
                path = grading / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            for name in changed:
                path = grading / name
                if name in current:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(current[name])
                else:
                    path.unlink(missing_ok=True)
            shutil.copyfile(self.task.verifier, grading / "_private_verifier.py")
            grader_name = f"budget-grade-{uuid.uuid4().hex[:12]}"
            verifier_command, marker = _verification_command("python")
            command = ["docker", "run", "--rm", "--name", grader_name, "--network", "none", "--cpus", "2", "--memory", "2g", "--pids-limit", "128", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--mount", f"type=bind,source={grading},target=/workspace,readonly", "-w", "/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1", self.config.docker_image, *verifier_command]
            result = _process(command, self.run_dir, self.config.timeout_seconds)
            if result.get("timed_out"):
                _process(["docker", "rm", "-f", grader_name], self.run_dir, 15)
        if result.get("returncode") in {125, 126, 127} and not result.get("timed_out"):
            return {"resolved": None, "status": "infrastructure_error", "output": result["output"], "test_results": None, "duration_seconds": time.monotonic() - started}
        resolved, verdict = _verified(result, marker)
        status = "grading_error" if resolved is None else "resolved" if resolved else "unresolved"
        return {"resolved": resolved, "status": status, "output": result["output"], "test_results": verdict, "duration_seconds": time.monotonic() - started}

    def close(self) -> None:
        _process(["docker", "rm", "-f", self.container], self.run_dir, 15)


def create_environment(task: Task, config: RunConfig, run_dir: Path) -> LocalEnvironment | DockerEnvironment:
    if config.environment == "docker":
        return DockerEnvironment(task, config, run_dir)
    if task.kind == "swebench":
        raise EnvironmentError("SWE-bench tasks require environment=docker")
    if config.provider != "scripted" and not config.allow_local_execution:
        raise EnvironmentError("Local execution is not sandboxed. Use Docker or explicitly set allow_local_execution=true.")
    return LocalEnvironment(task, config, run_dir)

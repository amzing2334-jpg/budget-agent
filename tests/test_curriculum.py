import json
from pathlib import Path

from budget_agent.environment import create_environment
from budget_agent.learning.curriculum import generate_curriculum
from budget_agent.schema import RunConfig, Task


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def resolve_task(row, root):
    return Task(**{**row, "repo": str(root / row["repo"]), "verifier": str(root / row["verifier"])})


def test_curriculum_partitions_and_teachers_have_no_evaluation_answers(tmp_path):
    paths = generate_curriculum(tmp_path, train_count=4, eval_count=4)
    train, evaluation, demonstrations = (read_rows(paths[name]) for name in ("train", "eval", "sft"))
    assert len(train) == len(evaluation) == len(demonstrations) == 4
    train_ids = {row["id"] for row in train}
    assert train_ids.isdisjoint(row["id"] for row in evaluation)
    assert {row["task_id"] for row in demonstrations} == train_ids
    assert {row["split"] for row in demonstrations} == {"train"}
    assert {(row["metadata"]["family"], row["metadata"]["parameter"]) for row in train}.isdisjoint(
        (row["metadata"]["family"], row["metadata"]["parameter"]) for row in evaluation)
    assert {row["metadata"]["family"] for row in train} == {"threshold", "range"}
    for row in train + evaluation:
        assert not Path(row["repo"]).is_absolute()
        assert row["metadata"]["editable"] == ["solution.py"]
        assert not (tmp_path / row["repo"] / "_private_verifier.py").exists()
        assert (tmp_path / row["verifier"]).read_text().count("def test_hidden_") == 5


def test_teacher_repairs_pass_actual_independent_verifiers(tmp_path):
    paths = generate_curriculum(tmp_path / "data", train_count=2, eval_count=2)
    tasks = {row["id"]: resolve_task(row, paths["train"].parent) for row in read_rows(paths["train"])}
    for demo in read_rows(paths["sft"]):
        task = tasks[demo["task_id"]]
        env = create_environment(task, RunConfig(), tmp_path / "runs" / task.id)
        try:
            before = (Path(task.repo) / "solution.py").read_text()
            assert env.grade()["resolved"] is False
            actions = [json.loads(message["content"]) for message in demo["messages"] if message["role"] == "assistant"]
            assert [action["tool"] for action in actions] == ["read_file", "write_file", "run", "finish"]
            for action in actions[:-1]:
                assert env.execute(action)["ok"]
            grade = env.grade()
            assert grade["resolved"] is True and grade["test_results"]["tests_run"] == 5
            assert (Path(task.repo) / "solution.py").read_text() == before
        finally:
            env.close()


def test_heldout_tasks_start_broken_and_are_repairable(tmp_path):
    paths = generate_curriculum(tmp_path / "data", train_count=1, eval_count=2)
    for row in read_rows(paths["eval"]):
        task = resolve_task(row, paths["eval"].parent)
        env = create_environment(task, RunConfig(), tmp_path / "runs" / task.id)
        try:
            assert env.grade()["resolved"] is False
            source = (Path(task.repo) / "solution.py").read_text()
            value = task.metadata["parameter"]
            fixed = (source.replace(" > ", " >= ") if task.metadata["family"] == "threshold"
                     else source.replace(f"range(start, {value})", f"range(start, {value + 1})"))
            env.execute({"tool": "write_file", "path": "solution.py", "content": fixed})
            assert env.grade()["resolved"] is True
        finally:
            env.close()


def test_generation_is_stable_when_other_split_grows(tmp_path):
    first = generate_curriculum(tmp_path / "small", train_count=1, eval_count=2)
    second = generate_curriculum(tmp_path / "large", train_count=8, eval_count=2)
    assert first["eval"].read_text() == second["eval"].read_text()

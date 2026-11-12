"""The optional learning CLI: curriculum generation needs no PyTorch; training commands validate first."""
import json

import pytest

from budget_agent.learning import cli


def test_curriculum_writes_disjoint_splits_and_reports_them(tmp_path, capsys):
    assert cli.main(["curriculum", "--output", str(tmp_path / "data"), "--train-count", "2", "--eval-count", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    rows = {name: [json.loads(line) for line in (tmp_path / "data" / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "eval")}
    assert len(rows["train"]) == 2 and len(rows["eval"]) == 1
    assert not {r["id"] for r in rows["train"]} & {r["id"] for r in rows["eval"]}
    assert report and all(str(tmp_path) in str(value) for value in report.values() if isinstance(value, str))


@pytest.mark.parametrize("argv", [
    [],                                         # a subcommand is required
    ["sft", "--output", "out"],                 # sft needs --data
    ["train", "--output", "out"],               # train needs --tasks
    ["evaluate", "--tasks", "t.jsonl"],         # evaluate needs --output
])
def test_incomplete_commands_are_rejected_before_any_model_loads(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)

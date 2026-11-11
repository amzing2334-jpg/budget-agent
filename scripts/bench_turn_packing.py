"""How many tokens prefix packing saves on real GRPO rollouts (CPU only).

For every attempt in the given step pools, renders each agent turn with the
policy's chat template, keeps the turns whose prompt digest matches what the
sampler saw, and counts the tokens a trainer encodes with one forward per turn
versus one forward per prefix chain (grpo_harness.py --pack-turns). Both modes
train exactly the same completion tokens.

Each chain break is classified: the next prompt still extends the previous
prompt but the chat template re-tokenized the agent's reply differently from
how it was sampled, or the context policy rewrote the history.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from budget_agent.learning.grpo_core import chain_layout, extract_turns, pack_turns
from budget_agent.models import token_digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="the policy's model directory")
    parser.add_argument("pools", nargs="+", type=Path, help="GRPO step pools (step-NNNN directories)")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    totals: Counter = Counter()
    chain_lengths: Counter = Counter()
    for pool in args.pools:
        for line in (pool / "attempts.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompts, completions = [], []
            for turn in extract_turns(Path(row["result_path"]).parent):
                prompt = tokenizer.apply_chat_template(turn.messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=False)
                if token_digest(prompt) != turn.prompt_digest:
                    totals["prompt_mismatch"] += 1
                elif turn.completion_ids:
                    prompts.append(prompt)
                    completions.append(turn.completion_ids)
            chains = pack_turns(prompts, completions)
            chain_lengths.update(len(chain) for chain in chains)
            for previous, current in zip(chains, chains[1:]):  # why each chain ended
                before, after = prompts[previous[-1]], prompts[current[0]]
                totals["break_retokenized_reply" if after[:len(before)] == before else "break_history_rewritten"] += 1
            totals["attempts"] += 1
            totals["turns"] += len(prompts)
            totals["sequences"] += len(chains)
            totals["completion_tokens"] += sum(len(c) for c in completions)
            totals["per_turn_tokens"] += sum(len(p) + len(c) - 1 for p, c in zip(prompts, completions))
            totals["packed_tokens"] += sum(len(chain_layout(prompts, completions, chain)[0]) for chain in chains)
    report = {**totals, "turns_per_sequence": round(totals["turns"] / max(1, totals["sequences"]), 2),
              "token_reduction": round(totals["per_turn_tokens"] / max(1, totals["packed_tokens"]), 2),
              "chain_length_counts": dict(sorted(chain_lengths.items()))}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

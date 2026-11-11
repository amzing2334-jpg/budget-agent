"""vLLM counter summaries used by the rollout benchmark and the round-4 A/B."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("vllm_metrics", Path(__file__).parents[1] / "scripts/vllm_metrics.py")
vllm_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vllm_metrics)


def test_summary_reports_interval_tokens_and_prefix_hit_rate():
    before = {"vllm:prompt_tokens": 10, "vllm:prefix_cache_queries": 100, "vllm:prefix_cache_hits": 40}
    after = {"vllm:prompt_tokens": 110, "vllm:generation_tokens": 5, "vllm:prefix_cache_queries": 300,
             "vllm:prefix_cache_hits": 190}
    assert vllm_metrics.summarize(before, after) == {
        "prompt_tokens": 100, "generation_tokens": 5, "prefix_cache_queries": 200, "prefix_cache_hits": 150,
        "prefix_cache_hit_rate": 0.75}


def test_summary_without_prefix_queries_has_no_hit_rate():
    assert vllm_metrics.summarize({}, {"vllm:prompt_tokens": 3})["prefix_cache_hit_rate"] is None

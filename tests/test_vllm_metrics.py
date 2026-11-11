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


def test_zero_hits_give_a_zero_rate_on_current_counter_names():
    # vLLM 0.33 exports only the unprefixed names. A falsy fallback to the old gpu_* names
    # used to turn zero hits into None and crash the benchmark on None / queries.
    before = {"vllm:prefix_cache_queries": 50, "vllm:prefix_cache_hits": 20}
    after = {"vllm:prefix_cache_queries": 80, "vllm:prefix_cache_hits": 20}
    summary = vllm_metrics.summarize(before, after)
    assert summary["prefix_cache_hits"] == 0 and summary["prefix_cache_hit_rate"] == 0.0


def test_summary_reads_the_older_gpu_counter_names():
    after = {"vllm:gpu_prefix_cache_queries": 10, "vllm:gpu_prefix_cache_hits": 4}
    assert vllm_metrics.summarize({}, after)["prefix_cache_hit_rate"] == 0.4


def test_summary_without_prefix_queries_has_no_hit_rate():
    assert vllm_metrics.summarize({}, {"vllm:prompt_tokens": 3})["prefix_cache_hit_rate"] is None

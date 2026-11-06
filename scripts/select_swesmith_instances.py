"""Pre-registered SWE-smith instance selection for the RFT experiments.

Fixed candidate repositories, a seeded train/eval split by repository, and
per-repository filters (problem statement present, single-file patch,
1 <= FAIL_TO_PASS <= 20, PASS_TO_PASS <= 2000), at most PER_REPO instances.

Usage: python scripts/select_swesmith_instances.py <dir with SWE-smith parquet shards> <out.jsonl> <selection.json>
Shards: https://huggingface.co/datasets/SWE-bench/SWE-smith (data/train-*.parquet). Needs `.[swesmith]`.
"""
import glob, json, random, sys
import pandas as pd
CANDIDATES = ["tkrajina__gpxpy.09fc46b3", "Suor__funcy.207a7810", "msiemens__tinydb.10644a0e",
              "alecthomas__voluptuous.a7a55f83", "python-hyper__h11.bed0dd4a", "gweis__isodate.17cb25eb",
              "keleshev__schema.24a30457", "google__textfsm.c31b6007", "jd__tenacity.0d40e76f",
              "theskumar__python-dotenv.2b8635b7", "r1chardj0n3s__parse.30da9e4f", "hukkin__tomli.443a0c1b"]
PER_REPO = 24
rng = random.Random(20261105)
repos = CANDIDATES[:]; rng.shuffle(repos)
split = {repo: ("train" if i < 8 else "eval") for i, repo in enumerate(repos)}
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(sys.argv[1] + "/*.parquet"))])
df["name"] = df.repo.str.removeprefix("swesmith/")
df = df[df.name.isin(CANDIDATES)].copy()
df["n_f2p"] = df.FAIL_TO_PASS.apply(len); df["n_p2p"] = df.PASS_TO_PASS.apply(len)
df["n_files"] = df.patch.str.count(r"(?m)^diff --git ")
keep = df[(df.problem_statement.fillna("").str.strip().str.len() > 0) & (df.n_files == 1)
          & df.n_f2p.between(1, 20) & (df.n_p2p <= 2000)]
rows = []
for name in CANDIDATES:
    pool = keep[keep.name == name].sort_values("instance_id")
    picked = pool.sample(n=min(PER_REPO, len(pool)), random_state=20261105) if len(pool) else pool
    for r in picked.itertuples():
        rows.append({"instance_id": r.instance_id, "repo": name, "split": split[name], "patch": r.patch,
                     "FAIL_TO_PASS": list(r.FAIL_TO_PASS), "PASS_TO_PASS": list(r.PASS_TO_PASS),
                     "problem_statement": r.problem_statement})
with open(sys.argv[2], "w") as out:
    for row in rows: out.write(json.dumps(row) + "\n")
json.dump({"split": split, "per_repo": PER_REPO, "filters": "problem_statement nonempty; 1 file; 1<=F2P<=20; P2P<=2000",
           "counts": {n: sum(r["repo"] == n for r in rows) for n in CANDIDATES}}, open(sys.argv[3], "w"), indent=1)
print(json.load(open(sys.argv[3])))

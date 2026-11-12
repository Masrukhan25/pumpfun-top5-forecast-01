"""pumpfun-top5-forecast-01 entry (Python 3.13, stdlib only, deterministic).

Monte Carlo forecaster: simulates each candidate coin's future log market cap
(and phantom newcomer coins) to the answer reads, ranks them, and converts the
simulated frequencies into honest probabilities per place, plus size-bin
probabilities per place. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json).
"""
import json
import math
import os
import random

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tunable parameters ----
def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default

SEED = 20261112
N_SIMS = int(os.environ.get("PF_N_SIMS", "4000"))
DRIFT_DAMP = _f("PF_DRIFT_DAMP", 0.10)
DRIFT_CLIP = (_f("PF_DRIFT_LO", -0.6), _f("PF_DRIFT_HI", 0.5))
SIGMA_BASE = _f("PF_SIGMA_BASE", 0.08)
SIGMA_PER_HOUR = _f("PF_SIGMA_PER_HOUR", 0.10)
NEW_MEAN = _f("PF_NEW_MEAN", 3.25)
NEW_SD = _f("PF_NEW_SD", 0.15)
NEW_BASE = _f("PF_NEW_BASE", 1.0)
NEW_PER_HOUR = _f("PF_NEW_PER_HOUR", 2.75)
NEW_MAX = int(os.environ.get("PF_NEW_MAX", "8"))
TOP_CANDIDATES = int(os.environ.get("PF_TOP_CAND", "30"))
# candidates: top N coins by cap at the last mark
# probability floors / caps
P_MIN = 0.002             # drop named rows below this (keep file small, avoid noise)
SIZE_P_MIN = 0.001


def load_input():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                if m.get("ok") and isinstance(m.get("rows"), list):
                    marks.append(m)
    if not marks:
        raise ValueError("no ok marks in history")
    return task, marks


def parse_time(s):
    # ISO 8601 UTC -> epoch seconds (stdlib only)
    from datetime import datetime, timezone
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def logcap_sol(row):
    return math.log10(int(row["market_cap_lamports"])) - 9


def main():
    task, marks = load_input()
    places = task["places"]
    edges = task["size"]["edges"]
    out_name = task.get("output", "forecast.json")
    reads = task.get("reads", [])
    last = marks[-1]
    last_t = parse_time(last["at"])
    # horizons (hours) from last mark to each answer read
    horizons = []
    for r in reads:
        h = (parse_time(r) - last_t) / 3600.0
        if h < 0:
            h = 0.0
        horizons.append(h)
    if not horizons:
        horizons = [1.0]
    h_mean = sum(horizons) / len(horizons)

    # previous marks for momentum (by mint)
    prev = {}
    for m in marks[:-1]:
        for r in m["rows"]:
            prev[r["mint"]] = r
    prev2 = {}
    if len(marks) >= 3:
        for m in marks[:-2]:
            for r in m["rows"]:
                prev2[r["mint"]] = r

    # candidate coins: top TOP_CANDIDATES by cap at last mark
    rows = sorted(last["rows"], key=lambda x: x["rank"])[:TOP_CANDIDATES]
    cands = []
    for r in rows:
        lc = logcap_sol(r)
        age_min = (last_t - parse_time(r["created_at"])) / 60.0
        # recent hourly log drift from last 10-min step (and 20-min if available)
        drift = 0.0
        p = prev.get(r["mint"])
        if p is not None:
            g10 = logcap_sol(r) - logcap_sol(p)  # per 10 min
            drift = g10 * 6.0  # per hour
            p2 = prev2.get(r["mint"])
            if p2 is not None:
                g20 = (logcap_sol(r) - logcap_sol(p2)) / 2.0
                drift = 0.5 * drift + 0.5 * (g20 * 6.0)
        # venue/age prior: young pool coins in pump phase drift up; curve coins flat/down;
        # old coins (past peak) fade
        prior = 0.0
        if r.get("venue") == "pool" and 10 <= age_min <= 40:
            prior = 0.15
        elif r.get("venue") == "curve":
            prior = -0.10
        if age_min > 40:
            prior -= _f("PF_FADE_OLD", 0.10)
        drift = DRIFT_DAMP * drift + prior
        drift = max(DRIFT_CLIP[0], min(DRIFT_CLIP[1], drift))
        cands.append({
            "mint": r["mint"], "logcap": lc, "drift": drift,
            "age": age_min, "venue": r.get("venue", ""),
        })

    _new_mean = NEW_MEAN
    rng = random.Random(SEED)
    n_phantom = int(round(NEW_BASE + NEW_PER_HOUR * h_mean))
    n_phantom = max(0, min(NEW_MAX, n_phantom))
    sigma = SIGMA_BASE + SIGMA_PER_HOUR * math.sqrt(max(h_mean, 0.0))

    # Monte Carlo over the 7 horizons (cycle through them per sim)
    place_counts = {k: {} for k in range(1, places + 1)}
    place_phantom = {k: 0 for k in range(1, places + 1)}
    size_counts = {k: {} for k in range(1, places + 1)}
    top5_counts = {}
    for s in range(N_SIMS):
        h = horizons[s % len(horizons)]
        sig = SIGMA_BASE + SIGMA_PER_HOUR * math.sqrt(max(h, 0.0))
        sampled = []
        for c in cands:
            mu = c["logcap"] + c["drift"] * h
            sampled.append((rng.gauss(mu, sig), c["mint"]))
        for _ in range(n_phantom):
            sampled.append((rng.gauss(_new_mean, NEW_SD), None))
        sampled.sort(key=lambda x: -x[0])
        for k in range(1, places + 1):
            if k - 1 >= len(sampled):
                break
            v, mint = sampled[k - 1]
            if mint is None:
                place_phantom[k] += 1
            else:
                place_counts[k][mint] = place_counts[k].get(mint, 0) + 1
                top5_counts[mint] = top5_counts.get(mint, 0) + 1
            # size bin of the cap at this place
            b = bin_of(v, edges)
            size_counts[k][b] = size_counts[k].get(b, 0) + 1

    probabilities = []
    for k in range(1, places + 1):
        for mint, cnt in sorted(place_counts[k].items(), key=lambda x: -x[1]):
            p = cnt / N_SIMS
            if p >= P_MIN:
                probabilities.append({"mint": mint, "place": k, "p": round(p, 6)})
    # top5: the 5 coins most often in the top 5, ordered by total top5 frequency,
    # tie-break by best (lowest) median place -- use total count then last-mark rank
    rank_of = {c["mint"]: i for i, c in enumerate(cands)}
    top5 = sorted(top5_counts.items(), key=lambda x: (-x[1], rank_of.get(x[0], 999)))[:places]
    top5 = [m for m, _ in top5]
    sizes = []
    for k in range(1, places + 1):
        for b, cnt in sorted(size_counts[k].items()):
            p = cnt / N_SIMS
            if p >= SIZE_P_MIN:
                sizes.append({"place": k, "bin": b, "p": round(p, 6)})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(out) + "\n")


def bin_of(s, edges):
    c = sum(1 for e in edges if e <= s)
    return min(c, 81)


if __name__ == "__main__":
    main()

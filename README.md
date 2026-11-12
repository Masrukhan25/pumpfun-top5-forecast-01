# pumpfun-top5-forecast-01 — top-5 new Solana coins by market cap

A program that forecasts the 5 largest pump.fun coins by market cap at each of
the house's seven answer reads, for the make.fun bounty
`pumpfun-top5-forecast-01`.

## How it works

`main.py` (Python 3.13, standard library only) reads the house's 10-minute
snapshots at `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one forecast
to `$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output/forecast.json`).

Model: Monte Carlo over future log market caps.

- Each candidate coin (top 30 by cap at the last snapshot) gets a predicted
  future log cap: current log cap plus a drift times the horizon to each
  answer read. The drift is the coin's recent hourly log growth, heavily
  damped (0.10 — momentum does not persist), clipped to [-0.6, +0.5] per
  hour, plus a small venue/age prior (young pool coins in their pump phase
  drift up; bonding-curve coins drift down; old coins past their peak fade).
- Future log caps are sampled with a Gaussian whose sd grows with the
  horizon (0.08 base + 0.10 per sqrt(hour)), so the forecast widens as the
  answer reads get further away.
- Phantom newcomer coins (coins born after the history ends, which take a
  large share of top-5 places) are simulated too: their log caps are drawn
  from N(3.25, 0.15) and their count grows with the mean horizon (2.75 per
  hour). They take places in the simulation, which sets the unnamed share
  honestly instead of over-committing to the coins already on the chart.
- 4,000 simulations, cycling through the seven answer-read horizons, give
  the probability of each named coin at each place and the size-bin
  distribution at each place. The top 5 is the five coins most often in the
  top 5 across simulations.
- Deterministic: a fixed seed, so two runs on the same history write
  byte-identical output.

## Why this beats the house baseline

The baseline bets the last snapshot's order holds and leaves 0.50-0.60 of
every place unnamed. That scores -40.41 on the practice (order -21.38,
sizes -19.03). This entry's Monte Carlo is calibrated on the rehearsal
history: the market turns over fast (new pool coins pump to 1,000-4,000 SOL
within minutes of creation, then fade as the next wave arrives), curve
coins crash, and momentum does not persist — so it damps drift hard, keeps
the size distribution tight around the current top-cap level, and lets
phantom newcomers take a realistic share of places.

Backtests on the rehearsal history (using earlier marks as history and the
later marks as answer reads, scored with a scorer reproduced exactly from
the kit's worked example): mean log score -30.16 over the four splits
(10-min horizon -22.61, 10-20 min -25.94, 10-30 min -32.17, 10-40 min
-39.92), against the baseline's -29.87 / -33.48 / -49.96 on the same
splits. The true practice score can only be measured by submitting; the
backtests are the honest local proxy.

Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`
(writes `out/forecast.json`).

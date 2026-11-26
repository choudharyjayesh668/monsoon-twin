# Monsoon Twin: brief

## Page 1: model design, drift and blur, failure modes

**What it is.** A digital twin of India's daily weather on the 32 × 36 grid. From the last 3 days of temperature, rain, wind, pressure and moisture, it rolls the state forward one day at a time. A shock added to the last day, such as a warmer Bay of Bengal, runs as a second "universe" next to the first.

**Design.**
- **State:** the anomaly from a 366-day climatology, divided by a per-channel scale. The twin predicts each day's *change* of that anomaly from the last 3 days, the target day's climatology, land–sea mask, orography, latitude and longitude, and the season.
- **Network:** a small convolutional net: a 3×3 stem, five residual 3×3 blocks and a 1×1 head, about 1 M weights. It has no pooling, attention, downsampling or map-wide normalisation, so information moves **at most 6 cells per day**. A shock therefore starts local and spreads physically, never jumping across India in one step. A test enforces the limit.
- **Training:**
  - The loss is matched to the score: half squared error scaled by the better reference (persistence or climatology), half each window's error over its own better reference.
  - Training first fits one step, then **unrolls the twin through 4 days**, so it learns from its own output how far to relax towards climatology.
  - bfloat16 on the Apple GPU with MLX (open source); about 18 minutes per run.
- **Data splits:** develop on 2005–2017, tune on 2018–2019, report once on 2020–2021, ship the same recipe retrained on 2005–2021.
- **Heavy rain:** the emitted rain is lifted to just above 10 mm/day where the expected rain passes a per-day cutoff (8.5 mm on day 1, about 6.5 mm by day 8). The decision is per cell, and the state fed back is never lifted.

**How drift and blur are prevented.**
- **Drift:** the twin works in anomalies, so the climatology anchors every step. Unrolled training penalises errors at days 1 to 4, which teaches the twin to shrink anomalies at the rate the data supports. Rain is kept ≥ 0 inside the loop. Output guards (rain ≥ 0, −40 to +60 °C, finite values) are a last line of defence.
- **Blur:** a mean-error model tends to smooth fields out. Here, day-8 spatial variance stays at 0.83 to 1.02 of the truth's for every field, so blur costs nothing.

**Results (2020–2021, held out).** Skill against the better of persistence and climatology:
- **+0.25 at day 1, +0.004 at day 4, −0.11 at day 8**, per window and lead.
- Beats persistence at day 4 by +0.225.
- Heavy-rain skill is positive at all three days.
- Every one of the five hidden-test-like event types (heavy rain, cyclone, heatwave, winter disturbance, monsoon onset) scores positive skill.
- The null shock changes nothing (bit-identical), and every rollout stays valid for 14 days.

**Failure modes we know about.**
1. **Rain under moisture shocks.** With +15% moisture over the Bay of Bengal, single heavy-rain cells can move or strengthen by more than 5 std (38 mm/day), in about 26% of windows.
2. **Weak wind shocks fade.** A 40% weaker Arabian Sea wind spreads far enough by days 4–8 in only about half the windows; the twin damps it.
3. **Day 8 is close to climatology.** Beyond day 4, skill comes from relaxing well rather than from predicting new weather. At day 8 the twin is slightly worse than climatology.
4. **Leap years.** Requests carry no year, so from March in a leap year the climatology is one day late.
5. **The domain edge.** Weather entering from outside the 32 × 36 box cannot be seen. Forecasts near the edges degrade first.
6. **Overfitting.** On years it trained on, the twin scores far higher (0.17 vs 0.05). Stronger regularisation is the clearest route to more skill.

## Page 2: visualisation and impact use case

To be written after the Stage 2 work.

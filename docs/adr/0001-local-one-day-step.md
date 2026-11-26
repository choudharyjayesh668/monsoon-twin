# 1. The twin is a local network stepped one day at a time

Status: accepted, 2026-11-26

## Context

The benchmark rewards forecast skill at days 1, 4 and 8, and separately checks that a shock stays near its box on day 1, spreads no faster than 8 cells a day, and never grows unstable. The brief asks for a simulator that rolls the weather forward day by day.

Measured alternatives (2020–2021, the kit's scoring functions):

- A linear model on global map patterns reached 0.093 pooled skill, but every output cell depends on every input cell, so a shock would reach all of India in one step.
- One network per lead, predicting each day directly, matched a stepped model at day 4 in the design review. It is not a simulator, and a shock would not move step by step.
- Attention or a deep U-Net would see the whole map at once, which breaks shock locality.

## Decision

One small convolutional network predicts the next day's anomaly change from the last three days. It has only 3×3 convolutions (one stem and five residual blocks) and no pooling, downsampling, attention or map-wide normalisation. Information therefore moves at most 6 cells per day, and a test enforces this. The network is trained by unrolling it through 4 days, so it learns how far to relax towards climatology from its own output. An 8-day unroll was started but not finished within the time budget.

## Consequences

- Shock locality and the speed limit hold by construction, not by training luck.
- Inference is fast: milliseconds per step on 4 CPU cores.
- Influence from beyond the domain edge cannot enter. The twin sees only the 32 × 36 box.
- Whether a shock spreads far enough by day 8 depends on what the network learns. It is measured, not guaranteed.

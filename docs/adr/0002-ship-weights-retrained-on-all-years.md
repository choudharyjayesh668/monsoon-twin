# 2. Shipped weights are retrained on all years after the recipe is frozen

Status: accepted, 2026-11-26

## Context

The data covers 2005–2021; the hidden test is 2023–24. Development needs untouched years to measure honestly, but the shipped model gains from the most recent years.

## Decision

- Develop with three splits: train on 2005–2017, tune on 2018–2019, and report once on 2020–2021 after the recipe is frozen.
- Then retrain the frozen recipe (same settings, iterations and seed) on 2005–2021, and ship those weights.
- The heavy-rain cutoffs are not re-tuned: they are copied from the development model.

## Consequences

- The reported numbers belong to the development model with 2020–2021 held out, not to the shipped weights, and the brief says so.
- The shipped model's internal climatology is exactly the grader's, which covers all of 2005–2021.

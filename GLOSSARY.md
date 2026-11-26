# Glossary

**Twin**: our simulator of India's daily weather on the 32 × 36 grid. Given three days, it rolls the weather forward one day at a time.

**State**: one day of weather: six fields (temperature, rain, two wind components, sea-level pressure, water vapour) on the grid, in physical units.

**Window**: one forecast case. Three consecutive input days, plus the days that follow, which the forecast is judged against. The hidden test has five windows.

**Step**: one day of simulation. A rollout of 10 steps gives 10 days.

**Lead (day k)**: how many days after the last input day a forecast is valid. The score looks at days 1, 4 and 8.

**Persistence**: the forecast "every day looks like the last input day" (also "repeat yesterday").

**Climatology**: the long-term average weather for each calendar day: a 366-day mean over 2005–2021, smoothed over 15 days.

**Anomaly**: a state minus the climatology for its calendar day. The twin thinks in anomalies.

**Better reference**: whichever of persistence and climatology is closer to the truth. Skill is measured against it: skill = 1 − error / error of the better reference. Positive skill beats both.

**Reading A, B, C**: three ways the grader might average skill. A: per window, days 1, 4 and 8 pooled. B: per window, each day on its own. C: all windows pooled. We report all three, because the grader's choice is not published.

**Heavy rain**: more than 10 mm in a day, in one grid cell. Heavy-rain skill is measured with CSI (hits over hits, misses and false alarms).

**Heavy-rain lift**: raising the twin's emitted rain to just above 10 mm/day where its expected rain passes a cutoff for that step. It changes what the twin reports, never what it simulates.

**Shock (perturbation)**: a change added to the last input day, such as a warmer Bay of Bengal. The difference between the shocked and the unshocked rollout is the shock's response.

**Null shock**: a shock of all zeros. It must change nothing.

**Local, spreads, stable**: the three properties a shock's response must have: on day 1 it stays near the shocked box; by days 4–8 it has moved outward, no faster than 8 cells a day; and it never grows beyond 5 standard deviations.

**Blur**: a forecast that has collapsed towards a smooth average. Measured as the spatial variance of the day-8 field over that of the truth.

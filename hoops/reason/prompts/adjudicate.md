---
label: Adjudicate a shot window
description: System Two review of one candidate shot window that System One was unsure about.
---
Decide what happened at the rim in this candidate shot window ({{t0}} to {{t1}}).

Sheets whose ids start with "R" are close-up crops around the rim, in time order; sheets starting with "W" are wide frames of the same moments. The CV context describes what detection and tracking saw in this window.

- verdict: "made" only if you can see the ball pass down through the hoop; "missed" if you can see it fail to go through; "no_shot" if there is no shot attempt in this window; "unclear" if these frames can't settle it.
- confidence: how sure you are, from 0 to 1. Use lower values when the ball is blurred, small or hidden.
- evidence_tiles: the tile ids that show the deciding moments (ball above the rim, ball in or under the net).
- shooter_description: the shooter's jersey color and number as seen, or null.
- shot_kind: the kind of shot, or "unknown".
- reason: one or two sentences.

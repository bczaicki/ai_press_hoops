# Evaluation: is the shot detector any good on *your* footage?

Accuracy depends heavily on how a game was filmed, so we measure it on a few clips you label by hand.

## 1. Label a clip
Create `eval/labels/<anything>.json`. Use the `clip_id` that the app shows for the uploaded clip, and the times as they appear in the app's player:

```json
{
  "clip_id": "3f9c0d2e8a1b4c5d",
  "footage_type": "fixed_camera",
  "events": [
    {"t_s": 4.2, "type": "made_shot", "note": "layup"},
    {"t_s": 17.8, "type": "missed_shot"},
    {"t_s": 41.0, "type": "shot_attempt", "note": "outcome hidden"}
  ]
}
```

- `t_s` is the release moment (when the ball leaves the hand), in seconds. Within about half a second is fine.
- `type` is one of:
  - `made_shot`
  - `missed_shot`
  - `shot_attempt`: you saw a shot but not whether it went in.
- `footage_type` is one of `fixed_camera`, `broadcast`, `handheld`, `highlights`, or `practice`. Results are reported separately for each type.
- `note` is optional.

The Review tab can also write this file for you: 👍/👎 and "add missed shot" save a label set.

**What counts as a shot.** Decide these before labeling and stay consistent:
- Free throws, tip-ins and put-backs each count as a separate attempt.
- A blocked shot is an attempt (a miss).
- Passes that hit the rim are not shots.
- Shots in replays (broadcast) are not labeled again.

## 2. Score
```bash
uv run hoops eval              # scores the newest finished run of each labeled clip
```
Predictions match labels one-to-one when they fall within **±1 s** of each other.

## 3. Read the report
| Term | Meaning |
|---|---|
| Precision | Of the shots the app reported, how many were real. Low precision means false alarms ("phantom makes"). |
| Recall | Of the real shots, how many the app found. Low recall means missed shots. |
| 95% CI | The plausible range given how few shots were labeled, using Wilson intervals. With 30 labeled shots, expect ±10–15 points. Don't trust a difference smaller than the CI. |
| Made/missed right | On shots that were found, how often the outcome call agreed with your label. |
| Confidence AUROC | Do high-confidence calls tend to be right? 0.5 means confidence is noise, so sending "unsure" calls to an AI reviewer can't help. 0.8 or higher means the review queue is worth it. |

The **Strict ↔ Eager** slider trades precision for recall. `hoops.eval.sweep()` shows that trade-off at each threshold, without re-running the models.

Aim for **3–5 clips per footage type and 30 or more shots in total** before drawing conclusions.

`example.json` is only a format sample. Its `clip_id` matches nothing, so the scorer skips it.

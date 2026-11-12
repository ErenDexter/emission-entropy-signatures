# Results behind the paper

Outputs of the runs reported in Section 2.5 of the paper: 138 banknotes, each enrolled from a first-angle photo and tested with a second photo taken from another angle and under different lighting.

| File | Made by | Contents |
|---|---|---|
| `evaluation_imagenet/summary.json`, `results.csv` | `ees evaluate` with the untrained ImageNet network | End-to-end test: identification, claimed-ID check, and unregistered banknotes (half of the banknotes removed from the database). |
| `evaluation_trained/summary.json`, `results.csv` | `ees --weights models/ees_embed.pt evaluate` | The same test with the trained network. Verdicts, matched banknotes and same-note inliers are identical to the ImageNet run; times differ. |
| `all_pairs_inliers.csv` | `ees evaluate --all-pairs` | Every test photo matched droplet by droplet against every enrolled banknote: 138 × 138 = 19,044 pairs, 138 of them the same banknote. |
| `training/train_report.json` | `ees train` | Five-fold cross-validation on banknotes not seen in training (ImageNet versus trained network), the projected shortlist sizes, and all training settings. |
| `training_proj32/train_report.json` | `ees train --proj-dim 32` | The same with a smaller embedding (320 numbers): 97.8% ranked first. Not used in the paper. |

## Where the paper's numbers come from

| Paper statement | Value | Source |
|---|---|---|
| Test photos matched to the right banknote | 138 of 138 | `evaluation_*/summary.json`, `A_identification` |
| Same-banknote inliers | 42 to 990, median 236 | `all_pairs_inliers.csv`, rows with `same_note = 1` |
| Best score among different-banknote pairs | 10 (of 18,906 pairs) | `all_pairs_inliers.csv`, rows with `same_note = 0` |
| Accept threshold | 20 inliers | `summary.json`, `scores.accept_inliers` |
| Removed (unregistered) banknotes accepted | 0 of 69 | `summary.json`, `C_unenrolled.false_accepts` |
| Registered banknotes rejected in that test | 0 of 69 | `summary.json`, `C_unenrolled.false_rejects` |
| Verification time per photo | 0.7 s (RTX 3060 GPU, includes HEIC decoding) | `evaluation_trained/summary.json`, `timing_s.verify_mean` |
| Right banknote ranked first, unseen banknotes | 96.4% (ImageNet), 100% (trained) | `training/train_report.json`, `top1` |
| Separation d′ | 3.14 (ImageNet), 6.05 (trained) | `training/train_report.json`, `d_prime` |
| Shortlist for 99% recall at 1,000,000 banknotes | 142,273 (ImageNet), 2,274 (trained) | `training/train_report.json`, `shortlist_needed` |
| Embedding length | 10,240 (ImageNet), 1,280 (trained) | `training/train_report.json`, `dim` |

The shortlist sizes for large databases are projections from the measured score distributions (`ees/scaling.py`), not measurements.

## Columns

`results.csv`: `query` (test photo), `status` (verdict), `note_id` (matched banknote), `inliers` (agreeing droplets), `cosine` (embedding similarity to the matched banknote), `second_inliers` (best wrong banknote in the 20-candidate shortlist), `message`.

`all_pairs_inliers.csv`: `test_photo`, `enrolled_note`, `inliers`, `same_note` (1 if the test photo shows that banknote).

The pairing of test photos to banknotes was recovered by the system, since the two photo folders do not share file names. Every pair is the best match in both directions by a wide margin.

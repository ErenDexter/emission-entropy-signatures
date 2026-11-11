# Trained embedding

`pes_embed.pt` is the network used for the paper's results (Section 2.5, Step 4).

- ResNet-50 pre-trained on ImageNet, kept up to its third stage, with a 1 × 1 convolution to 128 channels per region and GeM pooling over the window and a 3 × 3 grid. The embedding has 1,280 numbers.
- Fine-tuned with the InfoNCE loss on all 138 banknotes (300 steps, 24 banknotes × 2 views per step). The full training settings are stored in the file under `meta` and in `results/training/train_report.json`.
- SHA-256: `5f31b73535d2b50a65b28420982b38d1b90ada1ee79ea5e3452d3c4420815ef7`
- Feature version written to enrolled records: `BDT-2-2023-front:0fba0cff074f4652`

Use it with `--weights models/pes_embed.pt`. A database holds records of one embedding model only, so enrol and verify with the same `--weights` setting.

Because this model saw all 138 banknotes during training, the numbers to quote for unseen banknotes are the five-fold cross-validation results in `results/training/train_report.json`. The end-to-end results in `results/evaluation_trained/` give the same verdicts, matched banknotes and same-note inlier counts as the untrained ImageNet network (`results/evaluation_imagenet/`), so they do not depend on the training photos.

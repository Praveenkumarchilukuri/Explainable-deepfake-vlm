# Saved experiment results

Transcribed from saved outputs in the uploaded notebook; not independently rerun during packaging.

| Level | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Frame (450 frames) | 92.89% | 87.39% | 99.05% | 92.86% | 99.82% |
| Video (15 videos) | 93.33% | 87.50% | 100.00% | 93.33% | 100.00% |

Video-level confusion matrix (actual rows, predicted columns): `[[7, 1], [0, 7]]` for labels `[real, fake]`.

The video-level test set contains only 15 videos (8 real, 7 fake), so these results are uncertain and do not demonstrate broad generalization. Training accuracy reached 99.02%, while best recorded validation accuracy was 74.89%; this suggests possible overfitting.

# test file to visualise the plotting and metrics code working together

import torch
from utils.metrics import evaluate_fold
from utils.plotting import plot_folds

num_test_samples = 20
num_roi = 10

pred = torch.randn(num_test_samples, num_roi, num_roi).numpy()
gt   = torch.randn(num_test_samples, num_roi, num_roi).numpy()

folds = [
    (pred[:7], gt[:7]),
    (pred[7:14], gt[7:14]),
    (pred[14:], gt[14:])
]

fold_results = [evaluate_fold(p, g) for p, g in folds]
plot_folds(fold_results)
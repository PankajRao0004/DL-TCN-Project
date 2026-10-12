# Temporal Convolutional Networks (TCN)

An independent PyTorch implementation and reduced-scale evaluation of Temporal Convolutional Networks for sequence modeling.

## 1. Reference Paper

**Title:** An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

**Authors:** Shaojie Bai, J. Zico Kolter, Vladlen Koltun

**Paper:** https://arxiv.org/abs/1803.01271

**Official reference implementation:** https://github.com/locuslab/TCN

The reference paper evaluates TCNs against recurrent models on synthetic sequence tasks and real-world datasets. This project focuses on the assigned Adding Problem, Copy Memory, and Sequential MNIST benchmarks.

## 2. Implementation

The core implementation is written independently in PyTorch in `tcn.py`.

The TCN contains:
- Causal 1D convolutions with left padding.
- Dilated convolutions with exponentially increasing dilation.
- Residual blocks with channel projection when required.
- ReLU activations, dropout, and optional weight normalization.
- A configurable temporal convolutional network with a calculated receptive field.

LSTM and GRU baselines use standard PyTorch recurrent layers. The notebook contains the task-specific model wrappers, data generators, training loops, evaluation, checkpointing, and plotting code.

## 3. Requirements and Setup

The recommended environment is Google Colab with a GPU runtime when available.

The notebook uses:
- Python
- PyTorch
- torchvision
- pandas
- matplotlib

A GPU is helpful for Sequential MNIST, but availability depends on Colab's resource allocation.

### Running the project

1. Open `TCN_Implementation.ipynb` in Google Colab.
2. Ensure the notebook has access to the Google Drive folder `DL_TCN_Project`.
3. Run the notebook cells in order, starting with the setup and TCN validation cells.
4. Verify the model shape and causality tests before running experiments.
5. Training checkpoints, histories, results, and plots are saved in Google Drive.
6. Inspect the saved CSV files for the final reported metrics.

Do not use **Run all** without checking the checkpoint configuration and execution order. Training cells can resume compatible saved experiments, but a fresh runtime still needs its imports, datasets, and model definitions initialized.

### Project outputs

The Drive project directory contains:

- `checkpoints/` — latest and best model checkpoints.
- `results/` — experiment comparisons, training histories, and consolidated results.
- `results/plots/` — generated figures.
- `data/mnist/` — downloaded MNIST dataset.
- `report/` — reserved for report materials.

Large model checkpoints and datasets are not required to be committed to GitHub.

## 4. Experimental Configuration

### Adding Problem

- Sequence length: 100.
- Training samples: 4,096.
- Validation samples: 1,024.
- Test samples: 2,048.
- Training epochs: 20 per model.
- TCN: five blocks, 16 channels per block, kernel size 3, dropout 0.1.
- LSTM and GRU: hidden size 48.
- Evaluation metric: test mean squared error (MSE).
- Measured constant-prediction baseline: MSE 0.17104.

### Copy Memory

Two configurations were evaluated.

**Longer-gap run**
- Memory gap: 100.
- Total sequence length: 120.
- Training samples: 4,096.
- Validation samples: 1,024.
- Test samples: 2,048.
- Training epochs: 20.
- Metric: accuracy on the final ten recalled digits.

**Shorter-gap diagnostic**
- Memory gap: 20.
- Total sequence length: 40.
- Training samples: 2,048.
- Validation samples: 512.
- Test samples: 1,024.
- Training epochs: 30.
- Metric: accuracy on the final ten recalled digits, plus exact-sequence accuracy.
- The training loss focuses on the final ten recall positions.

The shorter-gap diagnostic is not a reproduction of the paper's longest Copy Memory benchmark.

### Sequential MNIST

- Training samples: 12,000.
- Validation samples: 2,000.
- Test samples: 5,000.
- Training epochs: 8 per model.
- Sequence length: 784 pixels.
- The image is flattened into a sequence of pixel intensities.
- Metric: test classification accuracy.

### Dilation Ablation

- Task: Adding Problem.
- Five temporal blocks with 16 channels per block.
- Kernel size: 3.
- Dropout: 0.1.
- Training epochs: 20 per configuration.
- Comparison: exponential dilation versus fixed dilation of 1.
- The dataset splits are held fixed.
- Receptive fields: 125 steps with dilation and 21 steps without dilation.

## 5. Results

These are the actual results from the saved experiments. The paper's results use different model sizes, sequence lengths, training configurations, or metrics in some cases.

### Adding Problem

| Model | Test MSE | Constant baseline |
|---|---:|---:|
| TCN | 0.013991 | 0.171037 |
| LSTM | 0.171229 | 0.171037 |
| GRU | 0.015329 | 0.171037 |

The TCN and GRU learned to reduce the prediction error substantially below the baseline. The LSTM did not improve meaningfully under the selected training configuration.

### Copy Memory

| Model | Test recall accuracy, gap 100 | Test digit accuracy, gap 20 |
|---|---:|---:|
| TCN | 12.62% | 99.94% |
| LSTM | 12.35% | 12.42% |
| GRU | 15.25% | 32.13% |

On the shorter-gap diagnostic, the TCN also achieved 99.41% exact-sequence accuracy. The longer-gap experiment did not achieve reliable recall for any of the three models.

The longer-gap experiment reports recall accuracy, while the reference paper reports loss in its comparison table. These values must not be compared numerically as if they were the same metric.

### Sequential MNIST

| Model | Test accuracy | Parameter count |
|---|---:|---:|
| TCN | 92.34% | 12,282 |
| LSTM | 11.08% | 17,802 |
| GRU | 42.32% | 13,514 |

The TCN learned the task under the reduced-scale training configuration. The recurrent baselines did not converge adequately within the selected eight epochs.

### Dilation Ablation

| Configuration | Receptive field | Test MSE |
|---|---:|---:|
| Dilated TCN | 125 | 0.023378 |
| TCN without dilation | 21 | 0.153073 |

The dilated configuration achieved lower test MSE in this experiment. Both models used the same data splits and channel widths.

## 6. Random Seeds and Reproducibility

The main experiments use fixed data-generation seeds and fixed model initialization seeds.

- Adding Problem data: 101 (training), 202 (validation), 303 (test). Model seeds: 1000, 1001, 1002.
- Copy Memory longer-gap data: 101, 202, 303. Model seeds: 2000, 2001, 2002.
- Copy Memory shorter-gap data: 401, 402, 403. Model seeds: 5000, 5001, 5002.
- Sequential MNIST: split seed 12345; model seeds 6000, 6001, 6002.
- Dilation ablation data: 101, 202, 303. Experiment seeds: 7101 and 7102.

The notebook saves checkpoints after completed epochs, including model weights and optimizer state, so compatible interrupted runs can resume. The reported results are from individual reduced-scale runs; repeated-seed statistical evaluation was not performed.

## 7. Limitations

This project is a reduced-scale implementation, not a full reproduction of every experiment in the reference paper.

The Copy Memory benchmark uses shorter sequences than the paper's longest benchmark. The recurrent baselines in Sequential MNIST and Copy Memory did not achieve strong performance under the selected training budgets. These results should not be interpreted as proof that recurrent networks are generally ineffective.

The dilation ablation supports the benefit of a larger receptive field in the tested Adding Problem setup, but it is a single controlled experiment rather than a multi-seed statistical study.

## 8. References

Bai, S., Kolter, J. Z., and Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271.

Paper: https://arxiv.org/abs/1803.01271

Reference code: https://github.com/locuslab/TCN
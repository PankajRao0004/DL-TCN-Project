# Temporal Convolutional Networks for Sequence Modeling

A PyTorch implementation of a **Temporal Convolutional Network (TCN)** based on Bai, Kolter, and Koltun (2018), with small-scale comparisons against LSTM and GRU baselines on the Adding Problem, Copy Memory, and Sequential MNIST.

> **Project status:** This is a learning-oriented experimental implementation, not an exact reproduction of the paper. The notebook has been run end-to-end and its measured CSV tables and plots are included in `results/`. Results describe one saved run; they are not multi-seed averages.

## Reference

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun, [*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*](https://arxiv.org/abs/1803.01271), 2018. The authors' reference implementation is available at [locuslab/TCN](https://github.com/locuslab/TCN).

## What is implemented

- Causal 1D convolutions so an output at time *t* does not depend on future inputs.
- Dilated convolution layers, residual connections, dropout, and a receptive-field calculation.
- TCN, LSTM, and GRU comparisons on three sequence-learning tasks.
- A controlled Adding Problem ablation comparing a dilated TCN with a non-dilated TCN of the same parameter count.
- Saved experiment tables, a consolidated summary, a paper-reference comparison, and plots under `results/`.

The notebook loads `tcn.py` from this repository in its first code cell. Open `TCN_Implementation.ipynb` in Google Colab, select a GPU runtime (the executed run used a Tesla T4), and run the notebook from top to bottom. MNIST is downloaded by `torchvision` when needed. Required libraries are PyTorch, torchvision, pandas, and matplotlib; Colab normally supplies them already.

## Experiments and current results

All values below come from the saved notebook run. Lower MSE is better; higher accuracy is better.

### 1. Adding Problem

The input is a sequence of 100 random values with two marked positions. The target is the sum of the two marked values. Training/validation/test sizes are 4,096 / 1,024 / 2,048; training runs for 20 epochs. The TCN and GRU use five 16-channel temporal blocks (kernel size 3, dropout 0.1); the recurrent baseline uses a 48-unit hidden state.

| Model | Parameters | Test MSE | Constant baseline MSE |
|---|---:|---:|---:|
| TCN | 7,393 | **0.009946** | 0.171037 |
| GRU | 7,537 | 0.012469 | 0.171037 |
| LSTM | 10,033 | 0.170794 | 0.171037 |

**Interpretation:** TCN and GRU learn this task in the current setup. The LSTM result is effectively at the constant-prediction baseline: its numerical improvement is only about 0.000243 MSE and should not be described as meaningful learning.

### 2. Copy Memory — long gap

The total sequence length is 120 (memory gap 100). The first ten input tokens are digits sampled from 1–8; one delimiter token `9` is placed at position `-11`; the final ten input tokens are zero. Targets are zero except for the final ten positions, which copy the original ten digits. The training objective weights each of the final ten target positions by 10. Training/validation/test sizes are 4,096 / 1,024 / 2,048, with 20 epochs.

| Model | Test accuracy on final ten digits |
|---|---:|
| TCN | 12.82% |
| LSTM | 13.00% |
| GRU | 15.91% |

**Interpretation:** None of the models reliably solves long-gap recall in this setup. These scores are low and close to the 10% per-digit accuracy expected from uniform guessing over ten classes. The notebook also records an all-zero prediction baseline of 0%; that is a different, deliberately simple baseline and should not be confused with chance accuracy. The TCN receptive field is 125 time steps, so it spans this 120-step input, yet the task still fails—having sufficient receptive field is not by itself sufficient to learn recall.

### 3. Copy Memory — short-gap diagnostic

This diagnostic uses memory gap 20 (total length 40), 2,048 / 512 / 1,024 train/validation/test examples, and 30 epochs. The training/evaluation loss and accuracy focus on the final ten recall positions.

| Model | Test digit accuracy | Exact ten-digit sequence accuracy |
|---|---:|---:|
| TCN | **99.98%** | **99.80%** |
| LSTM | 12.42% | 0.00% |
| GRU | 30.21% | 0.00% |

**Interpretation:** The TCN solves the short-gap diagnostic, while the recurrent baselines do not reproduce a complete sequence in this configuration. This diagnostic is not the paper's long-memory benchmark and should be labeled separately.

### 4. Sequential MNIST

Each 28×28 image is flattened into a sequence of 784 pixel values, and the model predicts one of ten digit classes from the final sequence representation. The run uses 12,000 training images, 2,000 validation images selected from the MNIST training set, and a 5,000-image test subset; training lasts eight epochs with batch size 128. The TCN has eight 16-channel blocks, kernel size 3, dropout 0.1, and receptive field 1,021. LSTM and GRU use 64 hidden units.

| Model | Parameters | Test accuracy |
|---|---:|---:|
| TCN | 12,282 | **91.22%** |
| GRU | 13,514 | 42.32% |
| LSTM | 17,802 | 11.08% |

**Interpretation:** The TCN achieves the best test accuracy in this limited-data, eight-epoch setup. These results are not directly comparable to the paper's full benchmark settings: this run uses subsets of MNIST, fewer training epochs, and smaller models.

### 5. Dilation ablation — Adding Problem

Both ablation models have 7,393 parameters, use the same Adding Problem data split (4,096 / 1,024 / 2,048), and train for 20 epochs. The dilated model uses five blocks with dilations 1, 2, 4, 8, and 16; the non-dilated model uses dilation 1 in every block.

| Configuration | Receptive field | Parameters | Test MSE |
|---|---:|---:|---:|
| Dilated TCN | 125 | 7,393 | **0.022134** |
| Non-dilated TCN | 21 | 7,393 | 0.152661 |
| Constant-prediction baseline | — | — | 0.171037 |

**Interpretation:** Dilation substantially improves the Adding Problem result in this single-run ablation. It is evidence for this configuration, not a multi-seed statistical conclusion.

## Repository contents

```text
.
├── tcn.py
├── TCN_Implementation.ipynb
└── results/
    ├── adding_comparison.csv
    ├── copy_comparison_v2.csv
    ├── copy_comparison_v3.csv
    ├── seq_mnist_comparison.csv
    ├── adding_dilation_ablation.csv
    ├── final_results_summary.csv
    ├── paper_vs_our_results.csv
    └── plots/
        ├── adding_problem_mse.png
        ├── copy_memory_accuracy.png
        ├── sequential_mnist_accuracy.png
        └── dilation_ablation.png
```

The CSVs and figures in this repository are a **snapshot of the included run**. A fresh Colab run writes new artifacts to its local `results/` directory, which is temporary. Download/copy any new outputs before the runtime resets if you want to keep and commit them.

## Important limitations

- The experiments use smaller datasets, fewer epochs, and different settings than the paper; the project should not be presented as an exact reproduction.
- The long-gap Copy Memory task is not solved by any of the tested models.
- Only one run per configuration is reported. Although seeds are set in the experiment code, GPU operations may be nondeterministic, and no mean or standard deviation across repeated seeds is available.
- Small differences from older notebook/README versions can reflect revised code or task definitions as well as training randomness. Do not attribute every change solely to GPU nondeterminism.
- The `beats_baseline` Boolean in `adding_comparison.csv` is a strict numerical comparison. For LSTM, `True` reflects a tiny numerical margin and does not indicate meaningful learning.
- The paper-reference CSV is contextual only. The settings and metrics differ in several rows, so numeric comparisons should not be interpreted as like-for-like reproduction results.

## Suggested next work

After studying the current implementation, the highest-value follow-ups are: validate the long-gap Copy Memory generator and training objective against a clearly documented benchmark definition; report several seeds with mean ± standard deviation; and run controlled receptive-field/sequence-length experiments. Changes should be made only after the current code and metrics are understood.

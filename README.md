# Temporal Convolutional Networks for Sequence Modeling

A PyTorch implementation of a **Temporal Convolutional Network (TCN)** based on Bai, Kolter, and Koltun (2018), with reduced-scale comparisons against LSTM and GRU baselines on the Adding Problem, Copy Memory, and Sequential MNIST.

> **Project status:** The notebook has been executed end-to-end on a Colab Tesla T4 GPU. This is a reduced-scale implementation of the paper's central TCN design and selected experiments, not a full reproduction of every benchmark in the paper. The tables below report one run per configuration; they are not averages over repeated trials.

## Reference

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun, [*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*](https://arxiv.org/abs/1803.01271), 2018. The authors' reference implementation is available at [locuslab/TCN](https://github.com/locuslab/TCN). The paper and reference repository were consulted; the TCN core used in this project is implemented in `tcn.py`.

## What is implemented

- Causal one-dimensional convolutions with left-only padding, so an output at time *t* cannot depend on future inputs.
- Dilated convolutions with exponentially increasing dilation factors: 1, 2, 4, 8, ...
- Residual blocks with two causal convolutions, ReLU activations, dropout, and a 1 x 1 projection when channel dimensions differ.
- Weight normalization in the TCN convolutional layers.
- TCN, LSTM, and GRU comparisons on three sequence-learning tasks.
- A controlled Adding Problem ablation comparing an exponentially dilated TCN with a no-dilation TCN of the same parameter count.
- Shape, causality, and receptive-field checks for the core implementation.
- Result CSV files and plots under `results/`.

The project uses `nn.Dropout` in the residual blocks. The paper describes spatial dropout; this is a small implementation difference and is not presented as an exact reproduction of that regularization detail.

## Architecture and receptive field

Each temporal residual block contains two causal convolutions with the same dilation. For `L` blocks, kernel size `k`, and block dilations `d_i`, the receptive field is

\[
R = 1 + 2(k-1)\sum_{i=0}^{L-1} d_i.
\]

For exponential dilation, `d_i = 2^i`, giving

\[
R = 1 + 2(k-1)(2^L-1).
\]

With kernel size 3, the tested configurations have receptive fields of 29 (3 blocks), 125 (5 blocks), and 1,021 (8 blocks). Five blocks without dilation have a receptive field of 21.

## Experiments and latest results

All values below are from the latest executed notebook, `TCN_Implementation.ipynb`. Lower MSE is better; higher accuracy is better.

### 1. Adding Problem

Each sample is a length-100 sequence with two channels. The first contains random values in `[0, 1]`; the second marks exactly two positions. The target is the sum of the two values at the marked positions. The train/validation/test sizes are 4,096 / 1,024 / 2,048, with 20 epochs, batch size 128, and Adam learning rate 0.002. The TCN uses five 16-channel blocks, kernel size 3, and dropout 0.1. LSTM and GRU use 48 hidden units.

| Model | Parameters | Test MSE | Constant baseline MSE |
|---|---:|---:|---:|
| TCN | 7,393 | **0.013983** | 0.171037 |
| GRU | 7,537 | 0.012469 | 0.171037 |
| LSTM | 10,033 | 0.170794 | 0.171037 |

**Interpretation:** TCN and GRU learn the task in this configuration. The LSTM is effectively at the constant-prediction baseline: its improvement is only about 0.000243 MSE and should not be described as meaningful learning.

### 2. Copy Memory — long gap

The memory gap is 100 and the total sequence length is 120. The first ten input positions contain digits from 1 to 8; intermediate positions are zero; the final eleven input positions contain delimiter token `9`. The target is zero except for the final ten positions, which repeat the initial ten digits. This input-token definition follows the description in the paper. The train/validation/test sizes are 4,096 / 1,024 / 2,048, with 20 epochs and batch size 128. The full-sequence cross-entropy loss is unweighted (`RECALL_WEIGHT = 1.0`).

| Model | Test recall accuracy on final ten digits |
|---|---:|
| TCN | **12.53%** |
| LSTM | 12.22% |
| GRU | 12.34% |

**Interpretation:** None of the models reliably solves long-gap recall. The eight target digits (1–8) are equally likely, so uniform guessing among the valid digits has an expected per-digit accuracy of 12.5%. The measured results are approximately at this level. The all-zero predictor baseline is 0% on the recall positions and is a separate baseline. Although the TCN receptive field (125) spans the 120-step input, receptive-field coverage alone does not guarantee that the model learns the copying task.

### 3. Copy Memory — short-gap diagnostic

This diagnostic uses memory gap 20 (total sequence length 40), 2,048 / 512 / 1,024 train/validation/test examples, and 30 epochs. Its loss is computed only over the final ten recall positions; therefore, it changes both the gap and the loss scope relative to the long-gap run. It is a diagnostic, not a controlled gap-only comparison or a reproduction of the paper's long-memory benchmark.

| Model | Test digit accuracy | Exact ten-digit sequence accuracy |
|---|---:|---:|
| TCN | **100.00%** | **100.00%** |
| LSTM | 12.42% | 0.00% |
| GRU | 30.56% | 0.00% |

**Interpretation:** The TCN solves this shorter, recall-only diagnostic. Since both the memory gap and loss scope differ from the long-gap experiment, the improvement cannot be attributed to gap length alone.

### 4. Sequential MNIST

Each 28 x 28 MNIST image is flattened in row-major order into a sequence of 784 pixel values. The model predicts one of ten digit classes from the final sequence representation. The run uses 12,000 training images, 2,000 validation images selected from the MNIST training split, and 5,000 test images; training lasts eight epochs with batch size 128 and Adam learning rate 0.001. The TCN uses eight 16-channel blocks, kernel size 3, dropout 0.1, and receptive field 1,021. LSTM and GRU use 64 hidden units.

| Model | Parameters | Test accuracy |
|---|---:|---:|
| TCN | 12,282 | **91.62%** |
| GRU | 13,514 | 42.32% |
| LSTM | 17,802 | 11.08% |

**Interpretation:** The TCN is the strongest model in this reduced run. The LSTM loss stayed close to the ten-class uniform-prediction cross-entropy of about 2.3026, and its accuracy remained close to chance. The GRU learned partially but did not converge to a strong result in eight epochs. The recurrent baselines were not extensively tuned, so these results do not establish that TCN is universally better than properly tuned recurrent networks.

### 5. Dilation ablation — Adding Problem

Both models use the same Adding Problem data split (4,096 / 1,024 / 2,048), five blocks, 16 channels, kernel size 3, dropout 0.1, batch size 128, and 20 epochs. The dilated model uses dilation factors 1, 2, 4, 8, and 16; the no-dilation model uses dilation 1 in every block. Both have 7,393 parameters.

| Configuration | Receptive field | Parameters | Test MSE |
|---|---:|---:|---:|
| Dilated TCN | 125 | 7,393 | **0.020733** |
| No-dilation TCN | 21 | 7,393 | 0.152704 |
| Constant-prediction baseline | — | — | 0.171037 |

**Interpretation:** Dilation substantially improves test MSE in this single-run ablation. This supports the usefulness of a larger receptive field for this configuration, but it is not a multi-seed statistical conclusion.

## Reproducibility

The notebook sets data/split and model seeds as follows:

| Experiment | Data/split seeds | Model seeds |
|---|---|---|
| Adding Problem | 101 (train), 202 (validation), 303 (test) | 1000, 1001, 1002 |
| Copy Memory, gap 100 | 101, 202, 303 | 3000, 3001, 3002 |
| Copy Memory, gap 20 | 401, 402, 403 | 5000, 5001, 5002 |
| Sequential MNIST | Split seed 12345 (test split uses seed 12346) | 6000, 6001, 6002 |
| Dilation ablation | 101, 202, 303 | 7101, 7102 |

Seeds make the data/model setup reproducible, but some CUDA operations can still be nondeterministic. Only one run per configuration is reported; no mean, standard deviation, or confidence interval across repeated seeds is available.

## Run the notebook

1. Open `TCN_Implementation.ipynb` in Google Colab.
2. Select a GPU runtime (the reported run used a Tesla T4).
3. Run the notebook from top to bottom. Its first code cell clones or updates this repository and imports `TemporalConvNet` from `tcn.py`.
4. The notebook downloads MNIST through `torchvision` when needed and writes result tables/plots to a local `results/` directory.

Required libraries are PyTorch, torchvision, pandas, and matplotlib. Colab normally provides them; install any missing package with `pip` if required.

**Important:** Colab's `/content/results/` folder belongs to the temporary runtime. The CSV and PNG files committed in this repository are a snapshot of the reported run. Download or copy any new results before the runtime resets if you want to keep them.

## Repository contents

```text
.
├── tcn.py
├── TCN_Implementation.ipynb
├── README.md
├── TCN_IEEE_Report.pdf
├── TCN_IEEE_Report.tex
├── figures/                 # Figures used by the IEEE report
│   ├── tcn_block.png
│   ├── adding_problem.png
│   ├── copy_memory.png
│   ├── sequential_mnist.png
│   └── dilation_ablation.png
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

## Limitations and next work

- The project uses smaller datasets, shorter sequences, and fewer epochs than the paper; it is a reduced-scale implementation, not a complete reproduction.
- The long-gap Copy Memory task is not solved by the tested models. The short-gap experiment changes both gap length and loss scope and must remain labeled as a diagnostic.
- The Sequential MNIST LSTM and GRU baselines were not extensively tuned; the LSTM in particular did not learn adequately in the eight-epoch run.
- Results are single runs. Repeated-seed statistics are not available.
- The paper reports Copy Memory loss for a longer benchmark (T=1000), while this project reports recall accuracy for gap 100 (total length 120); these values are not directly comparable.
- The paper uses spatial dropout in its TCN block description; this implementation uses PyTorch `nn.Dropout`.
- Useful follow-up work would include a longer/tuned recurrent baseline study, multiple random seeds, and a larger Copy Memory gap after the present implementation is understood.

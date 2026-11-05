# Hardware, environment and compute

Recorded so the figures in the paper can be placed. Verified 2026-11-05.

## Training

The released v1.3 run, and the v1.1 run before it, were trained on a single
rented GPU.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| Precision | float32 |
| torch | 2.8.0+cu129 |
| Epochs | 4, each kept; the third was released |
| Epoch wall time | about 41 minutes |
| GPU time, released run | about 2.8 hours |
| Mean training loss by epoch | 0.6029, 0.4336, 0.2714, 0.0849 |

The epoch wall time is the spacing of the four checkpoints as they were written.
The same rented machine also built the data and scored the checkpoints, which is
not counted in the 2.8 hours.

A later experiment that re-trained the recipe on other seeds ran on the same
class of machine at about 65 minutes per epoch, slower because scoring of earlier
checkpoints was competing for the GPU.

## Inference and measurement

Every figure in the paper other than the training trajectory was produced on
CPU, on one and the same machine.

| Item | Value |
|---|---|
| Machine | Apple M4 |
| Precision | float32 |
| torch | 2.8.0 |
| transformers | 5.22.1 |
| datasets | 5.1.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pyyaml | 6.0.3 |

The latency table was measured with nothing else running on the machine, as the
median of 25 runs after 5 discarded warm-up runs.

## Comparison model

Laya runs in its own environment, because the package pins versions of its own.

| Item | Value |
|---|---|
| Package | `laya` 0.4.22 |
| Checkpoint | `convaiinnovations/laya-multilingual`, revision `1720e3e3357c` |
| torch | 2.15.0 |
| transformers | 5.22.1 |
| numpy | 2.5.4 |

Laya requires a one-line instruction that Ru-Jev does not take. Two instructions
were used and both are reported, so the spread over how the question is phrased
is visible rather than hidden behind one favourable choice.

## Reproducing the released temperature

`scripts/verify_temperature.py` refits the released temperature of v1.3 from the
released weights and the validation split and checks it against the 1.8640 in
`calibration.json`.

# When does synthetic data help? Imbalance, allocation and fidelity in food image classification

Code, data splits and results for the paper by Abhishek Kataria, Rahul Nijhawan
and Raman Kumar Goyal (Thapar Institute of Engineering and Technology).

The study holds a frozen Stable Diffusion v1.5 generator and a Swin-B
classifier fixed, and varies only (a) the imbalance ratio of the real training
set and (b) the rule that allocates a fixed budget of synthetic images to
classes, on Food101-LT and UEC Food-256.

## Main result

Gain in top-1 accuracy from the same 3,800 synthetic images (mean over seeds):

| dataset    | imbalance ratio | real only | + synthetic | gain  |
|------------|----------------:|----------:|------------:|------:|
| Food101-LT | 150             | 71.80     | 77.24       | +5.45 |
| Food101-LT | 50              | 80.55     | 82.38       | +1.83 |
| Food101-LT | 20              | 85.63     | 86.27       | +0.64 |
| Food101-LT | 5               | 89.81     | 89.84       | +0.04 |
| UEC-256    | 7.9             | 85.32     | 85.18       | -0.14 |

Standard deviations and seed counts are in `figures/curve_table.csv`;
every run's full record is in `results/`.

## Repository layout

```
scripts/        split construction, image generation, training, figures
notebooks/      food101lt_allocation_arms: Food101-LT arms at IR 150
                imbalance_sweep: sweep training runs at IR 50, 20 and 5
                sweep_synth_list: builds the 3,800-image sweep list
                uec256_arms_and_fidelity: UEC-256 split, arms A to D,
                fidelity scoring and allocations (outputs cleared)
splits/         Food101-LT split files with SHA256SUMS, manifests,
                class counts and every synthetic-image allocation list
uec256_meta/    hand-written UEC-256 prompts, per-class fidelity scores,
                allocation lists and the pooled validation confusion matrix
results/        one JSON per training run (accuracy, head/tail accuracy,
                per-epoch history, configuration) and confusion matrices
figures/        all figures of the paper (PDF) and the curve table
```

## Data

Download the datasets from their original sources:

* Food-101: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
* UEC Food-256: http://foodcam.mobi/dataset256.html

Copy `splits/food101_meta/train_lt*.txt` into `food-101/meta/` and check them
with `sha256sum -c SHA256SUMS`.

## Reproducing the experiments

1. `scripts/build_food101_lt.py` and `scripts/build_sweep.py` rebuild the
   Food101-LT splits (the released files are the ones used in the paper).
2. `scripts/gen_synth.py` and `scripts/gen_synth_uec.py` generate the
   synthetic pools with Stable Diffusion v1.5 (50 DPM-Solver steps,
   guidance scale 2.0).
3. The notebooks and `scripts/finish_sweep.py` / `scripts/fix_leakage.py`
   train every arm; each run writes a `*_results.json`.
4. `scripts/make_curve.py`, `scripts/make_plots.py` and
   `scripts/make_figures.py` regenerate the figures and tables.

Paths are set as constants at the top of each script (`/workspace/...`);
edit them for your machine.

## Initialisation

The paper initialises Swin-B from the visual tower of an image-text alignment
model that we trained on the image-recipe pairs of Recipe1M. That checkpoint is
not included in this repository and is available from the corresponding author
on request. All training code
also supports standard ImageNet initialisation (`init="imagenet"`).

## Contact

Rahul Nijhawan (corresponding author): rahulnijhawan2010@gmail.com

## License

MIT, see `LICENSE`.

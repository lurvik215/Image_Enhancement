# Low-Light Image Enhancement - official MIRNet-v2

```
lowlight_enhancement/
├── DATASETS/
│   ├── LoLI-Street Dataset/{Train,Val}/{low,high}   Test/ (images only, no ground truth)
│   └── LOL-v2/{Real_captured,Synthetic}/{Train,Test}/{Low,Normal}
├── enhancement_results/<run_name>/                 <- created automatically
│   ├── checkpoints/  logs/  predictions/  visualizations/
│   ├── config.json  train_metrics.csv  <split>_metrics.csv  <split>_summary.csv
├── config.py             task config + CLI overrides
├── model_arch.py         official MIRNet-v2 (port of the authors' code)
├── losses.py             L1 + LPIPS + FFT-L1, weights from cfg.loss_weights
├── data_loader.py        dataset / dataloaders (LoLI-Street + shared code)
├── data_loader_lolv2.py  LOL-v2 file discovery, pairing and train/val split
├── utils.py              conversion, PSNR/SSIM, eval loop, optimizer/scheduler, checkpoints
├── train.py              training (gradient accumulation, resume, --resume_lr)
├── evaluate.py           evaluation (--split val|test)
├── pipeline.py           check -> train -> evaluate
├── benchmark.py          finds the training bottleneck (data loading vs GPU / memory)
├── baseline_identity.py  scores the unenhanced inputs against ground truth
└── requirements.txt
```

## Usage
```
python pipeline.py --stage check                         # LoLI-Street
python pipeline.py --stage check --dataset lolv2         # LOL-v2 (real subset)

python pipeline.py --dataset lolv2 --lolv2_subset real   # train + test on LOL-v2 real
python pipeline.py --dataset lolv2 --lolv2_subset both
python pipeline.py                                       # LoLI-Street
python train.py --dataset lolv2 --resume auto
python evaluate.py --dataset lolv2 --split test
```
Defaults (config.py): patch 256, batch 4 x 3 accumulation steps (effective 12), mixed precision,
LR 2e-4 -> 1e-6 (3 warm-up epochs, then cosine), 40 epochs, validation on all of Val after every epoch.
Use `--data_root` if the dataset is stored somewhere else, `--run_name` to name the run, and
`--no_amp` on GPUs where mixed precision is slow (e.g. Pascal / P5000).

The LoLI-Street paper's schedule: `--lr 1e-4 --lr_schedule step --lr_step_iters 5000 --lr_gamma 0.8`.

## Resuming and changing the LR
```
python pipeline.py --run_name <same name> --resume auto            # continue after an interruption
python pipeline.py --run_name <same name> --resume auto --resume_lr 6.4e-5 --lr_schedule cosine
                                                                   # restart the LR schedule from a new value
```
Without --resume_lr, --lr / --lr_gamma / --lr_step_iters are restored from the checkpoint and ignored.

## Extra tools
```
python benchmark.py                          # is training limited by image loading or by the GPU? (memory, s/step)
python baseline_identity.py --run_name <n>   # PSNR / SSIM of the UNENHANCED inputs (the "do nothing" row)
```
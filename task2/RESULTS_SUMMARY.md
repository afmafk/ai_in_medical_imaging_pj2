# Experiment Results Summary

## Overview

This document summarizes the current test outputs and loss-curve artifacts in the project.
Detailed per-patient metrics remain in the original JSON files under `evaluation_outputs/`.

## Evaluated Runs

| Model | Target Mode | Modalities | Test Loss | Dice WT | Dice TC | Dice ET | Dice Mean | HD95 Mean | Metrics File |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `multiclass_t1n` | `multiclass` | `t1n` | 0.4856 | 0.7637 | 0.6485 | 0.4954 | 0.6359 | 19.1681 | [test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t1n/test_metrics.json) |
| `multiclass_t1c` | `multiclass` | `t1c` | 0.3233 | 0.7791 | 0.9053 | 0.8730 | 0.8525 | 9.1433 | [test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t1c/test_metrics.json) |
| `multiclass_t2w_t2f` | `multiclass` | `t2f + t1c + t2w` | 0.2269 | 0.9129 | 0.9125 | 0.8788 | 0.9014 | 4.9831 | [test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t2w_t2f/test_metrics.json) |
| `multiclass_all_modalities` | `multiclass` | `t1n + t1c + t2w + t2f` | 0.2075 | 0.9257 | 0.9285 | 0.8931 | 0.9158 | 4.1394 | [test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_all_modalities/test_metrics.json) |
| `multiclass_t2f_t1c_t2w` | `multiclass` | `t2f + t1c + t2w` | 0.2224 | 0.9369 | 0.9519 | 0.8997 | 0.9295 | N/A | [test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t2f_t1c_t2w/test_metrics.json) |
| `regions_t2f_t1c_t2w` | `regions` | `t2f + t1c + t2w` | 0.4810 | 0.9112 | 0.9236 | 0.8874 | 0.9074 | 7.7148 | [test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/regions_t2f_t1c_t2w/test_metrics.json) |

## Quick Takeaways

- The strongest `Dice Mean` among the current outputs is `multiclass_t2f_t1c_t2w` with **0.9295**.
- Among runs that also report `HD95 Mean`, `multiclass_all_modalities` is strongest overall with **Dice Mean = 0.9158** and **HD95 Mean = 4.1394**.
- Single-modality `t1n` is the weakest configuration in the current set, especially on `TC` and `ET`.
- The new three-modality multiclass run (`t2f + t1c + t2w`) outperforms the earlier three-modality multiclass result currently saved in `multiclass_t2w_t2f`.

## Current Run Notes

- Training log: [3652106.out](/d:/ai_medical_imaging/project2/3652106.out)
- Current run checkpoint metrics: [test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t2f_t1c_t2w/test_metrics.json)
- Corresponding loss curve: [loss_curve_multiclass_t2f_t1c_t2w.png](/d:/ai_medical_imaging/project2/loss_curve_multiclass_t2f_t1c_t2w.png)

From the current log, early stopping was not enabled in configuration, so the loss curve reflects the logged epochs directly rather than an early-stopped run.

## Loss Curve Files

- [loss_curve_multiclass_t1n.png](/d:/ai_medical_imaging/project2/loss_curve_multiclass_t1n.png)
- [loss_curve_multiclass_t1c.png](/d:/ai_medical_imaging/project2/loss_curve_multiclass_t1c.png)
- [loss_curve_multiclass_t2w_t2f.png](/d:/ai_medical_imaging/project2/loss_curve_multiclass_t2w_t2f.png)
- [loss_curve_multiclass_t2f_t1c_t2w.png](/d:/ai_medical_imaging/project2/loss_curve_multiclass_t2f_t1c_t2w.png)
- [loss_curve_regions_t2f_t1c_t2w.png](/d:/ai_medical_imaging/project2/loss_curve_regions_t2f_t1c_t2w.png)

## Source Files

- [evaluation_outputs/multiclass_t1n/test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t1n/test_metrics.json)
- [evaluation_outputs/multiclass_t1c/test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t1c/test_metrics.json)
- [evaluation_outputs/multiclass_t2w_t2f/test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t2w_t2f/test_metrics.json)
- [evaluation_outputs/multiclass_all_modalities/test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_all_modalities/test_metrics.json)
- [evaluation_outputs/multiclass_t2f_t1c_t2w/test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/multiclass_t2f_t1c_t2w/test_metrics.json)
- [evaluation_outputs/regions_t2f_t1c_t2w/test_metrics.json](/d:/ai_medical_imaging/project2/evaluation_outputs/regions_t2f_t1c_t2w/test_metrics.json)

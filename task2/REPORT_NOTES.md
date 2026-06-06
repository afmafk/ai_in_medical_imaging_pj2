# BraTS 2D Segmentation Experiment Notes

## 1. Dataset and Preprocessing

This project uses the preprocessed 2D dataset stored in `processed_2d/`.

Each sample is saved as an `.npz` file and contains:

- `image`: 4 MRI modalities with shape `(4, 177, 219)`
- `seg`: single-channel segmentation mask with labels `{0, 1, 2, 3}`
- `regions`: 3 binary region masks with shape `(3, 177, 219)`
- `modalities`: `["t1n", "t1c", "t2w", "t2f"]`
- `region_names`: `["WT", "TC", "ET"]`

The preprocessing recorded in `metadata.json` is:

1. Raw visualization before preprocessing
2. Global crop using non-zero voxels from all modalities and all patients
3. Per-patient, per-modality Z-score normalization over non-background voxels only
4. Background voxels set to 0 after normalization
5. 2D slices preserve every z slice after crop

After preprocessing, all 2D slices have a fixed spatial size of `177 x 219`.

## 1.1 Black Slice Removal

An important issue in this dataset is that the training, validation, and test sets all contain fully black slices.

These are slices where the image pixel values are entirely zero across all modalities.

To avoid wasting training iterations on non-informative samples, fully black slices are automatically removed before model training and evaluation.

### Filtering rule

- if all pixel values of the stored image are `0`, the slice is removed

In code, this is implemented by checking:

`np.all(sample["image"] == 0)`

### Observed dataset statistics

- total patients: `78`
- total slices before black-slice removal: `12090`
- fully black slices removed: `1210`

This filtering is applied consistently when constructing the train, validation, and test datasets.

## 2. Modality Selection

To determine which modalities provide the best tumor-background discrimination, the files `contrast_analysis.csv` and `contrast_summary.json` were analyzed.

The main selection criterion was `normalized_contrast`, which measures relative contrast between tumor tissue and healthy tissue.

### Motivation for modality comparison

The motivation for this part of the study was to understand whether each MRI modality contributes differently to different tumor components, and then verify this idea experimentally by training models with different modality inputs.

In brain tumor imaging, no single modality fully captures the entire tumor extent:

- `t1c` is most informative for enhancing tumor because contrast enhancement highlights regions where the blood-brain barrier is disrupted
- `t2w` and `t2f` are more informative for edema and the broader whole-tumor extent, because fluid-related abnormal regions appear hyperintense
- `t1n` can provide anatomical structure and some information about the lesion core, but usually has weaker direct tumor-background contrast than the other modalities

Therefore, the goal was not only to pick one "best-looking" modality, but to test whether:

1. each single modality can achieve its own best segmentation behavior for the tumor regions it represents most clearly
2. combining complementary modalities can produce better overall segmentation than any single modality alone

This is why the experiments include:

- single-modality baselines such as `t1n` and `t1c`
- a selected multi-modality combination `t2f + t1c + t2w`
- an all-modality input setting `t1n + t1c + t2w + t2f`

The comparison is meant to answer two questions:

- which modality is strongest when used alone
- whether multi-modal fusion provides a clear benefit by combining complementary information from edema-sensitive and enhancement-sensitive sequences

### Mean normalized contrast by region

| Modality | WT | TC | ET |
|---|---:|---:|---:|
| `t1n` | -0.037 | -0.212 | -0.100 |
| `t1c` | 0.311 | 1.396 | 2.078 |
| `t2w` | 0.957 | 1.180 | 0.931 |
| `t2f` | 2.659 | 2.867 | 2.847 |

### Modality decision

- `t2f` shows the strongest and most consistent discrimination for all tumor regions.
- `t1c` is particularly strong for enhancing tumor (`ET`) and also useful for `TC`.
- `t2w` provides additional complementary tumor contrast.
- `t1n` was not selected because its contrast is weak and sometimes negative.

### Final selected input modalities

- `t2f`
- `t1c`
- `t2w`

These three modalities are concatenated directly as the network input channels.

This selected combination is motivated by complementary tumor visibility:

- `t1c` contributes the clearest enhancement-related signal
- `t2f` contributes strong whole-tumor and edema contrast
- `t2w` provides additional fluid-sensitive structural support

So the design principle is to let each modality contribute where it is strongest, rather than expecting one modality alone to solve the full segmentation problem.

## 3. Input and Target Design

Two output designs were considered during development.

The purpose of comparing these two target designs was not only to compare segmentation accuracy, but also to check whether the region-based formulation would introduce severe overlap or consistency problems between predicted tumor regions.

### Option A: Single-channel multiclass mask

- target is `seg`
- labels are `{0, 1, 2, 3}`
- output logits have 4 channels
- final prediction is a 4-class segmentation map

In this representation:

- `0` = background
- `1`, `2`, `3` are the original tumor subregion labels from the dataset

This formulation is **mutually exclusive** at the pixel level:

- each pixel can belong to exactly one class only
- the final prediction is obtained with `argmax` over 4 logits

### Option B: Three binary channels

- target is `regions`
- channels are `WT`, `TC`, `ET`
- output logits have 3 channels
- each channel is trained as a binary map

These three channels are derived region targets rather than four mutually exclusive classes:

- `WT` = whole tumor = labels `1, 2, 3`
- `TC` = tumor core = labels `1, 3`
- `ET` = enhancing tumor = label `3`

This formulation is **not mutually exclusive**:

- a pixel in `ET` also belongs to `TC`
- a pixel in `TC` also belongs to `WT`
- therefore the three channels have a nested, overlapping relationship

The final prediction is obtained by thresholding each output channel independently.

### Final decision

The final choice is **Option A: single-channel multiclass segmentation**.

Reasons:

- it is the simplest setup
- it matches the original `seg` labels directly
- it works naturally with `CrossEntropy + Dice`
- it avoids consistency issues between overlapping region targets such as `WT`, `TC`, and `ET`

Although a three-channel region-based output was also implemented during development, the final main experiment uses the single-channel multiclass output.

### Why multiclass and regions are genuinely different

Although both formulations are evaluated using `WT`, `TC`, and `ET` Dice scores in the end, they are trained in different ways.

This comparison was especially important because the `regions` design predicts overlapping clinical regions directly. We therefore wanted to test whether this direct region prediction strategy would create serious overlap-related ambiguity or unstable behavior compared with the cleaner mutually exclusive `multiclass` formulation.

#### Multiclass

- the network learns a single 4-way classification problem at each pixel
- class competition is explicit because only one class can win at each location
- the output respects anatomical exclusivity automatically
- the target matches the stored segmentation labels directly

#### Regions

- the network learns three separate binary segmentation problems
- each region is optimized directly in clinically meaningful form
- overlaps between channels are allowed by design
- additional post-processing logic is needed conceptually because the three outputs are not independent in anatomy even though they are predicted separately

In short:

- `multiclass` asks: "which one of the 4 classes does this pixel belong to?"
- `regions` asks: "does this pixel belong to WT?", "does it belong to TC?", and "does it belong to ET?" at the same time

## 4. Network Architecture

Although a multimodal UNet implementation already exists in the project, the final model choice is a **simple 2D UNet** with direct channel concatenation.

### Final architecture

- model: `simple_unet`
- input channels: `3`
- output channels: `4`
- input order: `[t2f, t1c, t2w]`

### Design choice

Instead of using separate modality stems or attention-based fusion, the three selected modalities are concatenated directly and passed into a standard UNet:

`[B, 3, 177, 219] -> UNet -> [B, 4, 177, 219]`

This was chosen because:

- the goal is to use the simplest effective baseline
- the three selected modalities already contain the strongest contrast information
- direct concatenation reduces architectural complexity

## 5. Loss Function

Loss ablation was not performed.

The final training loss is:

- `CrossEntropyLoss`
- `DiceLoss`
- combined as `CrossEntropy + Dice`

This is implemented as `dice_ce` in the project.

### Why this loss was chosen

- `CrossEntropy` handles multiclass pixel classification directly
- `Dice` improves overlap quality, especially for imbalanced tumor regions
- the combination is a standard and reliable baseline for medical image segmentation

## 5.1 Learning Rate Decay Strategy

The training process uses **ReduceLROnPlateau** as the learning rate decay strategy.

The scheduler monitors the validation loss:

- if the validation loss stops improving for a number of epochs, the learning rate is reduced automatically

### Current scheduler settings

- scheduler: `ReduceLROnPlateau`
- mode: `min`
- factor: `0.5`
- patience: `10`
- threshold: `1e-4`
- minimum learning rate: `1e-6`

### Why this strategy was chosen

- it adapts to validation performance instead of using a fixed decay schedule
- it is simple and robust for segmentation training
- it helps stabilize later-stage optimization when validation improvement slows down

## 6. Data Augmentation

To improve model generalization, the following augmentations are applied during training:

- random rotation
- random horizontal flipping
- random vertical flipping
- random scaling
- Gaussian noise

### Important implementation detail

Because the dataset was cropped tightly using a minimal bounding box, direct rotation could cut off parts of the brain near the image boundary.

To avoid this, augmentation is implemented in a safer way:

1. pad the image and mask with zeros around all sides
2. apply random rotation and scaling
3. center-crop back to the original size
4. apply flipping
5. add Gaussian noise to the image only

### Current augmentation settings

- padding: `24` pixels on each side
- rotation range: `[-15 degrees, 15 degrees]`
- horizontal flip probability: `0.5`
- vertical flip probability: `0.5`
- scaling range: `[0.9, 1.1]`
- Gaussian noise probability: `0.3`
- Gaussian noise standard deviation: `0.05`

### Why the mask is also transformed

Spatial transforms must be applied to both the image and the segmentation mask using the same parameters.

- the image uses bilinear interpolation
- the mask uses nearest-neighbor interpolation

This preserves the discrete mask labels and avoids invalid interpolated class values.

## 7. Final Experimental Setup

The final baseline configuration is:

- dataset: `processed_2d`
- black slice removal: enabled
- input modalities: `t2f`, `t1c`, `t2w`
- input tensor shape: `[B, 3, 177, 219]`
- target: single-channel multiclass mask `seg`
- classes: `4` (`0, 1, 2, 3`)
- model: simple 2D UNet
- loss: `CrossEntropy + Dice`
- learning rate scheduler: `ReduceLROnPlateau`
- augmentation: rotation, flipping, scaling, Gaussian noise
- safe spatial augmentation: pad -> transform -> center crop

## 7.1 Patient-Level Data Split

The dataset split must be performed at the **patient level**, not at the slice level.

This means that all slices from the same patient must belong to only one subset:

- training set
- validation set
- test set

This avoids data leakage caused by putting different slices from the same patient into different subsets.

### Final split ratio

- training: `70%`
- validation: `10%`
- test: `20%`

This corresponds to a patient-level split ratio of `7:1:2`.

### Actual split for the current dataset

There are `78` patients in total, so the final split is:

- train: `54` patients
- validation: `7` patients
- test: `17` patients

After black-slice removal, the number of slices in each subset is:

- train: `8370` slices
- validation: `1085` slices
- test: `2635` slices

### Why patient-level splitting is necessary

- neighboring slices from the same patient are highly correlated
- slice-level random splitting would leak patient-specific information
- patient-level splitting provides a more realistic generalization estimate

## 7.2 Multi-GPU Support

The training code supports multi-GPU execution on a single node.

If more than one GPU is visible to PyTorch, the model is wrapped automatically using `torch.nn.DataParallel`.

### Current implementation behavior

- if `torch.cuda.device_count() > 1`, the model uses `DataParallel`
- the same training script can run on either 1 GPU or multiple GPUs
- when saving checkpoints, the code saves `model.module.state_dict()` for multi-GPU runs

This keeps checkpoint files compatible with later single-GPU or inference usage.

### Current cluster usage

The current cluster configuration is set up to use **2 GPUs**.

This provides practical acceleration while keeping the implementation simple, since no distributed training launch logic is required.

## 7.3 Cluster Job Configuration

Training is intended to run on the `wuyt` cluster using `.job` submission scripts.

### Current runtime environment

- cluster partition: `bme_gpu`
- GPUs requested: `2`
- CPU setting: `-n 1`, `-c 64`
- memory: `256G`
- wall time: `24:00:00`
- conda environment: `python310`
- working directory: `/home_data/home/wuyt22023/drz2024`

### Current job files

- `job_unet_multiclass.job`
- `job_unet_regions.job`

Among these two files, the primary experiment is `job_unet_multiclass.job`, because the final selected target format is the single-channel multiclass mask.

## 7.4 Test Set Results for All Trained Models

All reported test results are produced on the same fixed patient-level test split reconstructed from each saved `run_config.json`, using the same split seed and the same preprocessing pipeline.

The project currently contains six completed evaluation outputs:

1. `multiclass_t1n`
2. `multiclass_t1c`
3. `multiclass_t2w_t2f`
4. `multiclass_all_modalities`
5. `multiclass_t2f_t1c_t2w`
6. `regions_t2f_t1c_t2w`

### Important naming note

Some output folder names are historical and do not perfectly match the final modality list:

- `multiclass_t1c` corresponds to the checkpoint folder `unet_multiclass_t1ce`, which is the single-modality `t1c` run
- `multiclass_t2w_t2f` corresponds to the earlier 3-modality multiclass checkpoint stored in `unet_multiclass`, but its actual modalities are `t2f`, `t1c`, `t2w`
- `multiclass_t2f_t1c_t2w` is a later rerun of the same 3-modality multiclass setting with a separate checkpoint and improved final metrics

### Full model comparison

| Model label | Checkpoint | Target mode | Input modalities | Test loss | Dice WT | Dice TC | Dice ET | Mean Dice | HD95 Mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `multiclass_t1n` | `unet_multiclass_t1/best_model.pt` | `multiclass` | `t1n` | `0.4856` | `0.7637` | `0.6485` | `0.4954` | `0.6359` | `19.1681` |
| `multiclass_t1c` | `unet_multiclass_t1ce/best_model.pt` | `multiclass` | `t1c` | `0.3233` | `0.7791` | `0.9053` | `0.8730` | `0.8525` | `9.1433` |
| `multiclass_t2w_t2f` | `unet_multiclass/best_model.pt` | `multiclass` | `t2f`, `t1c`, `t2w` | `0.2269` | `0.9129` | `0.9125` | `0.8788` | `0.9014` | `4.9831` |
| `multiclass_all_modalities` | `unet_multiclass_all_modalities/best_model.pt` | `multiclass` | `t1n`, `t1c`, `t2w`, `t2f` | `0.2075` | `0.9257` | `0.9285` | `0.8931` | `0.9158` | `4.1394` |
| `multiclass_t2f_t1c_t2w` | `unet_multiclass_t2f_t1c_t2w/best_model.pt` | `multiclass` | `t2f`, `t1c`, `t2w` | `0.2224` | `0.9369` | `0.9519` | `0.8997` | `0.9295` | `N/A` |
| `regions_t2f_t1c_t2w` | `unet_regions/best_model.pt` | `regions` | `t2f`, `t1c`, `t2w` | `0.4810` | `0.9112` | `0.9236` | `0.8874` | `0.9074` | `7.7148` |

### What is different between these models

The runs differ along two main dimensions:

1. **Input modalities**
2. **Output target design**

#### A. Differences in input modalities

The following modality combinations were tested:

- `t1n` only
- `t1c` only
- `t2f + t1c + t2w`
- `t1n + t1c + t2w + t2f`

Their roles are different:

- `t1n` provides the weakest tumor contrast in the earlier contrast analysis, so it was tested mainly as a low-information single-modality baseline
- `t1c` is highly informative for enhancing tumor (`ET`) and also helps `TC`, so it was tested as a strong single-modality baseline
- `t2f + t1c + t2w` is the selected 3-modality combination because these three modalities showed the strongest complementary contrast
- `all modalities` adds `t1n` back on top of the selected 3-modality set to test whether extra input channels improve performance

#### B. Differences in output targets

Two output formulations were tested:

- `multiclass`: predict one 4-class segmentation map with labels `{0,1,2,3}`
- `regions`: predict 3 binary region maps corresponding to `WT`, `TC`, and `ET`

Their practical differences are:

- the `multiclass` setup is a mutually exclusive 4-class problem solved with one `argmax` prediction per pixel
- the `regions` setup is three overlapping binary problems solved independently for `WT`, `TC`, and `ET`
- in the `regions` setup, `ET` is nested inside `TC`, and `TC` is nested inside `WT`
- the `multiclass` setup maps directly to the original stored label map
- the `regions` setup maps directly to clinically used derived tumor regions
- the `regions` output may help some region-specific Dice metrics, but it is structurally more complex because the output channels are not independent

The motivation for running this comparison was to determine whether the overlapping region formulation would cause severe overlap or consistency issues in practice. If those issues had been severe, the region-based setup would have been less reliable even if some region Dice values looked competitive.

### Result interpretation

#### 1. Single-modality baselines

- `t1n` performs worst by a large margin, with `Mean Dice = 0.6359`
- `t1c` is much stronger than `t1n`, with `Mean Dice = 0.8525`

This confirms the earlier contrast analysis:

- `t1n` alone is not sufficient for reliable tumor segmentation
- `t1c` alone is useful, especially for contrast-enhancing tumor structure

#### 2. Effect of using the selected 3-modality input

Using `t2f + t1c + t2w` produces a large improvement over either single-modality baseline:

- earlier 3-modality multiclass run: `Mean Dice = 0.9014`
- later 3-modality multiclass rerun: `Mean Dice = 0.9295`

This shows that:

- combining the three selected modalities is substantially better than using only `t1n` or only `t1c`
- the chosen modality-selection strategy is justified by the final segmentation performance

#### 3. Effect of adding all four modalities

The `all modalities` run achieves:

- `Mean Dice = 0.9158`
- `HD95 Mean = 4.1394`

This is better than the earlier 3-modality multiclass run, but still below the later `multiclass_t2f_t1c_t2w` rerun in Dice mean.

This suggests that:

- adding `t1n` does not necessarily guarantee the best Dice score
- the selected 3-modality subset remains a very strong and efficient configuration
- more input channels can help, but the benefit is not guaranteed if the added modality contributes limited discriminative information

#### 4. Multiclass vs region-based output

For the same selected 3-modality input, the `regions` model reaches:

- `Mean Dice = 0.9074`

The later multiclass rerun with the same modalities reaches:

- `Mean Dice = 0.9295`

So in the current set of completed experiments:

- the best overall Dice result comes from the **multiclass** formulation with `t2f + t1c + t2w`
- the region-based output is still competitive, but it is not the strongest final result in this project

### Best-performing model in the current project

The strongest model by current reported `Mean Dice` is:

- checkpoint: `unet_multiclass_t2f_t1c_t2w/best_model.pt`
- target mode: `multiclass`
- input modalities: `t2f`, `t1c`, `t2w`
- test loss: `0.2224`
- Dice WT: `0.9369`
- Dice TC: `0.9519`
- Dice ET: `0.8997`
- Mean Dice: `0.9295`

Therefore, the final conclusion from the current experiments is:

- the best input choice is the selected 3-modality combination `t2f + t1c + t2w`
- the best output formulation in the current runs is the standard `multiclass` setup
- `t1n` alone is clearly insufficient, and adding `t1n` back to all-modality input did not beat the best 3-modality rerun

## 7.5 Loss Curve File Mapping

The generated loss-curve PNG files can be mapped to input modalities as follows:

| Input modalities | Target mode | PNG file | Training log | Notes |
|---|---|---|---|---|
| `t1n` | `multiclass` | `loss_curve_multiclass_t1n.png` | `3648916.out` | single-modality baseline |
| `t1c` | `multiclass` | `loss_curve_multiclass_t1c.png` | `3648915.out` | single-modality baseline |
| `t2f + t1c + t2w` | `multiclass` | `loss_curve_multiclass_t2f_t1c_t2w.png` | `3652106.out` | later rerun of the same 3-modality multiclass setting |
| `t2f + t1c + t2w` | `regions` | `loss_curve_regions_t2f_t1c_t2w.png` | `3648048.out` | region-based output run |
| `t1n + t1c + t2w + t2f` | `multiclass` | `loss_curve_multiclass_all_modalities.png` | `3648917.out` | all-modality multiclass run |

The same mapping can also be read from the existing PNG names:

- `loss_curve_multiclass_t1n.png` corresponds to input modality `t1n`
- `loss_curve_multiclass_t1c.png` corresponds to input modality `t1c`
- `loss_curve_multiclass_t2f_t1c_t2w.png` corresponds to input modalities `t2f + t1c + t2w`
- `loss_curve_multiclass_all_modalities.png` corresponds to input modalities `t1n + t1c + t2w + t2f`
- `loss_curve_regions_t2f_t1c_t2w.png` corresponds to input modalities `t2f + t1c + t2w`

### Notes on naming

- `loss_curve_multiclass_t2f_t1c_t2w.png` is the retained 3-modality multiclass figure for input `t2f + t1c + t2w`
- `loss_curve_multiclass_all_modalities.png` is the retained all-modality figure for input `t1n + t1c + t2w + t2f`

## 8. Implementation Summary

The following project files were added or updated to support this setup:

- `data.py`
  - dataset for `processed_2d`
  - modality selection
  - multiclass and region targets
  - black-slice removal
  - patient-level split utilities
  - training augmentation pipeline
- `models/simple_unet.py`
  - plain 2D UNet with concatenated input channels
- `builders.py`
  - model builder updated to support `simple_unet`
- `losses.py`
  - keeps `dice_ce` for multiclass segmentation
- `config.py`
  - model, scheduler, and augmentation configuration
- `train.py`
  - command-line training entry point
  - patient-level split loading
  - scheduler and augmentation argument control
- `engine.py`
  - training loop
  - validation-based `ReduceLROnPlateau` stepping
  - automatic multi-GPU support with `DataParallel`
- `.job` files
  - cluster submission scripts for multiclass and region-based runs

## 9. Final Rationale

This setup was chosen as a strong and simple baseline:

- use the three most informative modalities
- avoid unnecessary architectural complexity
- use a standard UNet
- use a standard `CrossEntropy + Dice` loss
- use an adaptive validation-based learning rate decay strategy
- apply practical augmentation while protecting against crop-induced information loss

This makes the method easy to explain, easy to reproduce, and appropriate for a course project or baseline experiment report.

# Master's Thesis: Determination of Stress Concentration Factors of Welded Joints from Surface Scans using Artificial Neural Networks

This repo includes the [written thesis](Masterthesis_Fraunhofer_Oener_Aydogan.pdf), the datasets, and the Python scripts used for data generation and the training and evaluation of the KNNs.  
This work utilizes the Pointnet++ model ([Qi](https://arxiv.org/abs/1706.02413)) and the 2DLaserNet model ([Kaleci](https://www.sciencedirect.com/science/article/pii/S2215098621001397?ref=pdf_download&fr=RR-2&rr=7fb27b4caa082c0d)).  
The implementation of the PointNet++ model is based on the PyTorch implementation by [Xu Yan](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

## Adjustment of ML Models

Both models, originally intended for classification, were adapted for regression by changing the last layer to output only one value that describes the stress concentration factor instead of k-class values. Additionally, the loss function was modified (you can choose between MAE loss, R2 loss, and MSE loss).

For the Pointnet++ model, the last layer:

```python
self.fc3 = nn.Linear(256, num_class)
```

was changed to include an additional linear layer:

```python
self.fc3 = nn.Linear(256, 128)
self.bn3 = nn.BatchNorm1d(128)
self.drop3 = nn.Dropout(0.5)
self.fc4 = nn.Linear(128, 1)
```

Since 2DLaserNet is designed for point clouds with a fixed number of points, all point clouds are padded with zeros when loading the data and extended to the length of the point cloud with the most points.

## Setup

Python 3.8 and Conda are used. To install the necessary libraries and Python version using the [environment file](environment.yml):

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate scf-knn
```

## Folder Structure

### Data Generation
In the folder [data_generation](data_generation), you can find the data and information for the artificial generation of point clouds and Abaqus models of butt and T-joints.

### Datasets
In the folder [data](data), you can find the datasets used for this work:
- artificial_point_clouds_in_cm:
  - param_field_0: artificially generated data from butt joints
  - param_field_1: artificially generated data from T-joints
- weld_scans_2d_cuts_rotated_in_cm:
  real data from scanned T-joints from the work of Matthias Jung (see [Contact](#contact))

### Training and Validation Sets
In the folder [train_test_sets](train_test_sets), you can define `train_set.txt` and `test_set.txt` files that contain the filenames of the samples used for training or evaluating the KNNs.

### Other Methods for Determining Stress Concentration Factors
In the folder [scf_calc_methods](scf_calc_methods), you can find scripts with methods for determining stress concentration factors from the works of Anthes, Rainer, Kiyak, and Oswald.

## Training

To train a model:

```bash
python train_regression.py
```

By default, the Pointnet++ model is used. Alternatively, the model can be selected with the `--model` parameter from the following options:
  - pointnet2_regr_msg
  - lasernet_regr

For training, a dataset and a train & test set are used, which can be specified via the `--data_path` and `--set_path` parameters. `--set_path` represents the path to the folder where the training and test sets are defined as text files (see [Explanation](#training-and-validation-sets)). The model is trained on the training set and evaluated on the test set at each epoch. The results of the best evaluation are stored in [log/regression](log/regression).

To see all optional arguments for training:

```bash
python train_regression.py -h
```

The model with the best test score will be saved in [log/regression](log/regression) under `checkpoints/best_model.pth`.

### Training with Cross-Validation

To train and evaluate the model with cross-validation, the option `--cv <number of folds>` can be used. The training set is divided into `<number of folds>`.

## Evaluation

To evaluate an existing model:

```bash
python test_regression.py --log_dir <path to the experiment folder in the log/regression directory>
```

The evaluation results will be saved in [test_results](test_results).

To see all optional arguments for evaluation:

```bash
python test_regression.py -h
```

## Logs

The logs of the training or evaluation can be found in [log/regression](log/regression).

## Example Scripts

Training Pointnet++:

```bash
python train_regression.py --use_cpu --model pointnet2_regr_msg --epoch 1
```

Training 2DLaserNet:

```bash
python train_regression.py --use_cpu --model lasernet_regr --epoch 1
```

Evaluating Pointnet++:

```bash
python test_regression.py --log_dir 2023-08-21_21-54_a4d9e8d5-e5f9-462e-8b9b-d9f0f7b60a4d --use_cpu --model pointnet2_regr_msg
```

Evaluating 2DLaserNet:

```bash
python test_regression.py --log_dir 2023-08-21_21-58_cc9f1d94-4487-46d1-9453-1e8addced9f0 --use_cpu --model lasernet_regr
```

## Trained Models

In the folder [log/regression/best_models](log/regression/beste_modelle), you can find already trained models from the master's thesis. The models were trained on the default train/test set of the corresponding dataset with seed 4567. The best models on the 3 datasets weld_scans, apc_butt_joints, and apc_T_joints have been stored.

| Model      | Trained on Dataset             | Folder Name                                                       | R2 Score | Learning Rate | Decay Rate |
|------------|--------------------------------|-------------------------------------------------------------------|----------|---------------|------------|
| Pointnet++ | weld_scans                    | thesis_exp7_2023-07-29_22-20_2d6ac1cd-9073-4872-afd4-96a6c7332a6f | 0.443451 | 0.01          | 0.0001     |
| Pointnet++ | apc_butt_joints 0.025 mm      | thesis_exp7_2023-07-30_05-09_77ff558a-344d-4832-82ea-c19c88d9012a | 0.973569 | 0.001         | 0.001      |
| Pointnet++ | apc_T_joints 0.025 mm         | thesis_exp7_2023-07-30_07-44_84121dbd-797c-465b-bb14-68ebc46644d3 | 0.990126 | 0.001         | 0.001      |
| 2DLaserNet | weld_scans                    | thesis_exp7_2023-07-30_04-03_a67711ed-bd3f-45ee-b75f-4346a6603e22 | 0.700938 | 0.001         | 0.001      |
| 2DLaserNet | apc_butt_joints 0.025 mm      | thesis_exp7_2023-07-30_00-29_3b64beaf-80be-458e-9b3c-426f12b8eee5 | 0.985637 | 0.0001        | 0.0001     |
| 2DLaserNet | apc_T_joints 0.025 mm         | thesis_exp7_2023-07-30_03-28_dea08961-10b9-489f-80d3-a6451ad85380 | 0.837963 | 0.0001        | 0.0001     |

## Contact
For questions and comments to the author of this master's thesis:
- Öner Aydogan (oner_996@web.de)

Supervisors at Fraunhofer:
- Jan Schubnell (jan.schubnell@iwm.fraunhofer.de)
- Matthias Jung (matthias.jung@iwm.fraunhofer.de)

# CenterNetImpl
A rudimentary centernet implementation based on the **object as point** paper: https://arxiv.org/pdf/1904.07850

implementation made in pytorch and pytorch-lightning

logs using TensorBoard

#### Dataset

the dataset was downloaded from here: https://universe.roboflow.com/james-bqgdn/playing-card-detection-8ejwj
see cardDetectionDataset/README.md for more info

#### Run the code

- pip the requirement.txt if needed

- change hyperparameters and environnement parameters directly in the **train.py** file (L299 - L328)

then 
```bash
python train.py
```

or use the trainTorch.py file for pure pytorch script version



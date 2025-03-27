# Video Label Refinement
The goal of this project to refine labels for temporal localization datasets.  The dataset that are currently supportrf are ShakeFive2 and UT-interaction

## Components
This project depends on OpenMMLab https://openmmlab.com/.  It includes redristributed files from those libaries which are allowed under https://github.com/openmm/openmm/blob/master/libraries/sfmt/LICENSE.txt, provided the following copyright is included:
Copyright (c) 2006,2007 Mutsuo Saito, Makoto Matsumoto and Hiroshima University. All rights reserved.

- File downloader: download.py
- Generate signal data: inference.py
- View signal data: main.py
- Create refinement template: autorevise_template.py
- Label refinement: refinement.py
- Classification assessment: train_model.py

## How to run
This project depends on PyTorch and Torch Vision, which are not installed automatically.  THe project uses mmpose from OpenMMLab.  Please use the installation instructions from OpenMMLab to use the signal extraction method in this project: https://mmpose.readthedocs.io/en/latest/installation.html 

### Download the checkpoint files
The checkpoint files were too large to include in the repository
From the within the project director, run
```shell
python downloader.py
```

### Download the datasets
The ShakeFive2 dataset cannot be redistributed and has be downloaded separately.
Download the ShakeFive2 dataset from https://www.projects.science.uu.nl/shakefive/

The UT-interaction dataset cannot be redistributed and has be downloaded separately.
Download the ShakeFive2 dataset from https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html


### Run Signal Generation
The first step is to generate signal data.  Run the command: python inference.py --path=./PathToVideos
```shell
python inference.py --path=./UT-interaction
```

### Create a template of properties and refine labels for UT-interaction
```shell
python autorevise_template.py 
```
Output is found in autorevise_results.txt

### Classification Assessment
Classification improvement is assessed by training models on two different sets of labels. 
This program requires a parameter to indicate whether to used the refined or human labels.  
To use the refined labels, set the parameter refined=True, to used the unmodified labels set refined=False.

```shell
python inference.py --path=./ShakeFive2
python refinement.py --path=./ShakeFive2
python train_model.py --refined=False --train=True
```
The output is in model_performance.csv


### Visualizing Signal Data
Visualizing videos, pose and signal data is helpful for understanding the results.  
Run this command:
```shell
python main.py
```


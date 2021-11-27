# 3DGazeFollowing
This repository contains the evaluation codes for our: We know where they are looking at from the RGB-D camera: Gaze following in 3D

## Introduction
We released the code and part of dataset for evaluation and visualization.

## Prerequisites
- Python>=3.5.0
- Pytorch>=1.5.0
- torchvision>=0.6.0
- numpy>=1.14.2
- open3d>=0.7.0 (Visualization needs)
- tqdm>=4.50.2

## Instruction

1. Download the [dataset](https://drive.google.com/file/d/1jLhCFgRA6GJqS7-HSHqgdGN6UYiJQcOO/view?usp=sharing) we provide.
2. Download the the [model weight](https://drive.google.com/file/d/1hQpLMxndLEb2RhLpK9Ahn0NMdmQJ-4Cn/view?usp=sharing) we provide.
3. Clone our code to the local device.

    ```
    git clone https://github.com/nkuhzx/3DGazeFollowing.git
    ```
4. Unzip the downloaded dataset file and put it in the project root directory
5. Move the model weight file to the "modelpara" folder.
6. run the evaluation code (Default on GPU)

    ```
    python evaluation.py --visualize
    ```
   
   or only evaluate without visualization
   
    ```
    python evaluation.py 
    ```  
   
   or only run on CPU

    ```
    python evaluation.py --cpu
    ```     

7. After the evaluation, the average angle error and average distance error will be given.

8. In the visualization, the red line represents the Ground Truth, and the green line represents the predicted gaze line.

## Ackonwledgement

We thank the reviewers for their constructive suggestions.



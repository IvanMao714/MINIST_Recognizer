# MINIST Recognizer
<p align="center">
    <img src="https://img.shields.io/badge/Python-yellow?style=for-the-badge&logo=python&logoColor=%233776AB
    ">
    <img src="https://img.shields.io/badge/Pytorch-blue?style=for-the-badge&logo=pytorch&logoColor=%23EE4C2C
    ">
    <img src="https://img.shields.io/github/last-commit/IvanMao714/Optimal-sample-selection?logoColor=blue&style=for-the-badge"/>
</p>

## Abstract
This project focuses on developing a highly accurate model for recognizing handwritten
digits using the well-known MINIST dataset. The dataset comprises 2,400 unique images for
each digit (0 to 9) and serves as a fundamental resource in the realms of computer vision
and machine learning. The proposed solution introduces a Convolutional Neural Network
(CNN) model, known for its success in image recognition by mimicking human visual system
functionality to identify patterns and features.
To optimize performance, the project adopts a strategy of fine-tuning the CNN’s architecture.
This involves adjusting the parameters of convolutional layers, pooling layers, and
fully connected layers to enhance the model’s ability to extract features and improve recognition
results. Additionally, the project employs data augmentation techniques during training,
incorporating rotations and translations to diversify the dataset and improve the model’s generalization
on new, unseen data. This not only increases sample diversity but also mitigates
overfitting.
Key metrics such as accuracy and loss functions play a crucial role during training. Accuracy
reflects the model’s proficiency in recognizing digits, while the loss function measures
the disparity between predicted and actual values. Continuous optimization of these metrics
throughout training leads to gradual improvements in the model’s performance.

## Installation
- pip install -f environment.yml 
- run *main.py* file



## Methodology

### Structure of CNN
<img src="./img/1 (2).png" width="70%">

### Structure of DRNN
<img src="./img/4 (1).png" width="100%">

### Data Augment

#### Data Transfermation
These techniques include image rotation and translation operations. The purpose of this
approach is to increase the model’s robustness, enabling it to better adapt to input images at
various angles, scales, and orientations, thus improving overall prediction accuracy.

<img src="./img/mnist_compose.png" width="90%">

#### Normalization
During training, this project also applies a normalization process to the images. To be more
precise, both the mean and standard deviation are set to 0.5, ensuring that the results fall within
the range of [-1, 1].

The number of augmented images increases to between approximately 5,500
and 7,000.

<img src="./img/mnist_count.png" width="90%">

## Performance
<p align="center">
    <img src="./img/mnist_acc_comparison.png" width="45%">
    <img src="./img/mnist_loss_comparison.png" width="45%">
</p>
<p align="center">
    <img src="./img/mnist_test_comparison.png" width="45%">
    <img src="./img/mnist_train_comparison.png" width="45%">
</p>
<p align="center">
    <img src="./img/mnist_train_comparison_bar.png" width="45%">
    <img src="./img/mnist_train_loss_comparison_bar.png" width="45%">
</p>

## Contributing

<table>
  <tr>
    <td align="center"><a href="https://github.com/IvanMao714"><img src="https://avatars.githubusercontent.com/u/72293808?s=400&u=4fab4e9793c14e354fea9adf888a6965526e2281&v=4" width="100px;" alt=""/><br /><sub><b>@IvanMao714</b></sub></a></td>
    <td>
        <ol>
            <li> Design DRNN and CNN models (90 lines of the code)
            <li> Data Augment Class(60 lines of the code)
            <li> Data Processing (80 lines of the code)
            <li> Code of the training model (100 lines of the code)
        </ol>
    </td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/Zachary886"><img src="https://avatars.githubusercontent.com/u/126328673?v=4" width="100px;" alt=""/><br /><sub><b>@Zachary886</b></sub></a>
    <td>
        <ol>
            <li> Generation of mode structure diagrams (60 lines of the code)
            <li> Generation of charts for reporting (90 lines of the code)
            <li> Report written 70%
            <li> Presentation
        </ol>
    </td>
  </tr>
</table>

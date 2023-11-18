![](images/Barcode_PNG.png)

# Polyelectrolytes recognition

![GitHub](https://img.shields.io/github/license/ShockOfWave/polyelectrolytes_recognition)
![GitHub last commit](https://img.shields.io/github/last-commit/ShockOfWave/polyelectrolytes_recognition)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ShockOfWave/polyelectrolytes_recognition)
![contributors](https://img.shields.io/github/contributors/ShockOfWave/polyelectrolytes_recognition) 
![codesize](https://img.shields.io/github/languages/code-size/ShockOfWave/polyelectrolytes_recognition)
![GitHub repo size](https://img.shields.io/github/repo-size/ShockOfWave/polyelectrolytes_recognition)

# Introduction

This work is a [continuation](https://github.com/ShockOfWave/barcode) of research on the application of topological data analysis algorithms to atomic force microscopy data. This project created a classification model capable of determining the number of polyelectrolyte bilayers from AFM data. TDA methods were used to increase the quality and quantity of the training set. We used CNN and a perceptron based on the PyTorch framework. Optimization was carried out using Optuna.
You can find out more about the use of TDA for real objects from material and surface sciences in the articles by Skorb, Nosonovsky, Zhukov, Aglikov, and Aliev. Links will be added later, as articles are published.
You can find out more about the use of TDA for real objects from material and surface sciences in the articles by Skorb, Nosonovsky, Zhukov, Aglikov, and Aliev. Links will be added later, as articles are published. Details and description of the approach will be given in the article in the near future.

# Installation and Setup

Clone project
```bash
git clone github.com/ShockOfWave/barcode
```

To install dependencies run
```bash
pip install -r requirements.txt
```

To start data generation and training you can run
```bash
python -m src
```

If you want only generate data run

```bash
python -m src.data.make_dataset
```

If you want to only train model run

```bash
python -m src.train.train_model
```

if you want to optimize model run

```bash
python -m src.train.tune_model_optuna
```

# Data

Data were collected using atomic force microscopes. Additional data was obtained using TDA analysis, as well as the code provided in our previous [repository](https://github.com/ShockOfWave/barcode).

# Results and evaluation

### Model structure

![alt text](images/multimodal_classifier.onnx.svg)

We used 3 input convolutional heads for images and 3 input heads based on the perceptron model, after which we combined them into one perceptron with an activation function for our task.

# Future work
Further potential work is associated with the use of neural networks for real-time analysis of images from an atomic force microscope.

# Acknowledgments/References
We thank the [Infochemistry Scientific Center ISC](infochemistry.ru) for the provided data and computing power.

# Reference & Citation

The authors are more then happy if you refer the following work:

```tex
@article{Aglikov2023,
  title = {Topological Data Analysis of Nanoscale Roughness of Layer-by-Layer Polyelectrolyte Samples Using Machine Learning},
  ISSN = {2637-6113},
  url = {http://dx.doi.org/10.1021/acsaelm.3c01358},
  DOI = {10.1021/acsaelm.3c01358},
  journal = {ACS Applied Electronic Materials},
  publisher = {American Chemical Society (ACS)},
  author = {Aglikov,  Aleksandr S. and Aliev,  Timur A. and Zhukov,  Mikhail V. and Nikitina,  Anna A. and Smirnov,  Evgeny and Kozodaev,  Dmitry A. and Nosonovsky,  Michael I. and Skorb,  Ekaterina V.},
  year = {2023},
  month = nov 
}
```

# License
The code is distributed under the [MIT license](https://opensource.org/license/mit/).

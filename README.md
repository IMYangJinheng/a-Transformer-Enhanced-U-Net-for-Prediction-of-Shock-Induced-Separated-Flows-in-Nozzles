# Rapid Prediction of Shock-Induced Separated Flows in Over-Expanded Nozzles with a Transformer-Enhanced U-Net
This repository provides the implementation of a Transformer-enhanced U-Net framework for rapid prediction of shock-induced separated flows in over-expanded nozzles.

The model learns a mapping from geometry-aware nozzle representations to multi-physics flow fields, including pressure, temperature, and velocity. The goal is to provide a fast surrogate model for design-oriented nozzle flow analysis while retaining the ability to capture key flow structures such as shock waves, expansion regions, and shock-induced separation.

![Overview](asset/FIG_1.png)

## 🚀 Highlights

* Transformer-enhanced U-Net architecture for nozzle internal flow-field prediction
* Geometry-aware inputs based on signed distance field (SDF) and identifier matrix (IM)
* Simultaneous prediction of pressure, temperature, and velocity fields
* Improved reconstruction of shock-induced separation regions compared with a conventional U-Net
* Fast inference for complete flow-field prediction after offline training
* Support for wall pressure and thrust evaluation based on predicted flow fields

## 🛠️ Model Architecture

![Model Architecture_1](asset/FIG_2.png)

![Model Architecture_2](asset/FIG_3.png)


## 📊 Results
![Result_1](asset/FIG_4.png)
![Result_2](asset/FIG_5.png)
![Result_3](asset/FIG_6.png)

## 📧 Contact
Since our paper is currently under review, the detailed code and dataset will be uploaded after the paper is accepted. If you need the code and dataset recently, please contact us: Jinheng Yang: 124101022118@njust.edu.cn

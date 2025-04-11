# NGNN
####  A Natural Growth Model for Predicting Physicochemical Property of Complex Systems.
For more design concepts and details of the model, please refer to Article  ***A Universal Framework for General Prediction of Physicochemical Properties: the Natural Growth Model***

# Overview

Here are the details about the model.

## Code running conditions

#NGNN is implemented using Pytorch and runs on windows11 with NVIDIA GeForce RTX 3060 graphics processing units,which relies on Pytorch Geometric.

The following are the required Python libraries to be installed：numpy、pandas、rdkit、sklearn.

## Data preparation
If you need to use your own dataset for prediction, simply prepare the required molecular smiles format and its properties for prediction, and then process the data into the same format as the data we provide.

## Code content
|name |content |
|----------------|--------------------------------|
|**NGNN.m** | NGNN code used to predict multi solvent molecular properties. |
|**msMPNN.m** |NGNN code used to predict single solvent molecular properties.|

## Data content
|name |content |
|----------------|--------------------------------|
|**fanjinming.csv** | Absorption wavelength data. |
|**fanjinming1.csv** |Solubility data.|
|**data_E.csv** | Environmental features (solvent). |
|**data_G.csv** |Solvational interaction parameter.|

## Different Model Prediction Results (MAE)

|model |solubility |lipophilicity |IE |abs |EM |PLQY |
|----------------|----------|-------|----------|------|-------|------|
|**GCN** | 0.457 | 0.495| 0308 | 21.194 | 25.252 | 0.162 |
|**GAT** | 0.464 | 0.658| 0387 | 30.967 | 28.825 | 0.187 |
|**GBRT** | 0.447 | 0.490| 0281 | 14.17 | 19.676 | 0.127 |
|**RF** | 0.410 | 0.517| 0342| 13.069 | 19.176 | 0.126 |
|**NGNN** | 0.428 | 0.467| 0275 | 12.017 | 17.339 | 0.123 |

## Continuously updated
We will continue to update the data and models in the future.If you have any questions, please contact us. The contact information can be found in our paper.




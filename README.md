# NGNN
####  A Natural Growth Model for Predicting Physicochemical Property of Complex Systems.
For more design concepts and details of the model, please refer to Article  ***A Universal Framework for General Prediction of Physicochemical Properties: the Natural Growth Model***

# Overview

Here are the details about the model.

## Code running conditions

NGNN is implemented using Pytorch and runs on windows11 with NVIDIA GeForce RTX 3060 graphics processing units,which relies on Pytorch Geometric.


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




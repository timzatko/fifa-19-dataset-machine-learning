# FIFA 19 Dataset Analysis + Prediction And Classification Tasks

Authors: [Timotej Zaťko](https://github.com/timzatko), [Tomáš Hoffer](https://github.com/tomhoffer)

In this work we aimed to predict aimed a football player's value and their game position on the FIFA 19 dataset which is publicly available on [kaggle](https://www.kaggle.com/karangadiya/fifa19). We did complete data analysis of player's data which was followed by data preprocessing and then by training the model. We tried to use only player's skill based attributes (finishing, sprint speed, heading...) and physical attributes (age, height, weight...) in our models.

## Setup

### Prerequisites

- [Docker](https://www.docker.com/)

### Instructions

1. Get into project root repository
2. Build a docker image -- `sh ./build.sh`
3. Run the docker container using command -- `sh ./run.sh`

## Repository structure

- `notebooks` - contains jupyter notebooks with data analysis, preprocessing...
    - [01_initial_analysis.ipynb](notebooks/01_initial_analysis.ipynb) - quick & brief overview of the dataset, number of examples, collumns, data types, missing values...
    - [02_preprocessing_for_analysis.ipynb](notebooks/02_preprocessing_for_analysis.ipynb) - conversion of date type attributes, weight, height a the money amounts for the main analysis
    - [03_analysis.ipynb](notebooks/03_analysis.ipynb) - main analysis dataset analysis
    - [04_preprocessing_for_prediction.ipynb](notebooks/04_preprocessing_for_prediction.ipynb) - dataset preprocessing for model training (for both tasks - classification of player's position and value prediction)
    - [05_model_selection_classification.ipynb](notebooks/05_model_selection_classification.ipynb) - comparison of some basic classification models (also we defined our baseline model)
    - [06_model_selection_regression.ipynb](notebooks/06_model_selection_regression.ipynb) - comparison of some basic regression models (also we defined our baseline model)
    - [07_classification_NN.ipynb](notebooks/07_classification_NN.ipynb) - classification approach using neural network
    - [09_SMOTE.ipynb](notebooks/09_SMOTE.ipynb) - data oversampling using several methods including SMOTE and its variations
    - [10_feature_selection.ipynb](notebooks/10_feature_selection.ipynb) - comparison of several feature selection approaches
    - [11_ensemble_classification.ipynb](notebooks/11_ensemble_classification.ipynb) - comparison of several ensemble models for the classification task
    - [12_ensemble_regression.ipynb](notebooks/12_ensemble_regression.ipynb) - comparison of several ensemble models for the regression task
    - [13_hyper_parameter_tuning.ipynb](notebooks/13_hyper_parameter_tuning.ipynb) - hyper-parameter tunning on our best models using grid search
- `src` - contains helper functions and classes for preprocessing/analysis/evaluation...
- `data` - contains FIFA 19 dataset
- `report` - LaTex files for final report generation, the report is written in Slovak
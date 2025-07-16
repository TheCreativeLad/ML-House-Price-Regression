# ML-House-Price-Regression
Solution to the Kaggle 'Home Data for ML Course' competition, predicting house prices using Random Forest.").

# Project Title: House Price Prediction (Kaggle Competition Exercise)

## Overview
This repository contains my solution to the "Home Data for ML Course" Kaggle competition, focusing on predicting house prices in Ames, Iowa. This project served as an excellent introduction to applying core machine learning concepts to a real-world regression problem.

## Problem Statement
The goal is to accurately predict the sales price of residential homes based on a comprehensive set of features, providing insights into the factors that influence property value.

## Methodology
* **Data Loading & Exploration:** Initial loading of training and test datasets.
* **Feature Selection:** Careful selection of relevant numerical features to train the model, including `LotArea`, `YearBuilt`, `1stFlrSF`, `2ndFlrSF`, `FullBath`, `BedroomAbvGr`, `TotRmsAbvGrd`, `MSSubClass`, `OverallQual`, `OverallCond`, `LowQualFinSF`, `GrLivArea`, `HalfBath`, `Fireplaces`, and `MiscVal`.
* **Data Splitting:** Division of the training data into training and validation sets to evaluate model performance.
* **Model Training:** A **Random Forest Regressor** model was trained on the processed features.
* **Model Evaluation:** Performance was assessed using Mean Absolute Error (MAE) on the validation set.
* **Prediction Generation:** Predictions were generated for the unseen test dataset.

## Technologies Used
* **Python**
* **Pandas** (for data manipulation)
* **Scikit-learn** (for `RandomForestRegressor`, `train_test_split`, `mean_absolute_error`)
* **Jupyter Notebook** (`.ipynb` format)

## Project Files
* `exercise-machine-learning-competitions.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, evaluation, and prediction.
* `submission.csv`: The generated submission file for the Kaggle competition.
*(Note: The `train.csv` and `test.csv` datasets are part of the Kaggle competition environment and are not directly included here due to size, but are accessible via the Kaggle platform.)*

## Results & Learnings
The Random Forest Regressor achieved a validation MAE of approximately 17,820, demonstrating a solid baseline for house price prediction. This exercise provided hands-on experience with a complete ML pipeline, emphasizing the importance of feature engineering and robust model evaluation. It laid the groundwork for exploring more advanced techniques and feature handling.

## Future Enhancements
* Explore more advanced feature engineering techniques (e.g., handling categorical features, missing values).
* Experiment with other regression models (e.g., Gradient Boosting, XGBoost for improved accuracy).
* Hyperparameter tuning to optimize model performance.

## How to Run
1.  Clone this repository: `git clone https://github.com/your-username/your-repo-name.git`
2.  Navigate to the project directory: `cd your-repo-name`
3.  Ensure you have Python and the required libraries (pandas, scikit-learn) installed.
4.  Open the notebook: `jupyter notebook exercise-machine-learning-competitions.ipynb`
5.  Download the `train.csv` and `test.csv` datasets from the [Kaggle Home Data for ML Course competition page](https://www.kaggle.com/c/home-data-for-ml-course/data) and place them in the appropriate `input` directory as referenced in the notebook, or modify the paths.

---
*Connect with me on [LinkedIn](https://www.linkedin.com/in/your-linkedin-profile/) for more insights on Data Science, Machine Learning, and Leadership!*

# Rain in Australia
In this project for MLOps we're working with this dataset for apply the acquire in class knowlegge.

## About Dataset
This dataset contains about 10 years of daily weather observations from many locations across Australia. Predict next-day rain by training classification models on the target variable RainTomorrow.

RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more.

## Dataset Information
- Number of Instances: 127,536
- Number of Features: 23
- Target Variable: RainTomorrow

## Solution
In this project we will show we will work with a notebook downloaded from the Kaggle platform, in this solution they propose to use an artificial neural network. The process includes loading data, visualizing the data, cleaning the data, preprocessing the data and building the model.

In addition to the above, we will work with the code to apply refactorization, docstrings, annotation and some other best practices in a correct management of a MLOps project. The project going to up to github platform for colaborate with others parthers if that is necesary.


# Steps to run project in windows
## Activate virtual environment
1. Create a virtual environment with `Python 3.10+`
    * Create venv
        ```bash
        python3.10 -m venv venv_project
        ```

    * Activate the virtual environment
        ```
        venv_project\Scripts\activate
        ```
    * Install the packages
        ```bash
        pip install -r requirements.txt
        ```



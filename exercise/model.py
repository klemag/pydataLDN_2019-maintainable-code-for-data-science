import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import DATA_DIR, DTYPES
from transformers import (CategoriesExtractor, CountryTransformer, GoalAdjustor,
                          TimeTransformer)


def load_dataset(x_path, y_path):
    x = pd.read_csv(os.sep.join([DATA_DIR, x_path]),
                    dtype=DTYPES,
                    index_col="id")
    y = pd.read_csv(os.sep.join([DATA_DIR, y_path]))

    return x, y


def build_model():
    cat_processor = Pipeline([("transformer", CategoriesExtractor()),
                              ("one_hot",
                               OneHotEncoder(sparse=False,
                                             handle_unknown="ignore"))])

    country_processor = Pipeline([("transformer", CountryTransformer()),
                                  ("one_hot",
                                   OneHotEncoder(sparse=False,
                                                 handle_unknown="ignore"))])

    # Add your code here to create the missing
    # ColumnTransformer and Pipeline
    # [...]


def tune_model():
    # Add your code here
    # [...]
    pass


def train_model(print_params=False):
    # Add your code here
    # [...]
    pass


def test_model():
    # Add your code here
    # [...]
    pass

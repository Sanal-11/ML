import os.path
import typing

import joblib
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.tree import DecisionTreeClassifier
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

GEN_MODEL = False
MODEL_FILE = 'depression.joblib'
CSV_DATASET = 'DataSets/Depression Professional Dataset.csv'


def create_model(x: DataFrame, y: DataFrame) -> list:
    """

    @param x:
    @param y:
    @return: list of file names
    """
    clf_model = DecisionTreeClassifier()
    clf_model.fit(x, y)
    return joblib.dump(value=clf_model, filename="depression.joblib")


def ManageDS(df: DataFrame) -> (DataFrame, DataFrame):
    """

    @param df:
    @return: _X(input df) :dataframe, _y(output df): dataframe
    """

    'to suppress drop warnings'
    pd.set_option('future.no_silent_downcasting', True)
    df.replace({'Male': 1, 'Female': 0}, inplace=True)
    _X = df[['Gender', 'Age']]
    _y = df[['Depression']]
    return _X, _y


def generate_model_file(input_ds: DataFrame) -> list:
    """

    @param input_ds: DataFrame
    """
    if GEN_MODEL:
        input_x, output_y = ManageDS(input_ds)
        model_path = create_model(input_x.values, output_y.values)
        logger.debug('Model generated!')
        logger.debug(model_path)
        return model_path

    return [MODEL_FILE]


def with_legacy_model(input_ds: DataFrame) -> DecisionTreeClassifier:
    """

    @param input_ds: DataFrame
    """
    input_x, output_y = ManageDS(input_ds)

    logger.debug(input_x)

    logger.debug(output_y)

    model = DecisionTreeClassifier()
    model.fit(input_x.values, output_y.values)
    return model


dls_model = None
ds = pd.read_csv(CSV_DATASET).sort_values('Age')
# logger.debug(ds.to_string())
logger.debug(ds[(ds['Gender'] == 'Male') & (ds['Age'] == 19)].to_string())
logger.debug(ds[(ds['Gender'] == 'Female') & (ds['Age'] == 19)].to_string())

# clf_model = with_legacy_model(ds)

if not os.path.exists(generate_model_file(ds)[0]):
    logger.info("file not found!")
else:
    dls_model = joblib.load(MODEL_FILE)

if dls_model != '':
    for age in range(1, 66):
        predict = dls_model.predict([[0, age], [1, age]])
        logger.debug(str(age) + ' : ' + predict)

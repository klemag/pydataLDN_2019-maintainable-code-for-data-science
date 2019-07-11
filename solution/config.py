DATA_DIR = '../data'
MODEL_NAME = 'model.joblib'
X_TRAIN = 'X_train.zip'
Y_TRAIN = 'y_train.zip'
X_TEST = 'X_test.zip'
Y_TEST = 'y_test.zip'

DTYPES = {
    'id': 'int64',
    'photo': 'str',
    'name': 'str',
    'blurb': 'str',
    'goal': 'float64',
    'slug': 'str',
    'disable_communication': 'bool',
    'country': 'str',
    'currency': 'str',
    'currency_symbol': 'str',
    'currency_trailing_code': 'bool',
    'deadline': 'int64',
    'created_at': 'int64',
    'launched_at': 'int64',
    'static_usd_rate': 'float64',
    'creator': 'str',
    'location': 'str',
    'category': 'str',
    'profile': 'str',
    'urls': 'str',
    'source_url': 'str',
    'friends': 'str',
    'is_starred': 'str',
    'is_backing': 'str',
    'permissions': 'str',
    'state': 'int64'
}

PARAMS = {
    'model__max_depth': 7,
    'model__min_samples_split': 20,
    'preprocessor__categories__transformer__use_all': True
}

GRID_PARAMS = {
    'model__max_depth': [5, 7, 9],
    'model__min_samples_split': [5, 10, 20],
    'preprocessor__categories__transformer__use_all': [False, True]
}

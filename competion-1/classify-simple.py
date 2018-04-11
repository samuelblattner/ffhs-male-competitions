from numpy import average
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from pandas import read_csv, Series, DataFrame

MODELS = [
    {
        'name': 'Decision Tree Classifier',
        'abbr': 'DTC',
        'is_classifier': True,
        'class': DecisionTreeClassifier,
        'req_ravel': True,
        'extra_args': {
            'max_depth': 9
        }
    },
    {
        'name': 'Decision Tree Regressor',
        'abbr': 'DTR',
        'class': DecisionTreeRegressor,
        'req_ravel': True,
        'extra_args': {
            'max_depth': 9
        }
    },
    {
        'name': 'Random Forest Classifier',
        'abbr': 'RFC',
        'is_classifier': True,
        'class': RandomForestClassifier,
        'req_ravel': True,
        'extra_args': {
            'max_depth': 9
        }
    },
    {
        'name': 'Random Forest Regressor',
        'abbr': 'RFR',
        'class': RandomForestRegressor,
        'req_ravel': True,
        'extra_args': {
            'max_depth': 9
        }
    },
    {
        'name': 'Random Forest Regressor',
        'abbr': 'RFR',
        'class': RandomForestRegressor,
        'req_ravel': True,
        'extra_args': {
            'max_depth': 9
        }
    },
    {
        'name': 'K-nearest Neighbors Classifier k=10',
        'abbr': 'KNC_10',
        'is_classifier': True,
        'class': KNeighborsClassifier,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 10
        }
    },
    {
        'name': 'K-nearest Neighbors Classifier k=20',
        'abbr': 'KNC_20',
        'is_classifier': True,
        'class': KNeighborsClassifier,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 20
        }
    },
    {
        'name': 'K-nearest Neighbors Classifier k=10, distance',
        'abbr': 'KNC_10_dist',
        'is_classifier': True,
        'class': KNeighborsClassifier,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 10,
            'weights': 'distance'
        }
    },
    {
        'name': 'K-nearest Neighbors Classifier k=20, distance',
        'abbr': 'KNC_20_dist',
        'is_classifier': True,
        'class': KNeighborsClassifier,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 20,
            'weights': 'distance'
        }
    },
    {
        'name': 'K-nearest Neighbors Regressor k=10',
        'abbr': 'KNR_10',
        'class': KNeighborsRegressor,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 10,
        }
    },
    {
        'name': 'K-nearest Neighbors Regressor k=20',
        'abbr': 'KNR_20',
        'class': KNeighborsRegressor,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 20,
        }
    },
    {
        'name': 'K-nearest Neighbors Regressor k=10 distance',
        'abbr': 'KNR_10_dist',
        'class': KNeighborsRegressor,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 10,
            'weights': 'distance'

        }
    },
    {
        'name': 'K-nearest Neighbors Regressor k=20 distance',
        'abbr': 'KNR_20_dist',
        'class': KNeighborsRegressor,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 20,
            'weights': 'distance'

        }
    },
    {
        'name': 'K-nearest Neighbors Regressor k=100 distance',
        'abbr': 'KNR_100_dist',
        'class': KNeighborsRegressor,
        'req_ravel': True,
        'extra_args': {
            'n_neighbors': 100,
            'weights': 'distance'

        }
    },
    {
        'name': 'Gaussian Naive Bayes Classifier',
        'abbr': 'GNB',
        'is_classifier': True,
        'class': GaussianNB,
        'req_ravel': True,
        # 'extra_args': {
        #     'n_neighbors': 100,
        #     'weights': 'distance'
        #
        # }
    },
]

train_data_frame = read_csv('data/input/FeatureTransformNR_train.csv')
test_data_frame = read_csv('data/input/FeatureTransformNR_test.csv')

XTrain = train_data_frame[['X1', 'X2']]
XTest = test_data_frame[['X1', 'X2']]
yTarget = train_data_frame[['y']]

for model in MODELS:
    print('\nEvaluating {}\n{}'.format(
        model.get('name', 'UNSPECIFIED NAME'),
        '=' * 35
    ))

    instance = model.get('class')(**model.get('extra_args', {}))
    fitted = instance.fit(XTrain, yTarget.values.ravel())
    predicted = instance.predict(XTest)
    x_predicted = cross_val_predict(instance, XTrain, yTarget.values.ravel(), cv=20)

    print('Simple score: {}'.format(instance.score(XTrain, yTarget.values.ravel())))
    print('Xval score: {}'.format(
        average(cross_val_score(
            instance, XTrain, yTarget.values.ravel() if model.get('req_ravel', False) else yTarget, cv=20
        )))
    )

    if model.get('is_classifier', False):
        print('Precision: {}'.format(precision_score(yTarget, x_predicted, average='weighted')))
        print('Recall: {}'.format(recall_score(yTarget, x_predicted, average='weighted')))

    series = Series(predicted, name='y', dtype=int)
    series.index.name = 'Id'
    DataFrame(series).to_csv('data/submissions/sub_sblattner_{}.csv'.format(model.get('abbr')))

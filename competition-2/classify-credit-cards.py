from numpy import average
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, scale, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt


from pandas import read_csv, Series, DataFrame

plot_dim_per_page = (10, 10)

MODELS = [

    {
        'name': 'Decision Tree Classifier',
        'abbr': 'DTC',
        'is_classifier': True,
        'class': DecisionTreeClassifier,
        'req_ravel': True,
        'extra_args': {
            'max_depth': 3
        },
        'param_grid': {
            'max_depth': [2, 3, 5, 7, 10]
        }
    },
    {
        'name': 'Random Forest Classifier',
        'abbr': 'RFC',
        'is_classifier': True,
        'class': RandomForestClassifier,
        'req_ravel': True,
        'extra_args': {
            'max_depth': 3,
            'n_estimators': 10
        },
        'param_grid': {
            'n_estimators': [10, 20],
            'max_depth': [3, 5, 7, 10,]
        }
    },
    {
        'name': 'Bagging Classifier Decision Tree',
        'abbr': 'BGCDCT',
        'is_classifier': True,
        'class': BaggingClassifier,
        'req_ravel': True,
        'extra_args': {
            'base_estimator': DecisionTreeClassifier(),
            'n_estimators': 10
        },
        'param_grid': {
            'base_estimator': [DecisionTreeClassifier()],
            'n_estimators': [10,]
        }
    },
    {
        'name': 'Adaptive Boost Classifier',
        'abbr': 'ADB',
        'is_classifier': True,
        'class': AdaBoostClassifier,
        'req_ravel': True,
        'extra_args': {
            'base_estimator': DecisionTreeClassifier(max_depth=3),
            'n_estimators': 10,
            'learning_rate': 0.2
        },
        'param_grid': {
            'base_estimator': [DecisionTreeClassifier(max_depth=3)],
            'n_estimators': [2, 5, 8],
            'learning_rate': [0.2, 0.4, 0.6]
        }
    },
    {
        'name': 'SVC',
        'abbr': 'SVC',
        'is_classifier': True,
        'class': LinearSVC,
        'req_ravel': True,
        'param_grid': {
            'penalty': ['l2',],
            'loss': ['squared_hinge'],
            'dual': [True, False],
            'C': [0.1, 0.2, 0.5, 0.75, 1]
        }
    },
    # {
    #     'name': 'K-nearest Neighbors Classifier k=10',
    #     'abbr': 'KNC_10',
    #     'is_classifier': True,
    #     'class': KNeighborsClassifier,
    #     'req_ravel': True,
    #     'extra_args': {
    #         'n_neighbors': 5
    #     },
    #     'param_grid': {
    #         'n_neighbors': [10, 20, ],
    #         'weights': ['distance', 'uniform']
    #     }
    # },
   # {
   #     'name': 'Gaussian Naive Bayes Classifier',
    ##    'abbr': 'GNB',
    #    'is_classifier': True,
    #    'class': GaussianNB,
    #    'req_ravel': True,
    #    'param_grid': {
    #        'n_neighbors': [2, 5, 10, 20, 50],
    #    }
    #},
]


def show_confusion_matrix(classifier, XTrain, y):
    yhat = cross_val_predict(classifier, XTrain, y, cv=10)
    conf_mx = confusion_matrix(y, yhat, labels=('+', 'x', 'o'))

    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()


train_data_frame = read_csv('data/input/credit-cards-train.csv')
test_data_frame = read_csv('data/input/credit-cards-test.csv')

XTrain = train_data_frame[train_data_frame.columns[1:23]]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL',]]
#XTrain = train_data_frame.loc[:, ['EDUCATION',]]
#XTrain = train_data_frame.loc[:, ['SEX',]]
#XTrain = train_data_frame.loc[:, ['MARRIAGE',]]
#XTrain = train_data_frame.loc[:, ['AGE',]]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL', 'SEX']]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL', 'SEX']]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']]
#XTrain = train_data_frame.loc[:, ['PAY_2', 'BILL_AMT1', 'PAY_AMT1']]
#XTrain = train_data_frame.loc[:, ['QUOTA', 'AVG_BILL', 'AVG_REPAY']]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
#XTrain = train_data_frame.loc[:, ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT6']]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
#XTrain = train_data_frame.loc[:, ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
#XTrain = train_data_frame.loc[:, ['PAY_0', 'BILL_AMT1', 'PAY_AMT1']]
XTest = test_data_frame[test_data_frame.columns[1:23]]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL',]]
#XTest = test_data_frame.loc[:, ['EDUCATION',]]
#XTest = test_data_frame.loc[:, ['SEX',]]
#XTest = test_data_frame.loc[:, ['MARRIAGE',]]
#XTest = test_data_frame.loc[:, ['AGE',]]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL', 'SEX']]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL', 'SEX']]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']]
#XTest = test_data_frame.loc[:, ['PAY_0', 'BILL_AMT1', 'PAY_AMT1']]
#XTest = test_data_frame.loc[:, ['QUOTA', 'AVG_BILL', 'AVG_REPAY']]
#XTest = test_data_frame.loc[:, ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT6']]
#XTest = test_data_frame.loc[:, ['AGE', 'SEX', 'PAY_0', 'BILL_AMT6', 'PAY_AMT6']]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
#XTest = test_data_frame.loc[:, ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
#XTest = test_data_frame.loc[:, ['PAY_0', 'BILL_AMT1', 'PAY_AMT1']]
yTarget = train_data_frame[['default.payment.next.month']]

factor = 3
# Enh_XTrain, Enh_yTarget = enhance_data_frame(XTrain, yTarget, factor=factor)
XTest = PCA(n_components=16).fit_transform((XTest))
Enh_XTrain = PCA(n_components=16).fit_transform((XTrain) )
Enh_yTarget = yTarget

for model in MODELS:

    print('\nEvaluating {}\n{}'.format(
        model.get('name', 'UNSPECIFIED NAME'),
        '=' * 35
    ))

    instance = model.get('class')(**model.get('extra_args', {}))
    grid_params = model.get('param_grid', None)
    if grid_params:
        clf = GridSearchCV(instance, grid_params, cv=factor * 4)
        clf.fit(Enh_XTrain, Enh_yTarget.values.ravel())
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

    fitted = instance.fit(Enh_XTrain, Enh_yTarget.values.ravel())
    x_predicted = cross_val_predict(instance, Enh_XTrain, Enh_yTarget.values.ravel(), cv=factor * 4)
    print('Simple score: {}'.format(instance.score(Enh_XTrain, Enh_yTarget.values.ravel())))
    print('Xval score: {}'.format(
        average(cross_val_score(
            instance, Enh_XTrain, Enh_yTarget.values.ravel() if model.get('req_ravel', False) else Enh_yTarget,
            cv=factor * 4
        )))
    )

    if model.get('is_classifier', False):
        print('Precision: {}'.format(precision_score(Enh_yTarget, x_predicted, average='weighted')))
        print('Recall: {}'.format(recall_score(Enh_yTarget, x_predicted, average='weighted')))

    predicted = instance.predict(XTest)

    series = Series(predicted, name='y', dtype=str)
    series.index.name = 'Id'
    DataFrame(series).to_csv('data/submissions/sub-creditcards-sblattner_{}.csv'.format(model.get('abbr')))

    # show_confusion_matrix(instance, Enh_XTrain, Enh_yTarget)

import os
from copy import copy
from os.path import join
from random import randint, random

import math

from numpy import average
from scipy.ndimage import shift
from skimage import img_as_bool
from skimage.transform import rotate, rescale
from skimage.util import pad
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from pandas import read_csv, Series, DataFrame

plot_dim_per_page = (20, 20)

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
        'name': 'Gaussian Naive Bayes Classifier',
        'abbr': 'GNB',
        'is_classifier': True,
        'class': GaussianNB,
        'req_ravel': True,
    },
]


def load_additional_images():

    yc = []
    dc = []

    for parent, dirs, files in os.walk('data/input/additional_images'):
        for file in files:

            data = read_csv(join(parent, file))
            y_col = data[data.columns[0]]
            data_cols = data[data.columns[1:101]]
            # print(data.iterrows())
            for y, data in zip(y_col.values, data_cols.values):
                # print(row)
                # print(y)
                y = y[0]

                # row[0] = row[0][0]
                yc.append(y)
                dc.append(data)

    # print(frames)
    return yc, dc


def show_plot(bitmaps):
    """
    Simple 5x6 plot matrix to show image results.
    :param bitmaps: {list} list of bitmaps to plot.
    """

    plots_per_page = plot_dim_per_page[0] * plot_dim_per_page[1]

    for page in range(0, int(math.ceil(len(bitmaps) / plots_per_page))):

        page_plot_offset = page * plots_per_page

        fig, axes = plt.subplots(ncols=plot_dim_per_page[0], nrows=plot_dim_per_page[1], figsize=(10, 10))
        ax = axes.ravel()

        for b in range(page_plot_offset, page_plot_offset + max(len(bitmaps) - page_plot_offset, page_plot_offset)):
            print(bitmaps[b])
            ax[b - page_plot_offset].imshow(bitmaps[b], cmap=plt.cm.gray)
            # ax[b - page_plot_offset].set_title(bitmaps[b][1], fontsize=8)

        for a in ax:
            a.axis('off')

        plt.show()


def enhance_data_frame(data_cols, y_cols, factor=2):
    """
    Use existing training data for +'s, x's and o's and multiply the data
    by altering it slightly with some basic imaging methods in order to
    diversify the data while keeping its classification as is.

    :param dataframe:
    :param factor:
    :return:
    """

    add_y_cols, add_data_cols = load_additional_images()

    y_cols = y_cols.append(DataFrame(add_y_cols, columns=list('y')), ignore_index=True)
    data_cols = data_cols.append(DataFrame(add_data_cols, columns=data_cols.columns), ignore_index=True)

    orig_data_cols = copy(data_cols)
    orig_y_cols = copy(y_cols)

    bitmaps = []

    if 0 < factor < 1000:

        for itr in range(1, factor):

            append_data = []

            for instance, y in zip(orig_data_cols.values, orig_y_cols.values):

                # Create 10 by 10 image
                img = instance.reshape(10, 10)

                # bitmaps.append(img)

                if y in ('x', '+', 'o'):

                    # Rotate by multiple of 90°
                    img = rotate(img, randint(0, 3) * 90, clip=True, preserve_range=True)

                    # Scale between 0.5 and 1.5
                    resized_img = img_as_bool(rescale(img, 0.8 + random() * 0.4, preserve_range=True))

                    if resized_img.shape[0] < 10:
                        resized_img = pad(resized_img, 10 - resized_img.shape[0], mode='constant')
                    if resized_img.shape[0] > 10:
                        edge1 = int((resized_img.shape[0] - 10) / 2)
                        edge2 = resized_img.shape[0] - ((resized_img.shape[0] - 10) - edge1)
                        resized_img = resized_img[edge1:edge2, edge1:edge2]

                    img = resized_img

                    # Flip randomly
                    flip = randint(0, 3)
                    if flip == 1:
                        img = img[:, ::-1]
                    elif flip == 2:
                        img = img[::-1, :]

                    # Shift by 1 pixel randomly
                    img = shift(img, [randint(-1, 1), randint(-1, 1)], cval=0)

                    # Rotate by random angle (o's only)
                    if y == 'o':
                        img = img_as_bool(rotate(img, 0 + random() * 360, clip=True, preserve_range=True))

                    # Apply Hough Transformation

                    # bitmaps.append(img)

                # Unravel back to 1D array and append data
                append_data.append([int(x) for x in img.reshape(100, 1)])

            y_cols = y_cols.append(DataFrame(orig_y_cols, columns=list('y')), ignore_index=True)
            data_cols = data_cols.append(DataFrame(append_data, columns=orig_data_cols.columns), ignore_index=True)

            # show_plot(bitmaps)

    return data_cols, y_cols


def show_confusion_matrix(classifier, XTrain, y):
    yhat = cross_val_predict(classifier, XTrain, y, cv=10)
    conf_mx = confusion_matrix(y, yhat, labels=('+', 'x', 'o'))

    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()


train_data_frame = read_csv('data/input/male-FFHS-train.csv')
test_data_frame = read_csv('data/input/male-FFHS-Xtest.csv')

XTrain = train_data_frame[train_data_frame.columns[1:101]]
XTest = test_data_frame[test_data_frame.columns[1:101]]
yTarget = train_data_frame[['y']]

Enh_XTrain, Enh_yTarget = enhance_data_frame(XTrain, yTarget, factor=1)

load_additional_images()
# print(Enh_XTrain[Enh_XTrain.columns[0:10]])


#
# col = []
#
# print(array(
#                 [math.radians(x) for x in range(-90, -80)] +
#                 [math.radians(x) for x in range(-50, -40)] +
#                 [math.radians(x) for x in range(-5, 5)] +
#                 [math.radians(x) for x in range(40, 50)] +
#                 [math.radians(x) for x in range(80, 90)]
#             )
#         )
#
# for inp in (XTrain, XTest):
#
#     data = []
#
#     for val in inp.values:
#         img = val.reshape(10, 10)
#         hspace, theta, d = hough_line(img
#         )
#         out = [0 for _ in range(0, len(theta) * 2)]
#
#         # for o, ot in enumerate(out[len(theta):]):
#         #     out[o] = -100
#
#         for t, th in enumerate(theta):
#             for h, hs in enumerate(hspace):
#                 if hs[t] > out[t] and hs[t] > 3:
#                     out[t] = hs[t]
#                     out[len(theta) + t] = d[h]
#
#         data.append(out)
#     col.append(data)
#
# XTrain_Hough = DataFrame(col[0])
# XTest_Hough = DataFrame(col[1])
#

for model in MODELS:

    print('\nEvaluating {}\n{}'.format(
        model.get('name', 'UNSPECIFIED NAME'),
        '=' * 35
    ))

    instance = model.get('class')(**model.get('extra_args', {}))
    fitted = instance.fit(Enh_XTrain, Enh_yTarget.values.ravel())
    predicted = instance.predict(XTest)
    x_predicted = cross_val_predict(instance, Enh_XTrain, Enh_yTarget.values.ravel(), cv=20)

    print('Simple score: {}'.format(instance.score(Enh_XTrain, Enh_yTarget.values.ravel())))
    print('Xval score: {}'.format(
        average(cross_val_score(
            instance, Enh_XTrain, Enh_yTarget.values.ravel() if model.get('req_ravel', False) else Enh_XTrain, cv=20
        )))
    )

    if model.get('is_classifier', False):
        print('Precision: {}'.format(precision_score(Enh_yTarget, x_predicted, average='weighted')))
        print('Recall: {}'.format(recall_score(Enh_yTarget, x_predicted, average='weighted')))

    series = Series(predicted, name='y', dtype=str)
    series.index.name = 'Id'
    DataFrame(series).to_csv('data/submissions/sub_plus-minus-sblattner_{}.csv'.format(model.get('abbr')))


for dd, d in enumerate(test_data_frame[test_data_frame.columns[1:101]].replace(1, 'O').replace(0, ' ').values[0:20]):
    print(dd)
    print(d.reshape(10, 10))
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import dataset_categories

"""
Running the main method or using the high level functions of mushroom_classifier.py for 
classification produces flawed results. The bugs will be fixed in a later version.
For now use the main method of fixed_classifier.py for classification.
"""


mode_dict = {'nb': 'Gaussian Naive Bayes', 'log_reg': 'Logistic regression',
             'lda': 'Linear Discriminant Analysis'}
              # 'qda': 'Quadratic Discriminant Analysis' removed for sketchy results probably caused by collinear variables


def plot_data(hue, data):
    for i, col in enumerate(data.columns):
        plt.figure(i)
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        ax = sns.countplot(x=data[col], hue=hue, data=data)
        plt.show()


# kwargs:
# type : abs for absolute values, ratio for ratio of missing values
# min and max: float only including attributes with min < ratio <= max
def get_variables_missing_dict(data, **kwargs):
    if 'type' not in kwargs:
        kwargs['type'] = 'ratio'
    if 'min' not in kwargs:
        kwargs['min'] = 0.0
    if 'max' not in kwargs:
        kwargs['max'] = 1.0
    if 'print' not in kwargs:
        kwargs['print'] = False
    if 'round' not in kwargs:
        kwargs['round'] = 3
    attributes_missing_dict = {}
    missing_categories_count = 0
    for column in data.columns:
        attributes_missing_dict[column] = 0
        attributes_missing_dict[column] += data[column].isnull().sum()
        missing_ratio = attributes_missing_dict[column] / len(data)
        if missing_ratio > kwargs['min'] and missing_ratio <= kwargs['max']:
            if kwargs['type'] == 'ratio':
                attributes_missing_dict[column] = missing_ratio
            if attributes_missing_dict[column] > 0:
                missing_categories_count += 1
        else:
            attributes_missing_dict.pop(column)
    if kwargs['print']:
        print("numbers of categories with missing values:", missing_categories_count)
        for e in attributes_missing_dict:
            if attributes_missing_dict[e] > 0:
                print(e + ' : ' + str(round(attributes_missing_dict[e], kwargs['round'])))
    if kwargs['type'] == 'abs' or kwargs['type'] == 'ratio':
        return attributes_missing_dict
    else:
        raise TypeError("invalid argument for type")


from sklearn.impute import SimpleImputer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer, KNNImputer
def imputate_missing_values_nominal(data):
    # imputer = IterativeImputer(max_iter=100, random_state=0)
    # imputer = KNNImputer(n_neighbors=2, weights='uniform')
    for col in data.select_dtypes(include=['object']).columns:
        """if more than one element is the most frequent, just pick the first one:
        modes = data[col].mode()
        if modes.shape[0] > 1:
            imputer_const = SimpleImputer(strategy="constant", fill_value=modes[0])
            imputer_const.fit_transform(data[col].values.reshape(-1, 1))
        else:"""
        imputer_freq = SimpleImputer(strategy='most_frequent')
        data[col] = imputer_freq.fit_transform(data[col].values.reshape(-1, 1))
    missing_attributes_dict = get_variables_missing_dict(data, type='ratio')
    return data


def handle_missing_values(data, **kwargs):
    if 'type' not in kwargs:
        kwargs['type'] = 'ratio'
    if 'min' not in kwargs:
        kwargs['min'] = 0.0
    if 'max' not in kwargs:
        kwargs['max'] = 1.0
    if 'print' not in kwargs:
        kwargs['print'] = False
    if 'round' not in kwargs:
        kwargs['round'] = 3
    # print all absolute values and ratios of missing values:
    if kwargs['print']:
        get_variables_missing_dict(data, type=kwargs['type'], print=True)
    # find attributes with missing value ratios >= threshold and remove them
    missing_attributes_dict = get_variables_missing_dict(data, type=kwargs['type'], print=False, min=0.5)
    drop_list = []
    for missing_attribute in missing_attributes_dict.keys():
        drop_list.append(missing_attribute)
        data = data.drop(missing_attribute, 1)
    if kwargs['print']:
        print("Variables with missing val ratio >=", kwargs['min'], drop_list)
    # impute remaining nominal attributes
    data = imputate_missing_values_nominal(data)
    return data


# label encode and one-hot encode data into metrical values
from sklearn.preprocessing import LabelEncoder
def encode_data_numerical(data):
    encoded_data = data.copy()
    le = LabelEncoder()
    encoded_data['class'] = le.fit_transform(data['class'])
    encoded_data = pd.get_dummies(encoded_data)
    return encoded_data


# split into test and training data
from sklearn.model_selection import train_test_split
def get_train_test(*datas, **kwargs):
    if 'test_size' not in kwargs:
        kwargs['test_size'] = 0.2
    # One dataset -> use sklearn.model_selection.train_test_split
    if len(datas) == 1:
        X = datas[0].drop(columns='class')
        y = datas[0]['class'].values.reshape(-1, 1)
        return train_test_split(X, y, test_size=kwargs['test_size'], random_state=1)
    # Two datasets -> use the first as training set, the second as test set
    elif len(datas) == 2:
        # shuffle datasets
        data1 = datas[0].sample(frac=1, random_state=1).reset_index(drop=True)
        data2 = datas[1].sample(frac=1, random_state=1).reset_index(drop=True)
        # assign datasets as train and test set and divide into X and y
        X_train = data1.drop(columns='class')
        y_train = data1['class'].values.reshape(-1, 1)
        X_test = data2.drop(columns='class')
        y_test = data2['class'].values.reshape(-1, 1)
        return [X_train, X_test, y_train, y_test]
    else:
        raise TypeError("Invalid parameter for *datas")


# classifier with logistic regression, LDA and QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def get_model(mode):
    if mode == 'nb':
        model = GaussianNB()
    if mode == 'log_reg':
        model = LogisticRegression(max_iter=10000)
    if mode == 'lda':
        model = LinearDiscriminantAnalysis()
    if mode == 'qda':
        model = QuadraticDiscriminantAnalysis()
    return model


def train_model(X_train, y_train, mode):
    model = get_model(mode)
    model.fit(X_train, y_train.ravel())
    return model


from sklearn.model_selection import cross_val_score
def cross_fold_validation(data, **kwargs):
    if 'k' not in kwargs:
        kwargs['k'] = 5
    if 'scoring' not in kwargs:
        kwargs['scoring'] = 'accuracy'
    if 'mode' not in kwargs:
        kwargs['mode'] = 'log_reg'
    X = data
    y = data['class'].values.reshape(-1, 1).ravel()
    model = get_model(kwargs['mode'])
    scores = cross_val_score(model, X, y, cv=kwargs['k'], scoring=kwargs['scoring'])
    return scores


def print_cross_value_scores(data, **kwargs):
    if 'scoring' not in kwargs:
        kwargs['scoring'] = 'roc_auc'
    modes = ['nb', 'log_reg', 'lda', 'qda']
    for mode in modes:
        classify_data(data, mode=mode)
        data = imputate_missing_values_nominal(data)
        data = encode_data_numerical(data)
        print(mode + ": " + str(cross_fold_validation(data, mode=mode, scoring=kwargs['scoring'])))


# getting probability and confusion matrices
def get_y_prob_pred(X_test, model, **kwargs):
    if 'threshold' not in kwargs:
        kwargs['threshold'] = 0.5
    pred_proba = model.predict_proba(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_prob > kwargs['threshold'], 1, 0)
    return [y_prob, y_pred]


from sklearn.metrics import confusion_matrix
def get_confusion_matrix(y_test, y_pred, **kwargs):
    """
    Parameters
    ----------
    y_test, pandas.Series: actual class values from the test set
    y_pred, pandas.Series: predicted class values

    kwargs:
    print, bool: prints return confusion matrix to console
    reformat, bool: changes the sklearn format for the confusion matrix to a common format:
     [[TN  FP]  ->  [[TP  FN]
      [FN  TP]]      [FP  TN]]


    Returns
    -------
    numpy.ndarray of confusion matrix format depends on kwargs['reformat']
    """
    if 'print' not in kwargs:
        kwargs['print'] = True
    if 'reformat' not in kwargs:
        kwargs['reformat'] = True
    conf_mat = confusion_matrix(y_test, y_pred)
    if kwargs['reformat']:
        conf_mat_temp = np.zeros(shape=(2, 2))
        conf_mat_temp[0, 0] = conf_mat[1, 1]
        conf_mat_temp[0, 1] = conf_mat[1, 0]
        conf_mat_temp[1, 0] = conf_mat[0, 1]
        conf_mat_temp[1, 1] = conf_mat[0, 0]
        conf_mat = conf_mat_temp
    if kwargs['print']:
        print("Confusion Matrix:", conf_mat_temp, sep="\n")
    return conf_mat


# ROC and AUC
from sklearn.metrics import roc_curve, auc
def get_roc_auc(y_test, y_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    return [false_positive_rate, true_positive_rate, thresholds]


def plot_roc(false_positive_rate, true_positive_rate, **kwargs):
    if 'title' not in kwargs:
        kwargs['title'] = ''
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(10, 10))

    plt.title('ROC ' + kwargs['title'])
    plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_class_ratio(data, **kwargs):
    if 'print' not in kwargs:
        kwargs['print'] = False
    x = data['class']
    ax = sns.countplot(x=x, data=data)
    count_p = 0
    count_e = 0
    for entry in x.array:
        if entry == 'p':
            count_p += 1
        else:
            count_e += 1
    if kwargs['print']:
        print("poisonous: " + str(count_p / len(x)))
        print("edible: " + str(count_e / len(x)))
    plt.show()


def classify_data(data, **kwargs):
    if 'mode' not in kwargs:
        kwargs['mode'] = 'log_reg'
    if 'threshold' not in kwargs:
        kwargs['threshold'] = 0.5
    if 'encode' not in kwargs:
        kwargs['encode'] = True
    if 'impute' not in kwargs:
        kwargs['impute'] = True
    data_copy = data.copy()
    if kwargs['impute']:
        data_copy = imputate_missing_values_nominal(data)
    if kwargs['encode']:
        data_copy = encode_data_numerical(data)
    X_train, X_test, y_train, y_test = get_train_test(data_copy)
    model = train_model(X_train, y_train, kwargs['mode'])
    y_prob, y_pred = get_y_prob_pred(X_test, model, threshold=kwargs['threshold'])
    return X_train, X_test, y_train, y_test, model, y_prob, y_pred


from sklearn import metrics
def get_evaluation_scores_dict(y_test, y_pred, **kwargs):
    if 'beta' not in kwargs:
        kwargs['beta'] = 2
    if 'round' not in kwargs:
        kwargs['round'] = 2
    if 'print' not in kwargs:
        kwargs['print'] = True
    accuracy = round(metrics.accuracy_score(y_test, y_pred), kwargs['round'])
    precision = round(metrics.precision_score(y_test, y_pred), kwargs['round'])
    recall = round(metrics.recall_score(y_test, y_pred), kwargs['round'])
    f_beta = round(metrics.fbeta_score(y_test, y_pred, beta=kwargs['beta']), kwargs['round'])
    evaluation_scores_dict = {'Accuracy': accuracy, 'Precision': precision,
                              'Recall': recall, 'F' + str(kwargs['beta']): f_beta}
    if kwargs['print']:
        for score_key in evaluation_scores_dict:
            print(score_key + ": " + str(evaluation_scores_dict[score_key]))
    return evaluation_scores_dict


if __name__ == "__main__":
    # import datasets
    data_primary = pd.read_csv(dataset_categories.FILE_PATH_PRIMARY_EDITED, sep=';', header=0)
    data_secondary = pd.read_csv(dataset_categories.FILE_PATH_SECONDARY_SHUFFLED, sep=';', header=0, low_memory=False)
    data_original = pd.read_csv(dataset_categories.FILE_PATH_1987, sep=',', header=0, dtype=object, na_values='?')
    data_dict = {'Secondary dataset': data_secondary, 'Original dataset': data_original}

    ## exploratory data analysis ##
    # missing values #
    # print absolute values and ratios of missing values:
    data_secondary = handle_missing_values(data_secondary, min=0.5, print=True)
    data_original = handle_missing_values(data_original, min=0.5, print=True)
    data_secondary.to_csv(dataset_categories.FILE_PATH_SECONDARY_NO_MISS, sep=';', index=False)
    data_original.to_csv(dataset_categories.FILE_PATH_1987_NO_MISS, sep=';', index=False)

    # cross validation
    accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)
    f2_scorer = metrics.make_scorer(metrics.fbeta_score, beta=2)
    scorers = [accuracy_scorer, f2_scorer]
    for data_key in data_dict:
        for score in scorers:
            print(score)
            for mode_key in mode_dict:
                data_encoded = encode_data_numerical(data_dict[data_key])
                X = data_encoded.drop(columns='class')
                y = data_encoded['class']
                cross_val_scores = cross_val_score(get_model(mode_key), X, y, cv=5, scoring=score)
                print(data_key, mode_key, [round(s, 2) for s in cross_val_scores])
                print('mean:', round(np.mean(cross_val_scores), 2), 'var:', round(np.var(cross_val_scores), 4) * 100)
                print()


    ## classification task ##
    for data_key in data_dict:
        print("\n***" + data_key + "***")
        for mode_key in mode_dict:
            print("\n" + mode_dict[mode_key] + ":")
            X_train, X_test, y_train, y_test, model, y_prob, y_pred = \
                classify_data(data_dict[data_key], mode=mode_key, threshold=0.5)
            get_confusion_matrix(y_test, y_pred)
            scores_dict = get_evaluation_scores_dict(y_test, y_pred)
            cross_val_scores = cross_fold_validation(encode_data_numerical(data_dict[data_key]), mode=mode_key, k=5)
            print(cross_val_scores)


    ## direct test between datasets ##
    print("\n*** direct tests between datasets ***\n")
    # get datasets with encoded and matched columns created by dataset_column_matching.py
    data_new_matched = pd.read_csv(dataset_categories.FILE_PATH_SECONDARY_MATCHED, sep=';')
    data_original_matched = pd.read_csv(dataset_categories.FILE_PATH_1987_MATCHED, sep=';')

    # test reduced instance
    print('\nTest reduced dataset on itself')
    X_train, X_test, y_train, y_test, model, y_prob, y_pred = \
        classify_data(data_original_matched, mode='log_reg', encode=False)
    get_confusion_matrix(y_test, y_pred)
    get_evaluation_scores_dict(y_test, y_pred)


    # use one dataset as the training set and the other dataset as the test set
    for mode_key in mode_dict:
        print("\ntraining set = secondary -> test set = original", "model:", mode_key)
        X_train, X_test, y_train, y_test = get_train_test(data_new_matched, data_original_matched)
        model = train_model(X_train, y_train, mode_key)
        y_prob, y_pred = get_y_prob_pred(X_test, model)
        print("Conf.-Mat.: " + str(get_confusion_matrix(y_test, y_pred)))
        get_evaluation_scores_dict(y_test, y_pred)

        print("\ntraining set = original -> test set = secondary", "model:", mode_key)
        X_train, X_test, y_train, y_test = get_train_test(data_original_matched, data_new_matched)
        model = train_model(X_train, y_train, mode_key)
        y_prob, y_pred = get_y_prob_pred(X_test, model)
        print("Conf.-Mat.: " + str(get_confusion_matrix(y_test, y_pred)))
        get_evaluation_scores_dict(y_test, y_pred)

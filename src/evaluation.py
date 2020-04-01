from sklearn.metrics import classification_report, plot_confusion_matrix, mean_squared_error, mean_squared_log_error, explained_variance_score

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def custom_classification_report(clf, labels, x_test, y_test, **kwargs):
    y_pred = clf.predict(x_test)

    clf_report = classification_report(
        y_pred,
        y_test,
        target_names=labels,
        output_dict=True
    )

    # Custom print because of incorrect formatting of original function
    for key in clf_report:
        if isinstance(clf_report[key], dict):
            print(f'\033[1m{key}\033[0m')

            for metric in clf_report[key]:
                print(f'{metric}: {clf_report[key][metric]}')
        else:
            print(f'{key}: {clf_report[key]}')

        print('\n')

    if kwargs.get('plot_confusion_matrix', True):
        matplotlib.rcParams.update({'font.size': 29})
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (40, 30)))
            
        plot_confusion_matrix(clf, x_test, y_test, display_labels=labels, ax=ax, xticks_rotation='vertical')
        
        
def custom_classification_report_nn(clf, labels, x_test, y_test, **kwargs):
    y_pred = clf.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    clf_report = classification_report(
        y_pred,
        y_test,
        target_names=labels,
        output_dict=True
    )

    # Custom print because of incorrect formatting of original function
    for key in clf_report:
        if isinstance(clf_report[key], dict):
            print(f'\033[1m{key}\033[0m')

            for metric in clf_report[key]:
                print(f'{metric}: {clf_report[key][metric]}')
        else:
            print(f'{key}: {clf_report[key]}')

        print('\n')
    
    
def custom_regression_report(clf, x_test, y_test, **kwargs):
    y_pred = clf.predict(x_test)
    
    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
#     print(f'MSLE: {mean_squared_log_error(y_test, y_pred)}')
    print(f'Explained variance - uniform_average (higher is better): {explained_variance_score(y_test, y_pred, multioutput="uniform_average")}')
    print(f'Explained variance - variance_weighted (higher is better): {explained_variance_score(y_test, y_pred, multioutput="variance_weighted")}')
    
    data = pd.DataFrame(data={ 'y_test': y_test / 1000000, 'y_pred': y_pred / 1000000 })
    max_value = max(max(y_test), max(y_pred)) / 1000000
    
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    sns.scatterplot(x='y_test', y='y_pred', data=data)
  
    print('\n')
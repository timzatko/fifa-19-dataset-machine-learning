from sklearn.metrics import classification_report, plot_confusion_matrix

import matplotlib.pyplot as plt


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

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (20, 10)))

    plot_confusion_matrix(clf, x_test, y_test, display_labels=labels, ax=ax)

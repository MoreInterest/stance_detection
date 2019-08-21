def plot_confusion_matrix(true, predicted, labels=["Neutral", "Favour", "Against"]):
    
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    import numpy as np

    cm = confusion_matrix(true, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), ylabel='True label', xlabel='Predicted label', xticklabels=labels, yticklabels=labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100., fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax


def get_batch(step, batch_size, z):
    
    from math import ceil
    
    batches = ceil(len(z[0]) / batch_size)
    i = step % batches
    start, end = i * batch_size, min((i + 1) * batch_size, len(z[0]))
    return [a[start:end].cuda() for a in z]


def print_misclassifications(dataset, predictions, pair=None):
    
    """ Prints the misclassifications; if pair is given, only the misclassification
    having true label pair[0] and predicted label pair[1] are printed. """
    
    def get_stance_string(z):
        if z == 0:
            return "Neutral"
        elif z == 1:
            return "Favour"
        elif z == 2:
            return "Against"
    
    for text, target, stance, predicted_stance in zip(dataset.texts, dataset.targets, dataset.stances, predictions):
        if stance != predicted_stance:
            if pair is None or (stance == pair[0] and predicted_stance == pair[1]):
                print("Text:\t\t{}".format(text))
                print("Target:\t\t{}".format(target))
                print("Stance:\t\t{}".format(get_stance_string(stance)))
                print("Prediction:\t{}\n".format(get_stance_string(predicted_stance)))
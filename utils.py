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
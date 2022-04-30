from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    precision_score, recall_score
)


def conf_mx_plot(mx, title=None, ax=None, fontsize=None):
    """
    returns: a matplotlib black-white representation of the confusion matrix
            The higher the greater the intensity of the white shade and vice.
    """
    
    if ax is not None:
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
            
        ax.matshow(mx, cmap=plt.cm.gray)
    else:
        plt.matshow(mx, cmap=plt.cm.gray)
        
        if title is not None:
            plt.title(title, fontsize=fontsize)
            
        plt.show()


def compare_conf_mx(shape, mx_s, titles=None, fontsize=15, figsize=(12, 10)):
    """
    returns: a matplotlib.subplot comparing multiple matrices in order of their shape
            and for easier comparision.
    """
    
    fig, axes = plt.subplots(*shape, figsize=figsize)
    
    for i, ax in enumerate(axes.ravel()):
        title = titles[i] if titles is not None else None
        
        conf_mx_plot(mx_s[i], title=title, ax=ax, fontsize=fontsize)
        
    plt.show()


def plot_word_frequency(df, column=None, col_filter=None, use_col_filter=None, figsize=(12, 5)):
    """
    Parameters:
    --------------------
    df: Pandas Dataframe
        Must be provided
    
    column: string
        String value to extract feature column.
        Must be provided
        
    col_filter: string
        Values to equivalate the column with. 
        `column` is used if `use_col_filter` is
        not provided.
        
    use_col_filter: string
        Equivalates the `col_filter` to the
        `use_col_filter`
    
    Returns:
    --------------------
    returns the word frequency
    """
    
    if column is None:
        raise ValueError("column must not be none")
        
    data = None
    
    if col_filter is None:
        data = df[column]
    else:
        if use_col_filter is None:
            data = df[column][df[column] == col_filter]
        else:
            data = df[column][df[use_col_filter] == col_filter]

    texts = ' '.join([text for text in data])
    word_cloud = WordCloud(
        collocations=False, 
        background_color='white'
    ).generate(texts)

    plt.figure(figsize=figsize)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    
def scorer(y_true, y_pred, accuracy=True, precision=True, recall=True):
    """
    Performs multiple scoring on actual and predicted value
    
    You can turn off a metrics by setting it to False.
    """
    
    if accuracy:
        print("Accuracy score", accuracy_score(y_true, y_pred), end='\n\n')
    
    if precision:
        print("Precision score", precision_score(y_true, y_pred, average='macro'), end='\n\n')
    
    if recall:
        print("Recall score", recall_score(y_true, y_pred, average='macro'), end='\n\n')
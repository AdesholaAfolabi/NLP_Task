import re
import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stop_words(document):

    '''
    Remove uninformative words such as the, is, because, etc from the corpus
    
    Parameters:
    ------------------------
    document: The corpus to operate on
    
    Returns
    -------
        Corpus without stop words
    '''
  
    cleaned_document = " ".join([word.strip() for word in document.split() if word not in stop_words and len(word)>2])
    return cleaned_document

def preprocess_document_for_fine_tuning(document):
    
    '''
    Pre-process the corpus by converting each word to lower case, removing 
    special characters and apply the remove_stop_words method
    
    Parameters:
    ------------------------
    document: The corpus to operate on
    
    Returns
    -------
        Clean corpus
    '''
    document = document.lower()
    document = " ".join([re.sub('[^A-Za-z]+', ' ', word) for word in document.split()])
    document = remove_stop_words(document)
   
    return document

def embedding_gen(data: pd.DataFrame):

    '''
    Preprocess and get document embeddings based on the corpus

    Parameters:
    ------------------------

    data: DataFrame containing all text values
    
    Returns
    -------
        A generator object with a list of mean embeddings
    '''
    from sentence_transformers import SentenceTransformer, models
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    
    for index, row in data.iterrows():
        document = preprocess_document_for_fine_tuning(row['body_basic'].strip())
        embeddings = model.encode(document)
        yield [embeddings]
        
def flatten(list_of_lists):
    
    '''
    Flatten out word embeddings depending on the number of tokens extracted from a corpus
    The number of tokens is usually arbitrary

    Parameters:
    ------------------------
    list_of_lists: Could be a list or multiple lists depending on the tokens
    
    Returns
    -------
        A single list containing the mean embeddings of a corpus
    '''

    return [item for sublist in list_of_lists for item in sublist]
        
def embedding_dataframe(data: pd.DataFrame):
    '''
    Generate embeddings dataframe with n-dimension based on the transformer model
    '''

    EMBEDDING_VECTOR_DIMENSION = 384
    emb_cols = [f'emb_{i}' for i in range(EMBEDDING_VECTOR_DIMENSION)]
    df_datasetembeddings = pd.DataFrame(flatten(list(embedding_gen(data))), columns=emb_cols)
    return df_datasetembeddings

def merge_emb(data):
    '''
    Merge embeddings back to the original dataframe
    '''
    
    emb_frame = embedding_dataframe(data)
    emb_frame = emb_frame.fillna(0)
    mergedDf = pd.concat([data.reset_index(drop=True),emb_frame.reset_index(drop=True)],axis=1)
    mergedDf = mergedDf.drop(['body_basic'], axis=1)
    return mergedDf
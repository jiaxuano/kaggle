import numpy as np
import pickle
import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def open_pic(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
    # Load the data from the file
        training_data = pickle.load(file)
    training_data = pd.DataFrame(training_data)
    return training_data

def clean_features(lst):
    # Initialize all variables with NaN to handle missing values
    square_m = hab = bano = np.nan
    
    # Process each element in the list if it exists
    if len(lst) > 0 and lst[0]:
        square_m, _ = lst[0].split(' ')
        square_m = float(square_m)
    
    if len(lst) > 1 and lst[1]:
        hab, _ = lst[1].split(' ')
        hab = float(hab)
    
    if len(lst) > 2 and lst[2]:
        # Check if 'bano' information is available or it's the price per square meter
        if 'baño' in lst[2]:
            bano, _ = lst[2].split(' ')
            bano = float(bano)
    
    return square_m, hab, bano

def fix_euros(element):
    price, _ = element.split(' ')
    return float(price)


def vecotrizer_df(df):
    vectorizer = TfidfVectorizer(max_features=1000)  # Keep only the top 1000 features

    tfidf_matrix = vectorizer.fit_transform(df['desc'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())


    # Concatenate the TF-IDF features with the additional features
    final_features = pd.concat([tfidf_df, df], axis=1)
    return final_features
    
def prepare_tfidf_features(df_train, df_test, text_column):
    """
    This function fits a TF-IDF Vectorizer on the training data and transforms both
    training and test data to concatenate the TF-IDF features with the original features.

    Parameters:
    - df_train: DataFrame containing the training data.
    - df_test: DataFrame containing the test data.
    - text_column: The name of the column in the DataFrames containing text to vectorize.

    Returns:
    - final_features_train: Training data with TF-IDF features added.
    - final_features_test: Test data with TF-IDF features added.
    """
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)

    # Fit the vectorizer on the training text data and transform it
    tfidf_matrix_train = vectorizer.fit_transform(df_train[text_column])
    tfidf_df_train = pd.DataFrame(tfidf_matrix_train.toarray(), columns=vectorizer.get_feature_names_out())

    # Transform the test text data
    tfidf_matrix_test = vectorizer.transform(df_test[text_column])
    tfidf_df_test = pd.DataFrame(tfidf_matrix_test.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenate the TF-IDF features with the original features
    final_features_train = pd.concat([df_train.reset_index(drop=True), tfidf_df_train], axis=1)
    final_features_test = pd.concat([df_test.reset_index(drop=True), tfidf_df_test], axis=1)

    return final_features_train, final_features_test
def prepare_tokenized_features(df_train, df_test, text_column):
    """
    This function tokenizes the text in the specified column of the training and test data.

    Parameters:
    - df_train: DataFrame containing the training data.
    - df_test: DataFrame containing the test data.
    - text_column: The name of the column in the DataFrames containing text to tokenize.

    Returns:
    - final_features_train: Training data with tokenized text added.
    - final_features_test: Test data with tokenized text added.
    """
    # Get Spanish stop words as a list
    stop_words = stopwords.words('spanish')
    
    # Initialize the CountVectorizer for tokenization with Spanish stop words removed
    vectorizer = CountVectorizer(max_features=600, stop_words=stop_words)

    # Fit the vectorizer on the training text data and transform it
    tokenized_matrix_train = vectorizer.fit_transform(df_train[text_column])
    tokenized_df_train = pd.DataFrame(tokenized_matrix_train.toarray(), columns=vectorizer.get_feature_names_out())

    # Transform the test text data
    tokenized_matrix_test = vectorizer.transform(df_test[text_column])
    tokenized_df_test = pd.DataFrame(tokenized_matrix_test.toarray(), columns=vectorizer.get_feature_names_out())

    # Concatenate the tokenized features with the original features
    final_features_train = pd.concat([df_train.reset_index(drop=True), tokenized_df_train], axis=1)
    final_features_test = pd.concat([df_test.reset_index(drop=True), tokenized_df_test], axis=1)

    return final_features_train, final_features_test
def cleaning(dataframe):

    square_m_list , hab_list, bano_list, price_list = [], [], [], []
    for index, row in enumerate(dataframe['features']):
        square_m , hab, bano = clean_features(row)
        square_m_list.append(square_m)
        hab_list.append(hab)
        bano_list.append(bano)
    
    for row in dataframe['price']:
        price = fix_euros(row)
        price_list.append(price)
    dataframe['price'] = price_list
    dataframe['square_m'] = square_m_list
    dataframe['hab'] = hab_list
    dataframe['bano'] = bano_list
    dataframe.drop('features', axis=1, inplace=True)
    dataframe.drop('selltype', axis=1, inplace=True)
    dataframe.drop('subtype', axis=1, inplace=True)
    dummies = pd.get_dummies(dataframe['type'])
    dataframe = pd.concat([dataframe, dummies], axis=1)
    dataframe.drop('type', axis=1, inplace=True)
    dataframe.drop('loc', axis=1, inplace=True)
    dataframe.drop('loc_string', axis=1, inplace=True)
    dataframe.drop('title', axis=1, inplace=True)

    return dataframe
def cleaning_test(dataframe):
    square_m_list , hab_list, bano_list, price_list = [], [], [], []
    for index, row in enumerate(dataframe['features']):
        square_m , hab, bano = clean_features(row)
        square_m_list.append(square_m)
        hab_list.append(hab)
        bano_list.append(bano)
    
    #for row in dataframe['price']:
        #price = fix_euros(row)
        #price_list.append(price)
    #dataframe['price'] = price_list
    dataframe['square_m'] = square_m_list
    dataframe['hab'] = hab_list
    dataframe['bano'] = bano_list
    dataframe.drop('features', axis=1, inplace=True)
    dataframe.drop('selltype', axis=1, inplace=True)
    dataframe.drop('subtype', axis=1, inplace=True)
    dummies = pd.get_dummies(dataframe['type'])
    dataframe = pd.concat([dataframe, dummies], axis=1)
    dataframe.drop('type', axis=1, inplace=True)
    dataframe.drop('loc', axis=1, inplace=True)
    dataframe.drop('loc_string', axis=1, inplace=True)
    dataframe.drop('title', axis=1, inplace=True)

    return dataframe
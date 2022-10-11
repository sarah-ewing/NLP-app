import time
import datetime
start_time = time.time()

############################################################################
print("install & import packages", datetime.datetime.now())
try:
    import botocore
    import boto3
except ImportError:
    print("No module named botocore or boto3. You may need to install boto3")
    sys.exit(1)
boto3.compat.filter_python_deprecation_warnings()
print("imported boto3 using compat.filter\nIt's still deprecated, so we still need to fix that.")

import subprocess
import sys
import numpy as np
import pandas as pd
from pandas import read_csv
import sklearn
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
np.random.seed(7)

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "texthero"])

install("snowflake-connector-python")
import snowflake.connector

## !pip install -U spacy
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "spacy"])
install("spacy==3.4.0")
import spacy

install("srsly==2.4.0")
import srsly

install("spacy_langdetect")
import spacy_langdetect
from spacy_langdetect import LanguageDetector

# pip install texthero --update
import texthero as hero
from texthero import preprocessing

# subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
## sudo python3 -m spacy download en
# nlp = spacy.load('en_core_web_sm')

install("scispacy")
import scispacy

## !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz 
install("https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz")
import en_core_sci_lg

from spacy.language import Language

def create_lang_detector(nlp, name):
    return LanguageDetector()

Language.factory("language_detector", func=create_lang_detector)

import nltk
nltk.download('punkt')

# Get the credentials for Snowflake
ssm_params = ['/rev-ml/sagemaker/snowflake-user', '/rev-ml/sagemaker/snowflake-password']
def get_credentials(params):
    ssm = boto3.client('ssm','us-west-2')
    response = ssm.get_parameters(
        Names=params,
        WithDecryption=True
    ) 
    #Build dict of credentials
    param_values={k['Name']:k['Value'] for k in  response['Parameters']}
    return param_values

credentials = get_credentials(ssm_params)
# Gets the version
ctx = snowflake.connector.connect(
                                user=credentials['/rev-ml/sagemaker/snowflake-user'],
                                password=credentials['/rev-ml/sagemaker/snowflake-password'],
                                account='msa72542'
                                )

############################################################################
print("load the data", datetime.datetime.now())
SQL_query = """
SELECT DISTINCT
        TCSA.ID,
        TCSA.applicantid,
        TCSA.questionid,
        TRIM(TCSA.answertext) AS answertext
    FROM LAKE_REVCOM.DBO.TCAPPLICANTSHORTANSWERRESPONSES TCSA
    WHERE 
        answertext IS NOT NULL 
        AND answertext != ''
        AND questionid in (94, 224, 225)
    ORDER BY ID;   
"""

cur = ctx.cursor().execute(SQL_query)
df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
ctx.cursor().close()
print("Data loaded.", datetime.datetime.now())

############################################################################
## Apply Bag of words
## step 1 - clean text
print("Apply Bag of words step 1 - clean the answer text", datetime.datetime.now())

print("text start cleaning", datetime.datetime.now())
def replace_hyphen(match):
    match_string = match.group(0)
    return match_string[0] + " " + match_string[2] # replace the hyphen with a space

df['clean_text'] = df['ANSWERTEXT'].str.replace(r'[A-Za-z]-[A-Za-z]', replace_hyphen, regex=True)

custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_whitespace,
                   preprocessing.remove_diacritics,
                   preprocessing.remove_stopwords,
                   preprocessing.replace_punctuation
                  ]

df['clean_text'] = hero.clean(df['clean_text'], custom_pipeline)
df['clean_text'] = [n.replace('{','') for n in df['clean_text']]
df['clean_text'] = [n.replace('}','') for n in df['clean_text']]
df['clean_text'] = [n.replace('(','') for n in df['clean_text']]
df['clean_text'] = [n.replace(')','') for n in df['clean_text']]
print("text cleaned", datetime.datetime.now())

##### need to test / train split data for pipeline
# stratified random sample by question ID
X_train = df.groupby('QUESTIONID', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=200)) 
X_test = df.drop(X_train.index)
# date = str(datetime.datetime.now().date()).replace('-','_')
date_string = '2022_09_27'

############################################################################
## FUNCTION step 2 - make a bag of words top 50 words:
print("Apply Bag of words step 2 - make a bag of words top 50 words", datetime.datetime.now())
def make_bag_of_words(data_in):
    import nltk
    import heapq
    import pickle
    
    data_in = data_in.reset_index(drop=True)
    # Creating the Bag of Words model
    word2count = {}
    for i in range(0, len(data_in['clean_text'])):
        if i % 100000 == 0:
            print(i)
        words = nltk.word_tokenize(data_in['clean_text'][i])
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    freq_words = heapq.nlargest(50, word2count, key=word2count.get)
    return(freq_words)


## 94
print("top 50 words QUESTION ---- 94 ----", datetime.datetime.now())
try:
    BOW_94 = pd.read_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_94.csv'.format(DATE = date_string))
    BOW_94 = BOW_94.drop_duplicates()
    print("done", BOW_94.shape)
    BOW_94 = list(BOW_94['0'])
except:
    BOW_94 = make_bag_of_words(data_in = X_train[X_train['QUESTIONID'] == 94])
    BOW_94 = pd.DataFrame(BOW_94)
    BOW_94.to_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_94.csv'.format(DATE = date_string),
                      index = False)
    BOW_94 = list(BOW_94[0])
print(BOW_94)

## 224
print("top 50 words QUESTION ---- 224 ----", datetime.datetime.now())
try:
    BOW_224 = pd.read_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_224.csv'.format(DATE = date_string))
    BOW_224 = BOW_224.drop_duplicates()
    print("done", BOW_224.shape)
    BOW_224 = list(BOW_224['0'])
except:
    BOW_224 = make_bag_of_words(data_in = X_train[X_train['QUESTIONID'] == 224])
    BOW_224 = pd.DataFrame(BOW_224)
    BOW_224.to_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_224.csv'.format(DATE = date_string),
                      index = False)
    BOW_224 = list(BOW_224[0])
print(BOW_224)

## 225
print("top 50 words QUESTION ---- 225 ----", datetime.datetime.now())
try:
    BOW_225 = pd.read_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_225.csv'.format(DATE = date_string))
    BOW_225 = BOW_225.drop_duplicates()
    print("done", BOW_225.shape)
    BOW_225 = list(BOW_225['0'])
except:
    BOW_225 = make_bag_of_words(data_in = X_train[X_train['QUESTIONID'] == 225])
    BOW_225 = pd.DataFrame(BOW_225)
    BOW_225.to_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_225.csv'.format(DATE = date_string),
                      index = False)
    BOW_225 = list(BOW_225[0])
print(BOW_225)

## ALL
print("top 50 words QUESTION ---- ALL ----", datetime.datetime.now())

try:
    BOW_ALL = pd.read_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_ALL.csv'.format(DATE = date_string))
    BOW_ALL = BOW_ALL.drop_duplicates()
    print("done", BOW_ALL.shape)
    BOW_ALL = list(BOW_ALL['0'])
except:
    BOW_ALL = make_bag_of_words(data_in = X_train)
    BOW_ALL = pd.DataFrame(BOW_ALL)
    BOW_ALL.to_csv('s3://revcom-sagemaker-input-features/application/BoW/top_50_words/{DATE}/BOW_ALL.csv'.format(DATE = date_string),
                      index = False)
    BOW_ALL = list(BOW_ALL[0])
print(BOW_ALL)

############################################################################
#### step 3 - make bag of words matrix
print("Apply Bag of words step 3 - make bag of words matrix", datetime.datetime.now())

############################################################################
## FUNCTION to define the data missing for a bag of words matrix
def missing_BoW_matrix_data(df, date_string, questionID):
    print("save out BoW Matrix QUESTION ---- {} ----".format(questionID), datetime.datetime.now())

    try:
        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket('revcom-sagemaker-input-features')
        Matrix = pd.DataFrame()
        
        print("bucket name:", my_bucket, Matrix.shape, datetime.datetime.now())
        
        ## load the data if available
        for object_summary in my_bucket.objects.filter(Prefix="application/BoW/matrix/{DATE}".format(DATE = date_string)):
            print(object_summary.key, datetime.datetime.now())
            if object_summary.key.endswith(".csv"):
                if object_summary.key.startswith("application/BoW/matrix/{DATE}/BoW_{QUESTION_ID}_".format(
                    DATE = date_string,
                    QUESTION_ID = questionID
                )):
                    df3 = pd.read_csv("s3://revcom-sagemaker-input-features/{}".format(object_summary.key))
                    Matrix = Matrix.append(df3, ignore_index=True)
                    print(object_summary.key, Matrix.shape, datetime.datetime.now())
        Matrix = Matrix.drop_duplicates()
        
        if Matrix.shape[1] != 51:
            sys.exit("Question {question}, and the DIMS DO NOT make sense! there are {dimms} coolumns and there should be 50 + ID column. probably need to re-define the top 50 words.".format(
            question = questionID,
            dimms =  Matrix.shape[1]
        ))

        ## see what the difference between whats saved and what needs to be made?
        combination = df[['ID', 'QUESTIONID', 'clean_text']][df['QUESTIONID'] == questionID].merge(Matrix,
                                                       how='left',
                                                       left_on=['ID'],
                                                       right_on = ['ID'])
        print("making sure the matrix makes sense:", combination[combination[combination.columns[5]].isna() == False].iloc[[0]])
        ## look at the 4th column and only keep data that does not have matrix data
        combination = combination[combination[combination.columns[5]].isna() == True]
        
        

        combination = combination[['ID', 'QUESTIONID', 'clean_text']].drop_duplicates()
        print("loaded old data", Matrix.shape, 
              "current list of IDs", df[df['QUESTIONID'] == questionID].shape,
              "data that is needed", combination.shape)


    except:
        combination = df[['ID', 'QUESTIONID', 'clean_text']][df['QUESTIONID'] == questionID].drop_duplicates()
        print("loaded new data", combination.shape)


        
    if df[df['QUESTIONID'] == questionID].shape[0] - combination.shape[0] == Matrix.shape[0]:
        print("Question {}, and the DIMS make sense!".format(questionID))

    if df[df['QUESTIONID'] == questionID].shape[0] - combination.shape[0] != Matrix.shape[0]:
        sys.exit("Question {}, and the DIMS DO NOT make sense! probably need to re-define the top 50 words.".format(
            questionID))

    return(combination)

############################################################################
### FUNCTION to make the BoW matrix
def make_BoW_matrix(top_50_word_data, date_string, data_in, question):
    import pickle
    import time
    import os
    import pandas as pd
    start_time = time.time()
    import nltk
    import datetime
    
    #### this is the top 50 words
    freq_words = top_50_word_data

    df2 = pd.DataFrame([])
    print("starting the loop.")
    
    data_in = data_in.reset_index(drop=True)
    c = 0
    for i in range(0, len(data_in['ID'])):
        if i % 10000 == 0:
            print(i, question, datetime.datetime.now())
        c += 1
        temp = data_in.clean_text[i]
        temp_string = str(temp)

        vector = []
        for word in freq_words:
            if word in nltk.word_tokenize(str(temp_string)):
                vector.append(1)
            else:
                vector.append(0)
        if len(vector) != 50:
            print(i, data_in['clean_text'][i])
        sent1 = pd.DataFrame(data = vector).transpose()
        sent1['ID'] = data_in['ID'][i]

        df2 = df2.append(sent1)

        if c == 10000:
            df2 = df2.rename(columns={0:freq_words[0],
                                  1:freq_words[1],
                                  2:freq_words[2],
                                  3:freq_words[3],
                                  4:freq_words[4],
                                  5:freq_words[5],
                                  6:freq_words[6],
                                  7:freq_words[7],
                                  8:freq_words[8],
                                  9:freq_words[9],
                                  10:freq_words[10],
                                  11:freq_words[11],
                                  12:freq_words[12],
                                  13:freq_words[13],
                                  14:freq_words[14],
                                  15:freq_words[15],
                                  16:freq_words[16],
                                  17:freq_words[17],
                                  18:freq_words[18],
                                  19:freq_words[19],
                                  20:freq_words[20],
                                  21:freq_words[21],
                                  22:freq_words[22],
                                  23:freq_words[23],
                                  24:freq_words[24],
                                  25:freq_words[25],
                                  26:freq_words[26],
                                  27:freq_words[27],
                                  28:freq_words[28],
                                  29:freq_words[29],
                                  30:freq_words[30],
                                  31:freq_words[31],
                                  32:freq_words[32],
                                  33:freq_words[33],
                                  34:freq_words[34],
                                  35:freq_words[35],
                                  36:freq_words[36],
                                  37:freq_words[37],
                                  38:freq_words[38],
                                  39:freq_words[39],
                                  40:freq_words[40],
                                  41:freq_words[41],
                                  42:freq_words[42],
                                  43:freq_words[43],
                                  44:freq_words[44],
                                  45:freq_words[45],
                                  46:freq_words[46],
                                  47:freq_words[47],
                                  48:freq_words[48],
                                  49:freq_words[49]})
            print(i)
            date = str(datetime.datetime.now().date()).replace('-','_')
            file_name = "BoW_{question}_{i}_{date}.csv".format(question = question,
                                                        i = i,
                                                       date = date)
            df2.to_csv('s3://revcom-sagemaker-input-features/application/BoW/matrix/{DATE}/{FILE_NAME}'.format(
                DATE = date_string,
                FILE_NAME = file_name),
                      index = False) 
            print(i, "of", len(data_in['ID']), "so I am {}% done.".format(round(i/len(data_in['ID']), 2)*100),
                 datetime.datetime.now())
            df2 = pd.DataFrame([])
            c = 0
            
        if i == (data_in.shape[0] - 1):
            df2 = df2.rename(columns={0:freq_words[0],
                                  1:freq_words[1],
                                  2:freq_words[2],
                                  3:freq_words[3],
                                  4:freq_words[4],
                                  5:freq_words[5],
                                  6:freq_words[6],
                                  7:freq_words[7],
                                  8:freq_words[8],
                                  9:freq_words[9],
                                  10:freq_words[10],
                                  11:freq_words[11],
                                  12:freq_words[12],
                                  13:freq_words[13],
                                  14:freq_words[14],
                                  15:freq_words[15],
                                  16:freq_words[16],
                                  17:freq_words[17],
                                  18:freq_words[18],
                                  19:freq_words[19],
                                  20:freq_words[20],
                                  21:freq_words[21],
                                  22:freq_words[22],
                                  23:freq_words[23],
                                  24:freq_words[24],
                                  25:freq_words[25],
                                  26:freq_words[26],
                                  27:freq_words[27],
                                  28:freq_words[28],
                                  29:freq_words[29],
                                  30:freq_words[30],
                                  31:freq_words[31],
                                  32:freq_words[32],
                                  33:freq_words[33],
                                  34:freq_words[34],
                                  35:freq_words[35],
                                  36:freq_words[36],
                                  37:freq_words[37],
                                  38:freq_words[38],
                                  39:freq_words[39],
                                  40:freq_words[40],
                                  41:freq_words[41],
                                  42:freq_words[42],
                                  43:freq_words[43],
                                  44:freq_words[44],
                                  45:freq_words[45],
                                  46:freq_words[46],
                                  47:freq_words[47],
                                  48:freq_words[48],
                                  49:freq_words[49]})
            date = str(datetime.datetime.now().date()).replace('-','_')
            file_name = "BoW_{question}_{i}_{date}.csv".format(question = question, 
                                                               i = i,
                                                              date = date)
            df2.to_csv('s3://revcom-sagemaker-input-features/application/BoW/matrix/{DATE}/{FILE_NAME}'.format(
                DATE = date_string,
                FILE_NAME = file_name),
                      index = False)
            print(i, "of", len(data_in['ID']), "so I am {}% done.".format(round(i/len(data_in['ID']), 2)*100),
                 datetime.datetime.now())
            df2 = pd.DataFrame([])   

    print("DONE!")
    end_time = time.time()
    print(end_time - start_time)

############################################################################
### call the functions to define missing data and build the missing data
#### 94
combination = missing_BoW_matrix_data(df = df, 
                                      date_string = date_string,
                                      questionID = 94)
print("Top 50 words for Q 94:", BOW_94)

make_BoW_matrix(top_50_word_data = BOW_94, 
                data_in = combination[combination['QUESTIONID'] == 94],
                date_string = date_string,
                question = 94)
#### 224
combination = missing_BoW_matrix_data(df = df, 
                                      date_string = date_string,
                                      questionID = 224)
print("Top 50 words for Q 224:", BOW_224)

make_BoW_matrix(top_50_word_data = BOW_224, 
                data_in = combination[combination['QUESTIONID'] == 224],
                date_string = date_string,
                question = 224)
#### 225
combination = missing_BoW_matrix_data(df = df, 
                                      date_string = date_string,
                                      questionID = 225)
print("Top 50 words for Q 225:", BOW_225)

make_BoW_matrix(top_50_word_data = BOW_225, 
                data_in = combination[combination['QUESTIONID'] == 225],
                date_string = date_string,
                question = 225)

############################################################################
## Step 4 - Calculate the PCA
print("Calculate the PCA step 4", datetime.datetime.now())

## load the BoW matrix
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('revcom-sagemaker-input-features')
Matrix_94 = pd.DataFrame()

## load the data if available
for object_summary in my_bucket.objects.filter(Prefix="application/BoW/matrix/{DATE}".format(DATE = date_string)):
    if object_summary.key.endswith(".csv"):
        if object_summary.key.startswith("application/BoW/matrix/{DATE}/BoW_94_".format(DATE = date_string)):
            df3 = pd.read_csv("s3://revcom-sagemaker-input-features/{}".format(object_summary.key))
            Matrix_94 = Matrix_94.append(df3, ignore_index=True)
Matrix_94 = Matrix_94.drop_duplicates()
print("Question 94 BoW Matrix", Matrix_94.shape)
from sklearn import decomposition

## need to do train / test split
BoW = df3.drop(labels = 'ID', axis = 1)
pca = decomposition.PCA(n_components=3)

## fit the model
pca.fit(Matrix_94)

X1 = pca.transform(Matrix_94)

print(pca.explained_variance_)

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index = BOW_94)
## save out loadings to make model static
print(loadings)
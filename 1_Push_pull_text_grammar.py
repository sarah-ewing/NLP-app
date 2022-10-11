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
import datetime
import pandas as pd
from pandas import read_csv
import math

# fix random seed for reproducibility
np.random.seed(7)

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("snowflake-connector-python")
import snowflake.connector

# install("gensim")

# install("texthero")
# import texthero

install("language-tool-python")
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

# import nltk
# nltk.download('punkt')

# install("sklearn")
# import sklearn

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)

### snowflake connectors
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

ctx.cursor().execute('USE warehouse DATAQUERY_WH')

###############
## load the data
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
df.head()
ctx.cursor().close()

### load the prior data if available
try:
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('revcom-sagemaker-input-features')
    prior_data = pd.DataFrame()

    for object_summary in my_bucket.objects.filter(Prefix="application/language_tool_python"):
        if object_summary.key.endswith(".csv"):
            df3 = pd.read_csv("s3://revcom-sagemaker-input-features/{}".format(object_summary.key))
            prior_data = prior_data.append(df3, ignore_index=True)
        print(object_summary.key, prior_data.shape, datetime.datetime.now())
    prior_data = prior_data.drop_duplicates()

except:
    print("there is no prior data to load.", prior_data.shape, df.shape, datetime.datetime.now())

## if prior data loaded then merge to df and do not re-calc
try:
    ## define data not calculated - then send them into the loop
    combination = df.merge(prior_data,
             how='left',
             left_on=['ID', 'APPLICANTID', 'QUESTIONID', 'ANSWERTEXT'],
             right_on = ['ID', 'APPLICANTID', 'QUESTIONID', 'ANSWERTEXT'])
    loop_data = combination[combination['mistakes'].isna() == True]
    loop_data = loop_data.reset_index(drop=True)
    loop_data = loop_data.sort_values(by=['ID', 'APPLICANTID', 'QUESTIONID'])
    if loop_data.shape[0] == (df.shape[0] - prior_data.shape[0]):
        print("dims seems to work")
        print("loop_data", loop_data.shape[0], "prior_data", prior_data.shape, "df", df.shape)
    else:
        print("dims NOT work ----- please send help. prior data not being loaded")
        print("loop_data", loop_data.shape[0], "prior_data", prior_data.shape, "df", df.shape)
    
except: ## if there is no data to load, run all the loops
    print("there was no prior data to exclude from the loop")
    loop_data = df.drop_duplicates()
    
def return_number(error_type):
    try:
        x = temp_table[7][temp_table['mistake_type'] == error_type].item()
    except:
        x = 0
    return(x)

c = 0
df1 = pd.DataFrame([])
for i in range(0, len(loop_data['ANSWERTEXT'])):
    # get the matches
    matches = tool.check(loop_data['ANSWERTEXT'][i])
    mistakes = pd.DataFrame(matches)

    if mistakes.shape[0] != 0:
        mistakes = pd.DataFrame(matches)[7]
        loop_data.loc[i, 'mistakes'] = len(mistakes)
        
        try:
            temp_table = pd.DataFrame(pd.DataFrame(matches)[7].value_counts())
            temp_table.reset_index(inplace=True)
            temp_table = temp_table.rename(columns = {'index':'mistake_type'})
            
            df_temp = {
                'ID': loop_data.loc[i, 'ID'], 
                'APPLICANTID': loop_data.loc[i, 'APPLICANTID'], 
                'QUESTIONID': loop_data.loc[i, 'QUESTIONID'],
                'mistakes': len(mistakes),
                'TYPOS': return_number('TYPOS'),
                'GRAMMAR': return_number('GRAMMAR'),
                'TYPOGRAPHY': return_number('TYPOGRAPHY'),
                'PUNCTUATION': return_number('PUNCTUATION'),
                'REDUNDANCY': return_number('REDUNDANCY'),
                'MISC': return_number('MISC'),
                'CASING': return_number('CASING'),
                'STYLE': return_number('STYLE'),
                'CONFUSED_WORDS': return_number('CONFUSED_WORDS'),
                'COLLOCATIONS': return_number('COLLOCATIONS')
            }

        except:
            print(i, "error")
            df_temp = {
                'ID': loop_data.loc[i, 'ID'], 
                'APPLICANTID': loop_data.loc[i, 'APPLICANTID'], 
                'QUESTIONID': loop_data.loc[i, 'QUESTIONID'],
                'mistakes': len(mistakes),
                'TYPOS': np.nan,
                'GRAMMAR': np.nan,
                'TYPOGRAPHY': np.nan,
                'PUNCTUATION': np.nan,
                'REDUNDANCY': np.nan,
                'MISC': np.nan,
                'CASING': np.nan,
                'STYLE': np.nan,
                'CONFUSED_WORDS': np.nan,
                'COLLOCATIONS': np.nan
            }
    else:
        loop_data.loc[i, 'mistakes'] = 0
        df_temp = {
                'ID': loop_data.loc[i, 'ID'], 
                'APPLICANTID': loop_data.loc[i, 'APPLICANTID'], 
                'QUESTIONID': loop_data.loc[i, 'QUESTIONID'],
                'mistakes': len(mistakes),
                'TYPOS': np.nan,
                'GRAMMAR': np.nan,
                'TYPOGRAPHY': np.nan,
                'PUNCTUATION': np.nan,
                'REDUNDANCY': np.nan,
                'MISC': np.nan,
                'CASING': np.nan,
                'STYLE': np.nan,
                'CONFUSED_WORDS': np.nan,
                'COLLOCATIONS': np.nan
            }

    df1 = df1.append(df_temp, ignore_index = True)
    c += 1
    if c % 1000 == 0:
        print(round(i / len(loop_data['ANSWERTEXT']), 2)*100, i,len(loop_data['ANSWERTEXT']), datetime.datetime.now())
    if c % 10000 == 0:
        file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
        df1[df1['mistakes'].isna() == False].to_csv('s3://revcom-sagemaker-input-features/application/language_tool_python/{}'.format(file_name),
                  index = False)
        c = 0
        print("file saved", file_name)
    if i == (len(loop_data['ANSWERTEXT'])-1):
        file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
        df1[df1['mistakes'].isna() == False].to_csv('s3://revcom-sagemaker-input-features/application/language_tool_python/{}'.format(file_name),
                  index = False)
        c = 0
        print("file saved", file_name)
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

# fix random seed for reproducibility
np.random.seed(7)

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("snowflake-connector-python")
import snowflake.connector

install("pyspellchecker")
from spellchecker import SpellChecker
import re

# The Snowflake Connector library.
import snowflake.connector
import boto3

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

    for object_summary in my_bucket.objects.filter(Prefix="application/misspell/"):
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
             left_on=['ID', 'APPLICANTID', 'QUESTIONID'],
             right_on = ['ID', 'APPLICANTID', 'QUESTIONID'])
    loop_data = combination[combination['misspelled'].isna() == True]
    loop_data = loop_data.reset_index(drop=True)
    loop_data = loop_data.sort_values(by=['ID', 'APPLICANTID', 'QUESTIONID'])
    if loop_data.shape[0] == (df.shape[0] - prior_data.shape[0]):
        print("dims seems to work")
        print("loop_data", loop_data.shape[0], "prior_data", prior_data.shape, "df", df.shape)
    else:
        print("dims NOT work ----- please send help. prior data not being loaded")
        print("loop_data", loop_data.shape[0], "prior_data", prior_data.shape, "df", df.shape)
    
except: ## if there is no data to load, run all the data
    print("there was no prior data to exclude from the loop")
    loop_data = df

spell = SpellChecker(distance=1)
import time
start_time = time.time()

## after 100k loop save out csv and start with empty csv
#### saved into lang_detect folder
df2 = pd.DataFrame([])

if loop_data.shape[0] < 10:
    print("there is no reason to update data there are {} missing rows.".format(loop_data.shape[0]))
    
if loop_data.shape[0] > 10:
    for i in range(0, len(df.ID)):
        res = re.sub(r'-', ' ', loop_data.ANSWERTEXT[i])
        res = re.sub(r'[^\w\s]', '', res).split()

        # find those words that may be misspelled
        misspelled = spell.unknown(res)

        sent1 = pd.DataFrame(data = {'result': [res],
                                 'misspelled': [misspelled],
                                 'misspelled_ct': [len(misspelled)],
                                 'ID': [df.ID[i]],
                                 'APPLICANTID': [df.APPLICANTID[i]],
                                 'QUESTIONID': [df.QUESTIONID[i]]})
        df2 = df2.append(sent1)

        if i % 100000 == 0:
            file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
            df2[df2['misspelled_ct'].isna() == False].to_csv('s3://revcom-sagemaker-input-features/application/misspell/{}'.format(file_name),
                      index = False)
            print("file saved", file_name)
            df2 = pd.DataFrame([])

        if i == (df.shape[0]-1):
            file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
            df2[df2['misspelled_ct'].isna() == False].to_csv('s3://revcom-sagemaker-input-features/application/misspell/{}'.format(file_name),
                      index = False)
            print("file saved", file_name)
            df2 = pd.DataFrame([])

print("DONE!")
end_time = time.time()
print(end_time - start_time)

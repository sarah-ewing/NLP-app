import time
start_time = time.time()

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

install("textdistance")
from textdistance import levenshtein

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

print("load data from the database")
SQL_query = """
SELECT DISTINCT
        TCSA.ID,
        TCSA.applicantid,
        TCSA.questionid,
        TRIM(TCSA.answertext) AS answertext,
        TRIM(AQ.QUESTION) as QUESTION
    FROM LAKE_REVCOM.DBO.TCAPPLICANTSHORTANSWERRESPONSES TCSA
    LEFT JOIN ZDEV_LAKE_REVCOM.DBO.APPLICANTQUESTIONS AQ
        ON AQ.ID = TCSA.questionid
    WHERE 
        answertext IS NOT NULL 
        AND answertext != ''
        AND questionid in (94, 224, 225)
    ORDER BY ID, questionid;   
"""

cur = ctx.cursor().execute(SQL_query)
df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])
ctx.cursor().close()
print("format data")
df['ANSWERTEXT'] = df['ANSWERTEXT'].astype(str)
df['QUESTION'] = df['QUESTION'].astype(str)
df = df.dropna(subset=['ID', 'ANSWERTEXT', 'QUESTION'])
print("loaded dataframe", df.shape)

### load the prior data if available
try:
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('revcom-sagemaker-input-features')
    prior_data = pd.DataFrame()

    for object_summary in my_bucket.objects.filter(Prefix="application/distance/"):
        if object_summary.key.endswith(".csv"):
            df3 = pd.read_csv("s3://revcom-sagemaker-input-features/{}".format(object_summary.key))
            prior_data = prior_data.append(df3, ignore_index=True)
        print(object_summary.key, prior_data.shape, datetime.datetime.now())
    prior_data = prior_data.drop_duplicates()
    print("done", prior_data.columns, prior_data.shape)
except:
    print("there is no prior data to load.", prior_data.shape, df.shape, datetime.datetime.now())

## if prior data loaded then merge to df and do not re-calc
try:
    ## define data not calculated - then send them into the loop
    combination = df.merge(prior_data,
             how='left',
             left_on=['ID', 'APPLICANTID', 'QUESTIONID'],
             right_on = ['ID', 'APPLICANTID', 'QUESTIONID'])
    print("combination", combination.columns)
    loop_data = combination[combination['DISTANCE'].isna() == True]
    loop_data = loop_data.drop_duplicates()
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

print("loop data records:", len(loop_data['ID']), loop_data.shape[0])
## loop to update data

# loop_data['ANSWERTEXT'] = loop_data['ANSWERTEXT'].to_string()
# loop_data['QUESTION'] = loop_data['QUESTION'].to_string()
# loop_data = loop_data.dropna(subset=['ID', 'ANSWERTEXT', 'QUESTION'])

if loop_data.shape[0] <= 10:
    print("there is no reason to update data there are {} missing rows.".format(loop_data.shape[0]))

i = 0    
print(i, loop_data['ID'].loc[i], loop_data['ANSWERTEXT'].loc[i], loop_data['QUESTION'].loc[i])    

c = 0
if loop_data.shape[0] > 10:
    df1 = pd.DataFrame([])
    for i in range(0, len(loop_data['ID'])):
        
#         print(i, "of", len(loop_data['ID']), datetime.datetime.now())
#         print(i, loop_data['ANSWERTEXT'][i], loop_data['QUESTION'][i])
        
        out = levenshtein.distance(loop_data['ANSWERTEXT'][i], loop_data['QUESTION'][i])      
        
        ## add an extra row for the weekly perdiction into the future
        df_temp = {'ID': loop_data['ID'].loc[i], 
               'APPLICANTID': loop_data['APPLICANTID'].loc[i], 
               'QUESTIONID': loop_data['QUESTIONID'].loc[i],
               'DISTANCE': out}
        df1 = df1.append(df_temp, ignore_index = True)
        c += 1
       
        if c == 10000:
            file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
            df1[df1['DISTANCE'].isna() == False].to_csv('s3://revcom-sagemaker-input-features/application/distance/{}'.format(file_name), index = False)
            print(i, "file saved", file_name, datetime.datetime.now())
            df1 = pd.DataFrame([])
            c = 0
            
        if i == (len(loop_data['ID'])-1):
            file_name = '{date}.csv'.format(date = str(datetime.datetime.today())[0:19].replace(" ", "_"))
            df1[df1['DISTANCE'].isna() == False].to_csv('s3://revcom-sagemaker-input-features/application/distance/{}'.format(file_name), index = False)
            print("file saved", file_name, datetime.datetime.now())
            df1 = pd.DataFrame([])
            
print("DONE!")
end_time = time.time()
print(end_time - start_time)
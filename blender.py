import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
jp = pd.read_csv('blended/HomeCreditDefaultSubmitJP.csv')
dr = pd.read_csv('blended/Dromosys_submission_kernel02.csv')
kx = pd.read_csv('blended/kxx_tidy_xgb_0.78889.csv')
ol = pd.read_csv('blended/olivier_submission_with selected_features.csv')
hk = pd.read_csv('blended/hammadkhan_blended.csv')
ij = pd.read_csv('blended/ishaan_jain_blended.csv')
yk = pd.read_csv('blended/yoshiaki_blended.csv')

df = pd.merge(jp,dr,on= 'SK_ID_CURR')
df = pd.merge(df,kx,on ='SK_ID_CURR')
df = pd.merge(df,ol,on ='SK_ID_CURR')
df = pd.merge(df,hk,on ='SK_ID_CURR')
df = pd.merge(df,ij,on ='SK_ID_CURR')
df = pd.merge(df,yk,on ='SK_ID_CURR')
df.columns = ['SK_ID_CURR','jp','dr','kx','ol','hk','ij','yk']
df.head()

pred_prob = (0.1 * df['jp']) + (0.1 * df['dr']) + (0.1 * df['kx']) + (0.1 * df['ol']) + (0.25 * df['hk']) + (0.25 * df['ij']) + (0.27 * df['yk'])
pred_prob.head()

sub = pd.DataFrame()
sub['SK_ID_CURR'] = df['SK_ID_CURR']
sub['target']= pred_prob
sub.to_csv('blended.csv', index=False)


import numpy as np 
import pandas as pd
jp = pd.read_csv('blended/HomeCreditDefaultSubmitJP.csv')
dr = pd.read_csv('blended/Dromosys_submission_kernel02.csv')
kx = pd.read_csv('blended/kxx_tidy_xgb_0.78889.csv')
ol = pd.read_csv('blended/olivier_submission_with selected_features.csv')

df = pd.merge(jp,dr,on= 'SK_ID_CURR')
df = pd.merge(df,kx,on ='SK_ID_CURR')
df = pd.merge(df,ol,on ='SK_ID_CURR')
df.columns = ['SK_ID_CURR','T1','T2','T3','T4']
df.head()

pred_prob = 0.1 * df['T1'] + 0.3 * df['T2'] + 0.4 * df['T3'] + 0.05 * df['T4']
pred_prob.head()

sub = pd.DataFrame()
sub['SK_ID_CURR'] = df['SK_ID_CURR']
sub['target']= pred_prob
sub.to_csv('blended.csv', index=False)


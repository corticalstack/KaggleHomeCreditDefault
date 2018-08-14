import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True, drop_first = False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, drop_first = drop_first)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False, drop_first = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    
   
    
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan - Associated with pensioner
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)    
    
    # Some simple new features 
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
        
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category, drop_first = True)
    
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)  
    del test_df
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True, drop_first = False):
    bureau = pd.read_csv('bureau.csv', nrows = num_rows)
    bb = pd.read_csv('bureau_balance.csv', nrows = num_rows) 


    bb, bb_cat = one_hot_encoder(bb, nan_as_category, drop_first = drop_first)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category, drop_first = drop_first)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
      
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True, drop_first = False):
    prev = pd.read_csv('previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True, drop_first = drop_first)
    
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': ['max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True, drop_first = True):
    pos = pd.read_csv('POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True, drop_first = drop_first)
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True, drop_first = True):
    ins = pd.read_csv('installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True, drop_first = drop_first)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': [ 'min','max', 'mean'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True, drop_first = True):
    cc = pd.read_csv('credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True, drop_first = drop_first)

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    
    dropcolum=['CC_NAME_CONTRACT_STATUS_Refused_MAX',
    'CC_NAME_CONTRACT_STATUS_Refused_VAR',
    'FLAG_EMP_PHONE',
    'FLAG_MOBIL',
    'CC_NAME_CONTRACT_STATUS_Refused_SUM',
    'CC_NAME_CONTRACT_STATUS_Refused_MIN',
    'CC_NAME_CONTRACT_STATUS_Refused_MEAN',
    'CC_CNT_DRAWINGS_ATM_CURRENT_MIN',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'BURO_STATUS_nan_MEAN_MEAN',
    'CC_SK_DPD_DEF_MIN',
    'CC_NAME_CONTRACT_STATUS_Demand_VAR',
    'NAME_FAMILY_STATUS_Unknown',
    'CC_NAME_CONTRACT_STATUS_Demand_SUM',
    'NAME_INCOME_TYPE_Maternity leave',
    'PREV_NAME_YIELD_GROUP_nan_MEAN',
    'NAME_INCOME_TYPE_Pensioner',
    'NAME_INCOME_TYPE_Student',
    'NAME_INCOME_TYPE_Unemployed',
    'NAME_TYPE_SUITE_Group of people',
    'NAME_TYPE_SUITE_Other_A',
    'CC_NAME_CONTRACT_STATUS_Demand_MIN',
    'CC_NAME_CONTRACT_STATUS_Demand_MEAN',
    'CC_NAME_CONTRACT_STATUS_Demand_MAX',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN',
    'FLAG_CONT_MOBILE',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM',
    'HOUSETYPE_MODE_terraced house',
    'WALLSMATERIAL_MODE_Monolithic',
    'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN',
    'CC_NAME_CONTRACT_STATUS_nan_MAX',
    'CC_NAME_CONTRACT_STATUS_nan_MEAN',
    'BURO_CREDIT_CURRENCY_nan_MEAN',
    'BURO_CREDIT_CURRENCY_currency 4_MEAN',
    'BURO_CREDIT_CURRENCY_currency 3_MEAN',
    'BURO_CREDIT_CURRENCY_currency 2_MEAN',
    'BURO_CREDIT_CURRENCY_currency 1_MEAN',
    'BURO_CREDIT_ACTIVE_nan_MEAN',
    'CC_NAME_CONTRACT_STATUS_nan_MIN',
    'CC_NAME_CONTRACT_STATUS_nan_SUM',
    'INSTAL_DPD_MIN',
    'CC_SK_DPD_MIN',
    'CC_NAME_CONTRACT_STATUS_nan_VAR',
    'BURO_CREDIT_ACTIVE_Bad debt_MEAN',
    'CC_NAME_CONTRACT_STATUS_Signed_MIN',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN',
    'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN',
    'CC_SK_DPD_DEF_MAX',
    'WALLSMATERIAL_MODE_Wooden',
    'PREV_PRODUCT_COMBINATION_nan_MEAN',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR',
    'BURO_CREDIT_TYPE_nan_MEAN',
    'BURO_CREDIT_TYPE_Unknown type of loan_MEAN',
    'BURO_CREDIT_TYPE_Real estate loan_MEAN',
    'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
    'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
    'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN',
    'CLOSED_CREDIT_DAY_OVERDUE_MEAN',
    'CLOSED_CREDIT_DAY_OVERDUE_MAX',
    'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN',
    'BURO_CREDIT_TYPE_Interbank credit_MEAN',
    'CC_NAME_CONTRACT_STATUS_Signed_MAX',
    'POS_NAME_CONTRACT_STATUS_XNA_MEAN',
    'CC_NAME_CONTRACT_STATUS_Completed_MIN',
    'ORGANIZATION_TYPE_Postal',
    'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN',
    'ORGANIZATION_TYPE_Services',
    'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN',
    'ORGANIZATION_TYPE_Religion',
    'ORGANIZATION_TYPE_Realtor',
    'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN',
    'ORGANIZATION_TYPE_Mobile',
    'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN',
    'ORGANIZATION_TYPE_Legal Services',
    'PREV_NAME_CLIENT_TYPE_XNA_MEAN',
    'PREV_NAME_CLIENT_TYPE_nan_MEAN',
    'ORGANIZATION_TYPE_Insurance',
    'ORGANIZATION_TYPE_Industry: type 8',
    'ORGANIZATION_TYPE_Industry: type 7',
    'PREV_NAME_CONTRACT_STATUS_nan_MEAN',
    'ORGANIZATION_TYPE_Industry: type 6',
    'ORGANIZATION_TYPE_Telecom',
    'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_nan_MEAN',
    'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN',
    'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
    'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
    'ORGANIZATION_TYPE_XNA',
    'PREV_CODE_REJECT_REASON_SYSTEM_MEAN',
    'ORGANIZATION_TYPE_Transport: type 2',
    'ORGANIZATION_TYPE_Transport: type 1',
    'PREV_CODE_REJECT_REASON_nan_MEAN',
    'ORGANIZATION_TYPE_Trade: type 6',
    'ORGANIZATION_TYPE_Trade: type 5',
    'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
    'ORGANIZATION_TYPE_Trade: type 4',
    'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN',
    'PREV_CHANNEL_TYPE_nan_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN',
    'ORGANIZATION_TYPE_Trade: type 1',
    'POS_NAME_CONTRACT_STATUS_nan_MEAN',
    'ORGANIZATION_TYPE_Industry: type 4',
    'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
    'PREV_NAME_PRODUCT_TYPE_nan_MEAN',
    'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN',
    'OCCUPATION_TYPE_Secretaries',
    'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
    'OCCUPATION_TYPE_Realty agents',
    'PREV_NAME_PORTFOLIO_Cars_MEAN',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN',
    'PREV_CHANNEL_TYPE_Car dealer_MEAN',
    'PREV_NAME_PORTFOLIO_nan_MEAN',
    'POS_NAME_CONTRACT_STATUS_Demand_MEAN',
    'PREV_NAME_CONTRACT_TYPE_nan_MEAN',
    'OCCUPATION_TYPE_HR staff',
    'CC_NAME_CONTRACT_STATUS_Approved_MAX',
    'CC_NAME_CONTRACT_STATUS_Approved_MEAN',
    'CC_NAME_CONTRACT_STATUS_Approved_MIN',
    'CC_NAME_CONTRACT_STATUS_Approved_SUM',
    'CC_NAME_CONTRACT_STATUS_Approved_VAR',
    'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN',
    'ORGANIZATION_TYPE_Agriculture',
    'PREV_NAME_GOODS_CATEGORY_nan_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Weapon_MEAN',
    'ORGANIZATION_TYPE_Cleaning',
    'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
    'ORGANIZATION_TYPE_Industry: type 2',
    'ORGANIZATION_TYPE_Industry: type 13',
    'ORGANIZATION_TYPE_Industry: type 12',
    'ORGANIZATION_TYPE_Industry: type 11',
    'ORGANIZATION_TYPE_Industry: type 10',
    'PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Education_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Fitness_MEAN',
    'ORGANIZATION_TYPE_Industry: type 1',
    'ORGANIZATION_TYPE_Housing',
    'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN',
    'ORGANIZATION_TYPE_Emergency',
    'PREV_NAME_GOODS_CATEGORY_Medicine_MEAN',
    'ORGANIZATION_TYPE_Culture',
    'OCCUPATION_TYPE_IT staff',
    'ORGANIZATION_TYPE_Trade: type 2',
    'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN',
    'ORGANIZATION_TYPE_Security',
    'PREV_NAME_GOODS_CATEGORY_Other_MEAN',
    'PREV_CODE_REJECT_REASON_CLIENT_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN',
    'CC_NAME_CONTRACT_STATUS_Completed_MAX',
    'INSTAL_NUM_INSTALMENT_NUMBER_MIN',
    'PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN',
    'ORGANIZATION_TYPE_University',
    'ORGANIZATION_TYPE_Restaurant',
    'ORGANIZATION_TYPE_Business Entity Type 2',
    'PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN',
    'PREV_CODE_REJECT_REASON_XNA_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN',
    'PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN',
    'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN',
    'CC_NAME_CONTRACT_STATUS_Signed_SUM',
    'OCCUPATION_TYPE_Cooking staff',
    'CC_AMT_DRAWINGS_ATM_CURRENT_MIN',
    'PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX',
    'BURO_CNT_CREDIT_PROLONG_SUM',
    'REG_REGION_NOT_LIVE_REGION',
    'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN',
    'BURO_STATUS_5_MEAN_MEAN',
    'CC_MONTHS_BALANCE_MAX',
    'FONDKAPREMONT_MODE_reg oper account',
    'ORGANIZATION_TYPE_Industry: type 3',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN',
    'ORGANIZATION_TYPE_Trade: type 3',
    'CC_AMT_TOTAL_RECEIVABLE_MIN',
    'CC_AMT_TOTAL_RECEIVABLE_SUM',
    'NAME_HOUSING_TYPE_Rented apartment',
    'CC_NAME_CONTRACT_STATUS_Completed_VAR',
    'CC_AMT_TOTAL_RECEIVABLE_VAR',
    'WEEKDAY_APPR_PROCESS_START_THURSDAY',
    'BURO_CREDIT_DAY_OVERDUE_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
    'NAME_HOUSING_TYPE_With parents',
    'CC_NAME_CONTRACT_STATUS_Completed_SUM',
    'ORGANIZATION_TYPE_Industry: type 5',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN',
    'PREV_NAME_TYPE_SUITE_Group of people_MEAN',
    'OCCUPATION_TYPE_Cleaning staff',
    'CC_SK_DPD_DEF_VAR',
    'HOUSETYPE_MODE_specific housing',
    'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',
    'CLOSED_CNT_CREDIT_PROLONG_SUM',
    'PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN',
    'FLAG_OWN_CAR',
    'FLAG_EMAIL',
    'PREV_PRODUCT_COMBINATION_POS others without interest_MEAN',
    'ORGANIZATION_TYPE_Trade: type 7',
    'OCCUPATION_TYPE_Managers',
    'CC_SK_DPD_MAX',
    'CC_CNT_DRAWINGS_CURRENT_MIN',
    'ORGANIZATION_TYPE_Restaurant',
    'CC_AMT_INST_MIN_REGULARITY_MIN',
    'FONDKAPREMONT_MODE_org spec account',
    'CC_SK_DPD_DEF_SUM',
    'PREV_NAME_GOODS_CATEGORY_Tourism_MEAN',
    'OCCUPATION_TYPE_Security staff',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN',
    'BURO_STATUS_3_MEAN_MEAN',
    'ORGANIZATION_TYPE_Security',
    'ORGANIZATION_TYPE_Electricity',
    'PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN',
    'WALLSMATERIAL_MODE_Mixed',
    'CC_NAME_CONTRACT_STATUS_Signed_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN',
    'BURO_CREDIT_TYPE_Another type of loan_MEAN',
    'NAME_HOUSING_TYPE_Municipal apartment',
    'BURO_MONTHS_BALANCE_MAX_MAX',
    'CC_AMT_RECIVABLE_SUM',
    'FLOORSMIN_MEDI',
    'ORGANIZATION_TYPE_Police',
    'PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN',
    'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN',
    'CC_CNT_INSTALMENT_MATURE_CUM_MAX',
    'ORGANIZATION_TYPE_Security Ministries',
    'CC_AMT_DRAWINGS_POS_CURRENT_MIN',
    'FLOORSMAX_MODE',
    'REGION_RATING_CLIENT',
    'FLOORSMAX_MEDI',
    'POS_COUNT',
    'PREV_PRODUCT_COMBINATION_POS other with interest_MEAN',
    'NAME_TYPE_SUITE_Unaccompanied',
    'NAME_FAMILY_STATUS_Single / not married',
    'WEEKDAY_APPR_PROCESS_START_SUNDAY',
    'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN',
    'WEEKDAY_APPR_PROCESS_START_TUESDAY',
    'ELEVATORS_MODE',
    'CC_AMT_RECEIVABLE_PRINCIPAL_MIN',
    'BURO_CREDIT_DAY_OVERDUE_MAX',
    'PREV_NAME_GOODS_CATEGORY_Gardening_MEAN',
    'NONLIVINGAPARTMENTS_MEDI',
    'PREV_NAME_TYPE_SUITE_Other_B_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN',
    'CC_NAME_CONTRACT_STATUS_Signed_VAR',
    'WALLSMATERIAL_MODE_Panel',
    'ORGANIZATION_TYPE_Medicine',
    'CC_SK_DPD_VAR',
    'PREV_CODE_REJECT_REASON_VERIF_MEAN',
    'ELEVATORS_MEDI',
    'OCCUPATION_TYPE_Low-skill Laborers',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'BURO_STATUS_2_MEAN_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN',
    'OCCUPATION_TYPE_Sales staff',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_MAX',
    'CC_AMT_INST_MIN_REGULARITY_MAX',
    'NAME_EDUCATION_TYPE_Lower secondary',
    'PREV_NAME_SELLER_INDUSTRY_Industry_MEAN',
    'BURO_STATUS_4_MEAN_MEAN',
    'NAME_EDUCATION_TYPE_Incomplete higher',
    'FONDKAPREMONT_MODE_reg oper spec account',
    'CC_NAME_CONTRACT_STATUS_Completed_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Auto technology_MEAN',
    'NAME_FAMILY_STATUS_Separated',
    'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM',
    'NAME_CONTRACT_TYPE_Revolving loans',
    'NAME_TYPE_SUITE_Other_B',
    'CC_SK_DPD_MEAN',
    'ORGANIZATION_TYPE_Other',
    'REG_CITY_NOT_WORK_CITY',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_VAR',
    'ORGANIZATION_TYPE_Government',
    'INSTAL_PAYMENT_PERC_STD',
    'PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN',
    'NAME_TYPE_SUITE_Spouse, partner',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_VAR',
    'WALLSMATERIAL_MODE_Others',
    'CC_AMT_PAYMENT_TOTAL_CURRENT_MIN',
    'CLOSED_AMT_CREDIT_SUM_LIMIT_SUM',
    'CC_COUNT',
    'CC_AMT_TOTAL_RECEIVABLE_MAX',
    'ORGANIZATION_TYPE_Transport: type 4',
    'INSTAL_PAYMENT_DIFF_STD',
    'ORGANIZATION_TYPE_Hotel',
    'ORGANIZATION_TYPE_Business Entity Type 1',
    'NAME_TYPE_SUITE_Family',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_REGION_NOT_WORK_REGION',
    'PREV_NAME_TYPE_SUITE_Other_A_MEAN',
    'CC_AMT_TOTAL_RECEIVABLE_MEAN',
    'NAME_FAMILY_STATUS_Widow',
    'POS_NAME_CONTRACT_STATUS_Approved_MEAN',
    'OCCUPATION_TYPE_Waiters/barmen staff',
    'CC_MONTHS_BALANCE_MIN',
    'CC_CNT_INSTALMENT_MATURE_CUM_MIN',
    'CC_AMT_INST_MIN_REGULARITY_SUM',
    'PREV_PRODUCT_COMBINATION_Card Street_MEAN',
    'CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Sport and Leisure_MEAN',
    'CC_CNT_DRAWINGS_POS_CURRENT_MAX',
    'CC_CNT_DRAWINGS_POS_CURRENT_VAR',
    'CC_AMT_CREDIT_LIMIT_ACTUAL_MIN',
    'PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN',
    'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX',
    'WALLSMATERIAL_MODE_Stone, brick',
    'PREV_PRODUCT_COMBINATION_Cash Street: middle_MEAN',
    'INSTAL_PAYMENT_DIFF_MIN',
    'OCCUPATION_TYPE_High skill tech staff',
    'CC_MONTHS_BALANCE_MEAN',
    'NAME_HOUSING_TYPE_Office apartment',
    'ENTRANCES_MODE',
    'REFUSED_AMT_DOWN_PAYMENT_MAX',
    'CC_AMT_DRAWINGS_CURRENT_MAX',
    'WEEKDAY_APPR_PROCESS_START_MONDAY',
    'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'CC_AMT_BALANCE_MAX',
    'CLOSED_AMT_CREDIT_SUM_DEBT_MEAN',
    'CC_CNT_INSTALMENT_MATURE_CUM_VAR',
    'CC_AMT_RECEIVABLE_PRINCIPAL_VAR',
    'FLOORSMIN_AVG',
    'CC_AMT_RECIVABLE_VAR',
    'OCCUPATION_TYPE_Laborers',
    'YEARS_BUILD_MEDI',
    'CC_AMT_RECEIVABLE_PRINCIPAL_MAX',
    'PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN',
    'CC_AMT_INST_MIN_REGULARITY_MEAN',
    'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX',
    'CC_CNT_INSTALMENT_MATURE_CUM_MEAN',
    'CLOSED_MONTHS_BALANCE_MAX_MAX',
    'OCCUPATION_TYPE_Medicine staff',
    'CC_MONTHS_BALANCE_SUM',
    'CC_AMT_RECIVABLE_MAX',
    'REFUSED_AMT_DOWN_PAYMENT_MEAN',
    'CC_MONTHS_BALANCE_VAR',
    'CC_AMT_DRAWINGS_POS_CURRENT_MEAN',
    'PREV_CODE_REJECT_REASON_SCOFR_MEAN',
    'ELEVATORS_AVG',
    'CC_SK_DPD_SUM',
    'FLOORSMIN_MODE',
    'CC_CNT_DRAWINGS_POS_CURRENT_SUM',
    'NAME_HOUSING_TYPE_House / apartment',
    'INSTAL_PAYMENT_PERC_MAX',
    'LIVE_CITY_NOT_WORK_CITY',
    'PREV_NAME_GOODS_CATEGORY_Construction Materials_MEAN',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'CC_AMT_RECEIVABLE_PRINCIPAL_SUM',
    'NAME_INCOME_TYPE_Commercial associate',
    'FLAG_OWN_REALTY',
    'NONLIVINGAPARTMENTS_MODE',
    'PREV_PRODUCT_COMBINATION_POS industry without interest_MEAN',
    'ORGANIZATION_TYPE_Kindergarten',
    'WEEKDAY_APPR_PROCESS_START_SATURDAY',
    'FLAG_PHONE',
    'CC_CNT_DRAWINGS_POS_CURRENT_MIN',
    'CC_CNT_INSTALMENT_MATURE_CUM_SUM',
    'PREV_NAME_TYPE_SUITE_Children_MEAN',
    'CC_AMT_BALANCE_SUM',
    'PREV_NAME_SELLER_INDUSTRY_Construction_MEAN',
    'CNT_CHILDREN',
    'WEEKDAY_APPR_PROCESS_START_WEDNESDAY',
    'PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
    'CC_AMT_DRAWINGS_CURRENT_MIN',
    'ORGANIZATION_TYPE_Bank',
    'PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN',
    'CC_CNT_DRAWINGS_ATM_CURRENT_MAX',    
    'CC_AMT_RECIVABLE_MEAN',
    'CC_AMT_RECIVABLE_MIN',
    'REFUSED_RATE_DOWN_PAYMENT_MAX',
    'ENTRANCES_MEDI',
    'PREV_CODE_REJECT_REASON_LIMIT_MEAN',
    'BURO_CREDIT_ACTIVE_Sold_MEAN',
    'CC_AMT_DRAWINGS_POS_CURRENT_VAR',
    'CC_AMT_PAYMENT_TOTAL_CURRENT_VAR',
    'PREV_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_MEAN',
    'CC_AMT_DRAWINGS_CURRENT_SUM',
    'NEW_LIVE_IND_SUM',
    'CC_AMT_DRAWINGS_POS_CURRENT_MAX',
    'CC_CNT_DRAWINGS_ATM_CURRENT_SUM',
    'POS_NAME_CONTRACT_STATUS_Returned to the store_MEAN',
    'CC_AMT_BALANCE_VAR',
    'CNT_FAM_MEMBERS',
    'FLOORSMAX_AVG',
    'REFUSED_RATE_DOWN_PAYMENT_MEAN',
    'NAME_INCOME_TYPE_State servant',
    'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
    'PREV_NAME_GOODS_CATEGORY_XNA_MEAN',
    'NONLIVINGAPARTMENTS_AVG',
    'CC_AMT_BALANCE_MIN',
    'ORGANIZATION_TYPE_Industry: type 9'
    ]
    df= df.drop(dropcolum,axis=1)
    
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]  
    
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf = LGBMClassifier(
            nthread=4,
            #scale_pos_weight = 1.3,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )       

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    file_name = 'features importance output.csv'
    
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    cols1 = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    cols1.to_csv(file_name, sep=',', encoding='utf-8')
    
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

    
def main(debug = False):
    num_rows = 10000 if debug else None
    
    with timer("Process application train and test"):
        df = application_train_test(num_rows)
        print("Dataset df shape:", df.shape)
        gc.collect()
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "HomeCreditDefaultSubmit.csv"
    with timer("Full model run"):
        main()
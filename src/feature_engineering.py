import os
import gc
import pandas as pd
import numpy as np
from src.data_utils import reduce_mem_usage, one_hot_encoder

def process_bureau_and_balance(path, nan_as_category=True):
    bureau_balance_path = os.path.join(path, 'bureau_balance.csv')
    bureau_path = os.path.join(path, 'bureau.csv')
    
    if not os.path.exists(bureau_balance_path) or not os.path.exists(bureau_path):
        print("Bureau or Bureau Balance files not found. Skipping...")
        return pd.DataFrame()

    # Process bureau_balance
    bb = pd.read_csv(bureau_balance_path)
    bb = reduce_mem_usage(bb, verbose=False)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    
    # Bureau balance aggregations
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bb_agg = bb_agg.reset_index()
    
    del bb
    gc.collect()
    
    # Process bureau
    bureau = pd.read_csv(bureau_path)
    bureau = reduce_mem_usage(bureau, verbose=False)
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    
    del bb_agg
    gc.collect()
    
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau aggregations
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    
    # For status cols inside bureau balance
    for col in bureau.columns:
        if col.startswith('STATUS_') and col not in num_aggregations:
            num_aggregations[col] = ['mean']
            
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
        
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    bureau_agg = bureau_agg.reset_index()
    
    del bureau
    gc.collect()
    return bureau_agg

def process_previous_applications(path, nan_as_category=True):
    prev_path = os.path.join(path, 'previous_application.csv')
    if not os.path.exists(prev_path):
        print("Previous Application file not found. Skipping...")
        return pd.DataFrame()

    prev = pd.read_csv(prev_path)
    prev = reduce_mem_usage(prev, verbose=False)
    
    # Replace anomalous values
    days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols:
        prev[col] = prev[col].replace(365243, np.nan)
        
    # Add engineered ratio features
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT'] + 1e-5)
    
    prev, prev_cat = one_hot_encoder(prev, nan_as_category)
    
    # Previous applications aggregations
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE': ['min', 'max', 'mean'],
        'DAYS_TERMINATION': ['min', 'max', 'mean']
    }
    
    cat_aggregations = {}
    for cat in prev_cat:
        cat_aggregations[cat] = ['mean']
        
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    prev_agg = prev_agg.reset_index()
    
    del prev
    gc.collect()
    return prev_agg

def process_pos_cash(path, nan_as_category=True):
    pos_path = os.path.join(path, 'POS_CASH_balance.csv')
    if not os.path.exists(pos_path):
        print("POS CASH Balance file not found. Skipping...")
        return pd.DataFrame()

    pos = pd.read_csv(pos_path)
    pos = reduce_mem_usage(pos, verbose=False)
    
    pos, pos_cat = one_hot_encoder(pos, nan_as_category)
    
    # POS CASH aggregations
    num_aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'CNT_INSTALMENT': ['max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    
    cat_aggregations = {}
    for cat in pos_cat:
        cat_aggregations[cat] = ['mean']
        
    pos_agg = pos.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg = pos_agg.reset_index()
    
    del pos
    gc.collect()
    return pos_agg

def process_installments_payments(path, nan_as_category=True):
    ins_path = os.path.join(path, 'installments_payments.csv')
    if not os.path.exists(ins_path):
        print("Installments Payments file not found. Skipping...")
        return pd.DataFrame()

    ins = pd.read_csv(ins_path)
    ins = reduce_mem_usage(ins, verbose=False)
    
    # Delayed payment and unpaid amount
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    ins['PAYMENT_DIFFERENCE'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / (ins['AMT_INSTALMENT'] + 1e-5)
    
    ins, ins_cat = one_hot_encoder(ins, nan_as_category)
    
    num_aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_DIFFERENCE': ['max', 'mean', 'sum'],
        'PAYMENT_RATIO': ['min', 'max', 'mean']
    }
    
    cat_aggregations = {}
    for cat in ins_cat:
        cat_aggregations[cat] = ['mean']
        
    ins_agg = ins.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg = ins_agg.reset_index()
    
    del ins
    gc.collect()
    return ins_agg

def process_credit_card(path, nan_as_category=True):
    cc_path = os.path.join(path, 'credit_card_balance.csv')
    if not os.path.exists(cc_path):
        print("Credit Card Balance file not found. Skipping...")
        return pd.DataFrame()

    cc = pd.read_csv(cc_path)
    cc = reduce_mem_usage(cc, verbose=False)
    
    cc, cc_cat = one_hot_encoder(cc, nan_as_category)
    
    # Credit Card balance aggregations
    cc_aggregations = {}
    for col in cc.columns:
        if col not in ['SK_ID_CURR', 'SK_ID_PREV'] and col not in cc_cat:
            cc_aggregations[col] = ['min', 'max', 'mean', 'sum', 'var']
            
    for cat in cc_cat:
        cc_aggregations[cat] = ['mean']
        
    cc_agg = cc.groupby('SK_ID_CURR').agg(cc_aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg = cc_agg.reset_index()
    
    del cc
    gc.collect()
    return cc_agg

def process_application_train_test(path, nan_as_category=True):
    train_path = os.path.join(path, 'application_train.csv')
    test_path = os.path.join(path, 'application_test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Core application_train.csv or application_test.csv not found.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Save target variable
    targets = df_train['TARGET']
    df_train.drop(['TARGET'], axis=1, inplace=True)
    
    train_len = len(df_train)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    del df_train, df_test
    gc.collect()
    
    df = reduce_mem_usage(df, verbose=False)
    
    # Handle abnormal DAYS_EMPLOYED
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Basic engineered ratios
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1e-5)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
    
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    return df, train_len, targets

def run_pipeline(path, output_path, nan_as_category=True):
    print("Starting data engineering pipeline...")
    
    bureau = process_bureau_and_balance(path, nan_as_category)
    prev = process_previous_applications(path, nan_as_category)
    pos = process_pos_cash(path, nan_as_category)
    ins = process_installments_payments(path, nan_as_category)
    cc = process_credit_card(path, nan_as_category)
    
    df, train_len, targets = process_application_train_test(path, nan_as_category)
    
    print("Merging tables...")
    if not bureau.empty:
        df = df.merge(bureau, on='SK_ID_CURR', how='left')
        del bureau; gc.collect()
    
    if not prev.empty:
        df = df.merge(prev, on='SK_ID_CURR', how='left')
        del prev; gc.collect()
        
    if not pos.empty:
        df = df.merge(pos, on='SK_ID_CURR', how='left')
        del pos; gc.collect()
        
    if not ins.empty:
        df = df.merge(ins, on='SK_ID_CURR', how='left')
        del ins; gc.collect()
        
    if not cc.empty:
        df = df.merge(cc, on='SK_ID_CURR', how='left')
        del cc; gc.collect()
        
    # Split back into train and test
    train_df = df.iloc[:train_len].copy()
    test_df = df.iloc[train_len:].copy()
    
    # Restore target
    train_df['TARGET'] = targets.values
    
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    os.makedirs(output_path, exist_ok=True)
    
    print("Saving processed dataframes to parquet...")
    train_df.to_parquet(os.path.join(output_path, 'train_features.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_path, 'test_features.parquet'), index=False)
    
    print("Pipeline execution complete! Files saved to", output_path)

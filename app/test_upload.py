#%%
from supabase import create_client
import streamlit as st
import pandas as pd
from ff.db_operations import DataManage
from ff import general as ffgeneral

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

@st.cache_resource
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)

def convert_to_dict(df):
    upload = []
    for _, row in df.iterrows():
        cur_row = {}
        for c in df.columns:
            cur_row[c] = row[c]
        upload.append(cur_row)

    return upload


week = 2
set_year = 2022
pred_vers = 'sera1_rsq0_brier1_matt0_bayes'
ensemble_vers = 'random_kbest_sera1_rsq0_mse0_include2_kfold3'
std_dev_type = 'spline_pred_class80_q80_matt0_brier1_kfold3'

df = dm.read(f'''SELECT * 
                FROM Model_Predictions
                WHERE week={week}
                    AND year={set_year}
                    AND version='{pred_vers}'
                    AND ensemble_vers='{ensemble_vers}'
                    AND std_dev_type='{std_dev_type}'
                    AND pos !='K'
                    AND pos IS NOT NULL
                    AND player!='Ryan Griffin'
                    ''', 'Simulation')

df.to_csv('test.csv', index=False)

upload = convert_to_dict(df)
upload
supabase = init_connection()
supabase.table('model_predictions').insert(upload).execute() # inserting one record

# %%
supabase.table('model_predictions').select('*').execute()

# %%

#%%
from supabase import create_client
import json
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_resource
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)

supabase = init_connection()
df = pd.DataFrame({'num1': [10, 20, 30], 'num2': [40, 50, 60]})

upload = []
for _, row in df.iterrows():
    cur_row = {}
    for c in df.columns:
        cur_row[c] = int(row[c])
    upload.append(cur_row)

supabase.table('mytable').insert(upload).execute() # inserting one record

# %%
supabase.table('mytable').select('*').execute()

# %%

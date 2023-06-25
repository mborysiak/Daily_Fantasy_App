#%%
import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client

# Initialize connection.
# Uses st.cache_resource to only run once.
@st.cache_resource
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    return create_client(url, key)

supabase = init_connection()

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query():
    return supabase.table("mytable").select("*").execute()

def create_interactive_grid(data):
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        height=350, 
        width='100%',
        reload_data=False
    )

    data = grid_response['data']
    selected = grid_response['selected_rows'] 
    df = pd.DataFrame(selected) 

    return df

def create_plot(df):
    # Create a plot
    fig, ax = plt.subplots()
    ax.plot(df['Number 1'], df['Number 2'])

    # Display the plot using Streamlit
    return st.pyplot(fig)


def main():
    # Set page configuration
    st.set_page_config(layout="wide")
    st.title("My First Streamlit App")
    st.write("Welcome to my app!")
    
    col1, col2 = st.columns(2)
    data = run_query()
    data.columns = ['Number 1', 'Number 2']

    with col1:
        df = create_interactive_grid(data)

    with col2:
        create_plot(df)

if __name__ == '__main__':
    main()
# %%


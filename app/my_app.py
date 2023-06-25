#%%
import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from supabase import create_client, Client

# # Initialize connection.
# # Uses st.cache_resource to only run once.
# @st.cache_resource
# def init_connection():
#     url = st.secrets["supabase_url"]
#     key = st.secrets["supabase_key"]
#     return create_client(url, key)

# # Perform query.
# # Uses st.cache_data to only rerun when the query changes or after 10 min.
# @st.cache_data(ttl=600)
# def run_query():
#     supabase = init_connection()
#     return supabase.table("model_predictions").select("*").execute()

# def convert_to_df(data):
#     data_list = []
#     for row in data.data:
#         data_list.append(row.values())
#     df = pd.DataFrame(data_list, columns=row.keys())
#     return df

def run_query():
    return pd.read_csv('test.csv')

def create_interactive_grid(data):
    gb = GridOptionsBuilder.from_dataframe(data)
    # add pagination
    # gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        # fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        column_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        height=350, 
        width='100%',
        max_height=350,
        reload_data=False
    )

    data = grid_response['data']
    selected = grid_response['selected_rows'] 
    df = pd.DataFrame(selected) 

    return df

def create_plot(df):
    # Create a plot
    fig, ax = plt.subplots()
    ax.plot(df['num1'], df['num2'])

    # Display the plot using Streamlit
    return st.pyplot(fig)


def main():
    # Set page configuration
    st.set_page_config(layout="wide")
    st.title("My First Streamlit App")
    st.write("Welcome to my app!")
    
    col1, col2 = st.columns(2)

    data = run_query()
    # data = convert_to_df(data)

    # with col1:
    df = create_interactive_grid(data)
    st.write(data.shape)
    # with col2:
        # create_plot(df)

if __name__ == '__main__':
    main()
# %%


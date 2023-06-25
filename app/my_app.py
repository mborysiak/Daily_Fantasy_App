#%%
import streamlit as st
import pandas as pd
import random
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy
import sqlite3

def create_fake_data():
     # Create fake data
    data = []
    for i in range(10):
        num1 = i*2
        num2 = np.sin(num1)
        data.append((num1, num2))
    data = pd.DataFrame(data, columns=['Number 1', 'Number 2'])
    return data

def pull_data(db_path):
    # conn = st.experimental_connection('data_db', type='sql')
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('select * from data', conn)
    return df

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
    import os
    st.write(os.getcwd())
    # db_path = st.secrets['connections']
    db_path = '/app/Daily_Fantasy_App/app/data.sqlite3'
    st.write(db_path)
    
    col1, col2 = st.columns(2)
    data = pull_data(db_path)
    with col1:
        df = create_interactive_grid(data)

    with col2:
        create_plot(df)

if __name__ == '__main__':
    main()
# %%


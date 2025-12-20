#%%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import sqlite3
from zSim_Helper_Covar import FullLineupSim, RunSim
import streamlit_authenticator as stauth
from pathlib import Path
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, select, DateTime

year = 2025
week = 16
total_lineups = 10
db_name = 'Simulation_App.sqlite3'


#--------------------
# DB operations
#-------------------

def generate_postgres_url():
    secrets = st.secrets["connections"]["postgresql"]
    dialect = secrets["dialect"]
    host = secrets["host"]
    port = secrets["port"]
    database = secrets["database"]
    username = secrets["username"]
    password = secrets["password"]
    
    return f"{dialect}://{username}:{password}@{host}:{port}/{database}"

def get_engine():
    url = generate_postgres_url()
    return create_engine(url)


# def create_users_table():
#     engine = get_engine()
#     metadata = MetaData()
    
#     users_table = Table(
#         'usersdb', metadata,
#         Column('key', String, primary_key=True),
#         Column('name', String, nullable=False),
#         Column('password', String, nullable=False)
#     )
    
#     metadata.create_all(engine)

# def create_results_table():
#     engine = get_engine()
#     metadata = MetaData()
    
#     results_table = Table(
#         'resultsdb', metadata,
#         Column('id', String, nullable=False),
#         Column('created_at', DateTime, nullable=False),
#         Column('user', String, nullable=False),
#         Column('week', Integer, nullable=False),
#         Column('year', Integer, nullable=False),
#         Column('pos', String, nullable=False),
#         Column('player', String, nullable=False)
#     )
    
#     metadata.create_all(engine)

# # Call the function to create the table
# create_results_table()
# create_users_table()

def pull_user_list():
    engine = get_engine()
    metadata = MetaData()
    users_table = Table('usersdb', metadata, autoload_with=engine)
    
    with engine.connect() as conn:
        query = select([users_table])
        result = conn.execute(query)
        users = result.fetchall()
    
    credentials = {'usernames': {}}
    for user in users:
        credentials['usernames'][user['key']] = {
            'email': None, 
            'name': user['name'], 
            'password': user['password']
        }
    
    return credentials

def signup_new_user():
    engine = get_engine()
    metadata = MetaData()
    users_table = Table('usersdb', metadata, autoload_with=engine)
    
    with engine.connect() as conn:
        existing_users = [row['key'] for row in conn.execute(select([users_table])).fetchall()]

    st.subheader("Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    new_password = stauth.Hasher([new_password]).generate()[0]

    if st.button("Sign Up"):
        if not new_username or not new_password:
            st.error("Please provide a username and password.")
        else:
            if new_username in existing_users:
                st.error("Username already exists. Please choose a different one.")
            else:
                with engine.connect() as conn:
                    conn.execute(users_table.insert().values(key=new_username, name=new_username, password=new_password))
                st.success("Sign up successful! You can now log in by clicking Login in the sidebar.")


def authenticate_user(credentials):

    authenticator = stauth.Authenticate(credentials, 'daily_app', 'abcd1234', cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login('Login', 'main')

    return name, authentication_status, username, authenticator


#-----------------
# Pull Data In
#-----------------

def get_db_path(filename):
    db_path = Path(__file__).parents[0] / filename
    return db_path.__str__()

def get_conn(filename):
    db_path = get_db_path(filename)
    conn = sqlite3.connect(db_path)
    return conn

def pull_sim_requirements():
    # set league information, included position requirements, number of teams, and salary cap
    pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}

    # create a dictionary that also contains FLEX
    pos_require_flex = copy.deepcopy(pos_require_start)
    del pos_require_flex['DEF']
    pos_require_flex['FLEX'] = 1
    pos_require_flex['DEF'] = 1
    total_pos = np.sum(list(pos_require_flex.values()))
    
    return pos_require_flex, total_pos

@st.cache_data
def pull_op_params(filename, week, year):
    conn = get_conn(filename)
    # pull in the run parameters for the current week and year
    op_params = pd.read_sql_query(f'''SELECT * 
                                      FROM Run_Params
                                      WHERE week={week}
                                            AND year={year}''', conn)
    op_params = {k: v[0] for k,v in op_params.to_dict().items()}

    return op_params, copy.deepcopy(op_params['last_update'])


#------------------
# App Components
#------------------

def headings_text(name):
    titl = st.title('🏈 Fantasy Football Lineup Optimizer')
    subhead = st.subheader(f'Hello {name}! 😎')
    text1 = st.write('Welcome to the Fantasy Football Lineup Optimizer! This app will help you choose the optimal lineup for your DraftKings fantasy football team.')
    text2 = st.write('Follow the steps below to get started.')
    text3 = st.write(':red[**NOTE:**] *We recommend using desktop for the best experience.* 💻')
    return titl, subhead, text1, text2, text3

def side_bar_labels(last_update, week, year):
                    st.header('Simulation Information')
                    st.write('Last Update:', last_update)
                    st.write('Week:', str(week))
                    st.write('Year:', str(year))

def show_times_selected(df, week, year, username):
    engine = get_engine()
    metadata = MetaData()
    results_table = Table('resultsdb', metadata, autoload_with=engine)
    
    with engine.connect() as conn:
        query = select([results_table]).where(
            (results_table.c.week == week) & 
            (results_table.c.year == year) & 
            (results_table.c.user == username)
        )
        results = pd.DataFrame(conn.execute(query).fetchall())
    
    if not results.empty:
        total_lineups = len(results.id.unique())
        results = results.groupby(['player'], as_index=False).agg({'id': 'count'}).rename(columns={'id': 'exposure'})
        results.exposure = 100*(results.exposure / total_lineups).round(2)
        df = pd.merge(df, results, on='player', how='left').fillna(0)
    else:
        df['exposure'] = 0

    return df

def show_etr_ownership(df, conn, week, year):

    etr_own = pd.read_sql_query(f'''SELECT player, opp, etr_proj_large_own as etr_ownership
                                    FROM ETR_Projections_DK
                                    WHERE week={week}
                                          AND year={year}
                                          AND pos != 'DST'
                                    UNION 
                                    SELECT team player, opp, etr_proj_large_own as etr_ownership
                                    FROM ETR_Projections_DK
                                    WHERE week={week}
                                          AND year={year}
                                          AND pos = 'DST'
                                    ''', conn)
    df = pd.merge(df, etr_own, on='player', how='left')
    return df

def get_display_data(player_data, week, year, username):
    display_data = player_data[['player', 'pos', 'salary', 'pred_fp_per_game']].sort_values(by='salary', ascending=False).reset_index(drop=True)
    
    display_data['my_team'] = False
    display_data['exclude'] = False
    display_data.my_team = display_data.my_team.astype(bool)
    display_data.exclude = display_data.exclude.astype(bool)

    display_data = display_data.rename(columns={'pred_fp_per_game': 'pred_pts'})
    display_data.pred_pts = display_data.pred_pts.round(1)

    display_data = show_times_selected(display_data, week, year, username)
    display_data = show_etr_ownership(display_data, get_conn(db_name), week, year)
    display_data = display_data[['player', 'pos', 'opp', 'salary', 'pred_pts', 'my_team', 'exclude', 'etr_ownership', 'exposure']]
    return display_data

def update_interactive_grid(data):
    selected = st.data_editor(
            data,
            column_config={
                "my_team": st.column_config.CheckboxColumn(
                    "my_team",
                    help="Choose players to add to your team",
                    default=False,
                ),
                "exclude": st.column_config.CheckboxColumn(
                    "exclude",
                    help="Choose players to exclude from consideration",
                    default=False,
                )
            },
            use_container_width=True,
            disabled=["widgets"],
            hide_index=True,
            height=500,
        )
    return selected

@st.cache_resource
def init_sim(db_path, op_params, week, year):

    pred_vers = op_params['pred_vers']
    reg_ens_vers = op_params['reg_ens_vers']
    million_ens_vers = op_params['million_ens_vers']

    for k in ['week', 'year', 'pred_vers', 'reg_ens_vers', 'million_ens_vers', 'last_update']:
        op_params.pop(k, None)

    for k, v in op_params.items():
        try: op_params[k] = float(v)
        except: op_params[k] = eval(v)
    
    rs = RunSim(db_path, week, year, pred_vers, reg_ens_vers, million_ens_vers, 1, pull_stats=False)
    params = rs.generate_param_list(op_params)
    sim, p = rs.setup_sim(params[0])
    _, _ = rs.run_single_iter(sim, p, [], [],[])

    return rs, rs.player_data, sim, p


def create_previous_lineups(week, year, username):
    previous_lineups = []
    user_lineups = pull_user_lineups(week, year, username)
    for i in user_lineups.id.unique():
        previous_lineups.append(list(user_lineups[user_lineups.id==i].player.values))
    return previous_lineups

def run_sim(df, rs, sim, params):

    to_add = list(df[df.my_team==True].player.values)
    to_drop = list(df[df.exclude==True].player.values)
    
    if st.session_state["full_lineup"]: params['num_iters'] = 1
    else: params['num_iters'] = 25

    full_lineup, player_cnts = rs.run_single_iter(sim, params, to_add, to_drop, st.session_state['previous_lineups'])

    return full_lineup, player_cnts


def show_results(results):
    return st.data_editor(
                    results,
                    column_config={
                        "SelectionCounts": st.column_config.ProgressColumn(
                            "SelectionCounts",
                            help="Percent of selections in best lineup",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    height=500,
                    use_container_width=True
                )
    

def init_my_team_df(pos_require):

    my_team_list = []
    for k, v in pos_require.items():
        for _ in range(v):
            my_team_list.append([k, None, 0, 0])

    my_team_df = pd.DataFrame(my_team_list, columns=['Position', 'Player', 'Salary', 'Points Added'])
    
    return my_team_df


def team_fill(df, df2):
    '''
    INPUT: df: blank team template to be filled in with chosen players
           df2: chosen players dataframe

    OUTPUT: Dataframe filled in with selected player information
    '''
    # loop through chosen players dataframe
    for _, row in df2.iterrows():

        # pull out the current position and find min position index not filled (if any)
        cur_pos = row.pos
        min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()

        # if position still needs to be filled, fill it
        if min_idx is not np.nan:
            df.loc[min_idx, ['Player', 'Salary']] = [row.player, row.salary]

        # if normal positions filled, fill in the FLEX if applicable
        elif cur_pos in ('RB', 'WR', 'TE'):
            cur_pos = 'FLEX'
            min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()
            if min_idx is not np.nan:
                df.loc[min_idx, ['Player', 'Salary']] = [row.player, row.salary]

    return df[['Position', 'Player', 'Salary']]


#------------------
# Write out Results
#------------------

def format_df_upload(df, username):
    df['id'] = str(np.random.choice(range(1000000), size=1)[0])
    df['created_at'] = str(pd.to_datetime('now'))
    df['user'] = username
    df['week'] = week
    df['year'] = year   

    return df.loc[~df.player.isnull(),['id', 'created_at', 'user', 'week', 'year', 'pos', 'player']]


def upload_results(df):
    engine = get_engine()
    metadata = MetaData()
    results_table = Table('resultsdb', metadata, autoload_with=engine)

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(results_table.insert().values(row.to_dict()))

def create_database_output(my_team, filename, week, year):

    conn = get_conn(filename)
    ids = pd.read_sql_query(f"SELECT * FROM Player_Ids WHERE year={year} AND week={week}", conn)
    my_team_ids = my_team.rename(columns={'Player': 'player'}).copy()
    dk_output = pd.merge(my_team_ids, ids, on='player')

    for pstn, num_req in zip(['WR', 'RB', 'TE'], [4, 3, 2]):
        if len(dk_output[dk_output.pos == pstn]) == num_req:
            idx_last = dk_output[dk_output.pos == pstn].index[-1]
            dk_output.loc[dk_output.index==idx_last, 'pos'] = 'FLEX'

    pos_map = {
        'QB': 'aQB', 
        'RB': 'bRB',
        'WR': 'cWR',
        'TE': 'dTE',
        'FLEX': 'eFLEX',
        'DST': 'fDST'
    }
    dk_output.pos = dk_output.pos.map(pos_map)
    dk_output = dk_output.sort_values(by='pos').reset_index(drop=True)
    pos_map_rev = {v: k for k,v in pos_map.items()}
    dk_output.pos = dk_output.pos.map(pos_map_rev)

    dk_output_ids = dk_output[['pos', 'player_id']].T.reset_index(drop=True)
    dk_output_players = dk_output[['pos', 'player']].T.reset_index(drop=True)
    dk_output = pd.concat([dk_output_players, dk_output_ids], axis=1)

    dk_output.columns = range(dk_output.shape[1])
    dk_output = pd.DataFrame(dk_output.iloc[1,:]).T

    return dk_output


@st.cache_data
def pull_saved_auto_lineups(db_name, week, year, num_auto_lineups):
    conn = get_conn(db_name)
    saved_lineups = pd.read_sql_query(f'''SELECT *
                                            FROM Automated_Lineups
                                            WHERE week={week}
                                                  AND year={year}
                                       ''', conn)
    
    num_auto_lineups = np.min([num_auto_lineups, saved_lineups.shape[0]])
    saved_lineups = saved_lineups.sample(n=num_auto_lineups)
    saved_lineups = saved_lineups.drop(['week', 'year', 'lineup_num', 'trial_num'], axis=1).reset_index(drop=True)
    
    return saved_lineups


def pull_user_lineups(week, year, username):
    engine = get_engine()
    metadata = MetaData()
    results_table = Table('resultsdb', metadata, autoload_with=engine)
    
    with engine.connect() as conn:
        query = select([results_table]).where(
            (results_table.c.week == week) & 
            (results_table.c.year == year) & 
            (results_table.c.user == username)
        )
        results = pd.DataFrame(conn.execute(query).fetchall())
    
    if results.empty:
        results = pd.DataFrame({'id': [], 'created_at': [], 'user': [], 'week': [], 'year': [], 'pos': [], 'player': []})

    return results


def get_num_manual_lineups(week, year, username):
    num_manual_lineups = pull_user_lineups(week, year, username)
    if not num_manual_lineups.empty:
        num_manual_lineups = len(num_manual_lineups.id.unique())
    else:
        num_manual_lineups = 0
    return num_manual_lineups

def download_saved_teams(filename, week, year, username, num_auto_lineups):
    if num_auto_lineups > 0:
        auto_lineups = pull_saved_auto_lineups(filename, week, year, num_auto_lineups)
    else:
        auto_lineups = pd.DataFrame()
    
    try:
        results = pull_user_lineups(week, year, username)
         
        save_result = pd.DataFrame()
        for r in results['id'].unique():
            cur_result = create_database_output(results[results.id==r], filename, week, year)
            save_result = pd.concat([save_result, cur_result], axis=0)

        auto_lineups.columns = save_result.columns
        save_result['lineup_type'] = 'manual'
        auto_lineups['lineup_type'] = 'auto'
        save_result = pd.concat([save_result, auto_lineups], axis=0).reset_index(drop=True)
    
        return save_result.to_csv().encode('utf-8')
    
    except:
        st.write('No saved teams yet!')
        return pd.DataFrame().to_csv().encode('utf-8')
    
#======================
# Run the App
#======================

def main():
    
    st.set_page_config(layout="wide")

    menu = ['SignUp', 'Login']
    with st.sidebar:
        st.header('User Authentication')
        choice = st.radio("Menu", menu, label_visibility='collapsed')
        
    if choice == "SignUp":
        signup_new_user()
    
    elif choice == 'Login':
        
        credentials = pull_user_list()
        name, authentication_status, username, authenticator = authenticate_user(credentials)

        if authentication_status == False: st.error('Username/password is incorrect')
        if authentication_status == None: st.warning('Please enter your username and password')

        if authentication_status:
            
            headings_text(name)
            col1, col2, col3 = st.columns([5, 3, 3])
            
            # get current params + requirements
            op_params, last_update = pull_op_params(db_name, week, year)
            pos_require_flex, total_pos = pull_sim_requirements()
            team_display = init_my_team_df(pos_require_flex) 

            # intialize simulation
            db_path = get_db_path(db_name)
            rs, player_data, sim, cur_params = init_sim(db_path, op_params, week, year)
            
            # Initialize or sync session state for min_pass_catcher
            # On first load OR if user hasn't manually changed it, use DB value
            db_min_pass_catcher = cur_params.get('min_pass_catchers', 0)
            
            if 'min_pass_catchers' not in st.session_state:
                # First time - initialize from DB
                st.session_state['min_pass_catchers'] = db_min_pass_catcher
                st.session_state['min_pass_catcher_db_value'] = db_min_pass_catcher
            elif 'min_pass_catcher_db_value' not in st.session_state:
                # Upgrading from old version - save both
                st.session_state['min_pass_catcher_db_value'] = db_min_pass_catcher
            elif st.session_state.get('min_pass_catcher_db_value') != db_min_pass_catcher:
                # DB value changed (e.g., after refresh) - update session state
                st.session_state['min_pass_catchers'] = db_min_pass_catcher
                st.session_state['min_pass_catcher_db_value'] = db_min_pass_catcher
            
            # get the player selection data and set to session state for auto select
            display_data = get_display_data(player_data, week, year, username)
            if "dd" not in st.session_state: st.session_state["dd"] = display_data
            if 'my_team' not in st.session_state: st.session_state['my_team'] = None
            if 'save_click' not in st.session_state: st.session_state['save_click'] = False
            if 'run_sim' not in st.session_state: st.session_state['run_sim'] = False

            with st.sidebar:
                authenticator.logout('Logout', 'main')
                        
                st.header('Reset Current Selections')
                if st.button("Refresh Data"):
                    # Clear all caches to force re-pulling params and re-initializing sim
                    pull_op_params.clear()
                    init_sim.clear()
                    
                    # Re-pull params and re-init simulation (will get new randomized values)
                    op_params_new, _ = pull_op_params(db_name, week, year)
                    rs_new, player_data_new, sim_new, cur_params_new = init_sim(db_path, op_params_new, week, year)
                    
                    # Update min_pass_catcher from the newly loaded params
                    new_min_pass_catcher = cur_params_new.get('min_pass_catchers', 0)
                    st.session_state['min_pass_catchers'] = new_min_pass_catcher
                    st.session_state['min_pass_catcher_db_value'] = new_min_pass_catcher
                    
                    # Reset other session state
                    st.session_state["dd"] = get_display_data(player_data_new, week, year, username)
                    st.session_state['previous_lineups'] = create_previous_lineups(week, year, username)
                    st.session_state['my_team'] = None
                    st.session_state['save_click'] = False
                    st.session_state['run_sim'] = False
                    
                    st.experimental_rerun()


                st.header("Full Lineup or Player-by-Player")
                lineup_type = st.radio(
                        label='Choose how to fill in your lineup',
                        options=["Full Lineup", "Player-by-Player"]
                    )
                if lineup_type == "Full Lineup": st.session_state["full_lineup"] = True
                else: st.session_state["full_lineup"] = False
                
                st.header("QB-Stack Partners")
                st.session_state['min_pass_catchers'] = st.number_input(
                    'Number of QB-Stack Partners',
                    min_value=0,
                    max_value=5,
                    value=int(st.session_state['min_pass_catchers']),
                    step=1,
                    help='Minimum number of pass catchers to stack with QB'
                )
                
                # Display all simulation parameters
                with st.expander("📊 All Simulation Parameters", expanded=False):
                    st.write("**Current Parameters:**")
                    for key, value in sorted(cur_params.items()):
                        st.write(f"- {key}: {value}")
                
                side_bar_labels(last_update, week, year)

            with col3:
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.header("⚡:green[Your Team]⚡")  
                    st.write('*Players selected so far 🏈*')
                with subcol2:
                    st.header('')
                    if st.button("Save Team", type='primary', use_container_width=True):
                        my_team_upload = format_df_upload(st.session_state['my_team'], username)
                        st.session_state['save_click'] = True
                        upload_results(my_team_upload) 
                        st.text('Team Saved! Refresh Data.')
            
            with col1:
                st.header('1. Choose Players')
                st.write('*Check **my_team** box to select a player* ✅')
                selected = update_interactive_grid(st.session_state["dd"])
            
            with st.sidebar:
            
                st.header('CSV for Draftkings')
                
                num_manual_lineups = get_num_manual_lineups(week, year, username)
                st.write('Number Manual Lineups:', num_manual_lineups)
                num_auto_lineups = st.number_input('Number Auto Lineups', min_value=0, max_value=100, value=20, step=1)
                st.download_button(
                        "Download Saved Teams",
                        download_saved_teams(db_name, week, year, username, num_auto_lineups),
                        f"week{week}_year{year}_saved_teams.csv",
                        "text/csv",
                        key='download-csv'
                )

            if 'previous_lineups' not in st.session_state:
                st.session_state['previous_lineups'] = create_previous_lineups(week, year, username)
            
            # Update cur_params with user-modified min_pass_catcher value
            cur_params = copy.deepcopy(cur_params)
            cur_params['min_pass_catchers'] = st.session_state['min_pass_catchers']
            
            # Only run simulation when button is clicked
            if st.session_state['run_sim'] and not st.session_state['save_click']:
                full_lineup, results = run_sim(selected, rs, sim, cur_params)
                
                if st.session_state["full_lineup"]:
                    st.session_state['my_team'] = selected[selected.player.isin(full_lineup)].copy()
                    selected.loc[selected.player.isin(full_lineup), 'my_team'] = True
                
                else: 
                    st.session_state['my_team'] = selected.loc[selected.my_team==True]
                    rm_players = st.session_state['my_team'].player.unique()
                    results = results[~results.player.isin(rm_players)].reset_index(drop=True)
                
                # Reset run_sim flag after simulation completes
                st.session_state['run_sim'] = False
            
            elif st.session_state['save_click']:
                results = pd.DataFrame()
            
            else:
                # Before first simulation run, show empty results
                results = pd.DataFrame()
                if st.session_state['my_team'] is None:
                    st.session_state['my_team'] = selected.loc[selected.my_team==True]

            with col2: 
                st.header('2. Review Top Choices')
                st.write('*These are the optimal players to choose from* ⬇️')
                
                # Add Run Simulation button
                if st.button("🚀 Run Simulation", type='primary', use_container_width=True):
                    st.session_state['run_sim'] = True
                    st.experimental_rerun()
                
                # Display results or placeholder
                if results.empty and not st.session_state['save_click']:
                    st.info("Click 'Run Simulation' to see optimal player choices")
                else:
                    show_results(results)

            with col3:      
                
                try: st.table(team_fill(team_display, st.session_state['my_team']))
                except: st.table(team_display)

                subcol1, subcol2, subcol3 = st.columns(3)
                remaining_salary = 50000-st.session_state['my_team'].salary.sum()
                subcol2.metric('Remaining Salary', remaining_salary)
                
                if total_pos-len(st.session_state['my_team']) > 0: subcol2.metric('Per Player', int(remaining_salary / (total_pos-len(st.session_state['my_team']))))
                else: subcol3.metric('Per Player', 'N/A')

if __name__ == '__main__':
    main()
# %%

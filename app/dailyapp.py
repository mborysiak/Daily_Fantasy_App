#%%
import streamlit as st
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
import sqlite3
from zSim_Helper_Covar import FootballSimulation
import streamlit_authenticator as stauth
from deta import Deta

year = 2022
week = 2
num_iters = 150

total_lineups = 5
db_name = 'Simulation_App.sqlite3'
deta_key = st.secrets['deta_key']


#--------------------
# Deta operations
#-------------------

def deta_connect(deta_key):
    return Deta(deta_key)

def pull_user_list(deta_key):

    deta = deta_connect(deta_key)
    db_users = deta.Base('usersdb')
    
    users = db_users.fetch().items
    credentials = {'usernames': {}}
    for user in users:
        credentials['usernames'][user['key']] = {
                                                    'email': None, 
                                                    'name': user['name'], 
                                                    'password': user['password']
                                            }
   
    return credentials

def signup_new_user(deta_key):
    
    deta = deta_connect(deta_key)
    db_users = deta.Base('usersdb')
    existing_users = [ex_user['key'] for ex_user in db_users.fetch().items]

    st.subheader("Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    new_password = stauth.Hasher([new_password]).generate()[0]

    if st.button("Sign Up"):
        if not new_username or not new_password:
            st.error("Please provide a username and password.")
        else:
            # Check if the username already exists
            if new_username in existing_users:
                st.error("Username already exists. Please choose a different one.")
            else:
                # Save the new user in the database
                db_users.put({'key': new_username, 'name': new_username, 'password': new_password})
                st.success("Sign up successful! You can now log in by clicking Login in the sidebar.")


def authenticate_user(credentials):

    authenticator = stauth.Authenticate(credentials, 'daily_app', 'abcd1234', cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login('Login', 'main')

    return name, authentication_status, username, authenticator


#-----------------
# Pull Data In
#-----------------

def get_conn(filename):
    from pathlib import Path
    filepath = Path(__file__).parents[0] / filename
    conn = sqlite3.connect(filepath)
    
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
    
    return pos_require_start, pos_require_flex, total_pos

@st.cache_data
def pull_op_params(filename, week, year):
    conn = get_conn(filename)
    # pull in the run parameters for the current week and year
    op_params = pd.read_sql_query(f'''SELECT * 
                                      FROM Run_Params
                                      WHERE week={week}
                                            AND year={year}''', conn)
    op_params = {k: v[0] for k,v in op_params.to_dict().items()}
    return op_params

@st.cache_resource
class PullData:

    def __init__(self, week, year, filename, op_params):
        self.week = week
        self.year = year
        self.filename = filename

        self.pred_vers = op_params['pred_vers']
        self.ensemble_vers = op_params['ensemble_vers']
        self.std_dev_type = op_params['std_dev_type']
        self.full_model_weight = eval(op_params['full_model_weight'])
        self.covar_type = eval(op_params['covar_type'])

        if self.covar_type == 'no_covar': self.use_covar = False
        else: self.use_covar = True

        self.ownership = self.pull_ownership()

        st.write(self.full_model_weight, self.covar_type)

    def pull_ownership(self):
        conn = get_conn(self.filename)
        ownership = pd.read_sql_query(f'''SELECT player Player, pred_ownership Ownership
                                        FROM Predicted_Ownership
                                        WHERE week={self.week} 
                                                AND year={self.year}''', conn)
        ownership.Ownership = ownership.Ownership.apply(lambda x: np.round(100*np.exp(x),1))
        
        return ownership

    def get_drop_teams(self):

        conn = get_conn(self.filename)   
        df = pd.read_sql_query(f'''SELECT away_team, home_team, gametime 
                                FROM Gambling_Lines 
                                WHERE week={self.week} 
                                        AND year={self.year} 
                    ''', conn)
        df.gametime = pd.to_datetime(df.gametime)
        df['day_of_week'] = df.gametime.apply(lambda x: x.weekday())
        df['hour_in_day'] = df.gametime.apply(lambda x: x.hour)
        df = df[(df.day_of_week!=6) | (df.hour_in_day > 16) | (df.hour_in_day < 11)]
        drop_teams = list(df.away_team.values)
        drop_teams.extend(list(df.home_team.values))

        return drop_teams


    def pull_model_predictions(self):
            conn = get_conn(self.filename)
            df = pd.read_sql_query(f'''SELECT * 
                                        FROM Model_Predictions
                                        WHERE week={self.week}
                                            AND year={self.year}
                                            AND version='{self.pred_vers}'
                                            AND ensemble_vers='{self.ensemble_vers}'
                                            AND std_dev_type='{self.std_dev_type}'
                                            AND pos !='K'
                                            AND pos IS NOT NULL
                                            AND player!='Ryan Griffin'
                                                ''', conn)
            df['weighting'] = 1
            df.loc[df.model_type=='full_model', 'weighting'] = self.full_model_weight

            score_cols = ['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']
            for c in score_cols: 
                df[c] = df[c] * df.weighting

            df = df.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                                    'std_dev': 'sum',
                                                                    'weighting': 'sum',
                                                                    'min_score': 'sum',
                                                                    'max_score': 'sum'})
            for c in score_cols: df[c] = df[c] / df.weighting
            df.loc[df.pos=='Defense', 'pos'] = 'DEF'

            teams = pd.read_sql_query(f'''SELECT player, team 
                                        FROM Player_Teams 
                                        WHERE week={self.week} 
                                                AND year={self.year}''', conn)
            df = pd.merge(df, teams, on=['player'])

            drop_teams = self.get_drop_teams()
            df = df[~df.team.isin(drop_teams)].reset_index(drop=True)

            return df.drop('weighting', axis=1)

    def pull_covar(self):
        conn = get_conn(self.filename)
        player_data = pd.read_sql_query(f'''SELECT * 
                                            FROM Covar_Means
                                            WHERE week={self.week}
                                                    AND year={self.year}
                                                    AND pred_vers='{self.pred_vers}'
                                                    AND ensemble_vers='{self.ensemble_vers}'
                                                    AND std_dev_type='{self.std_dev_type}'
                                                    AND covar_type='{self.covar_type}' 
                                                    AND full_model_rel_weight={self.full_model_weight}''', 
                                                conn)
        
        covar = pd.read_sql_query(f'''SELECT player, player_two, covar
                                      FROM Covar_Matrix
                                      WHERE week={self.week}
                                            AND year={self.year}
                                            AND pred_vers='{self.pred_vers}'
                                            AND ensemble_vers='{self.ensemble_vers}'
                                            AND std_dev_type='{self.std_dev_type}'
                                            AND covar_type='{self.covar_type}'
                                            AND full_model_rel_weight={self.full_model_weight}''', 
                                            conn)
        
        covar = pd.pivot_table(covar, index='player', columns='player_two').reset_index().fillna(0)
        covar.columns = [c[1] if i!=0 else 'player' for i, c in enumerate(covar.columns)]

        return player_data, covar
    
    def join_salary(self, df):
        conn = get_conn(self.filename)
        # add salaries to the dataframe and set index to player
        salaries = pd.read_sql_query(f'''SELECT player, salary
                                         FROM Salaries
                                         WHERE year={self.year}
                                               AND week={self.week} ''', conn)

        df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
        df.salary = df.salary.fillna(10000)

        return df

    def pull_player_data(self):

        if self.use_covar: 
            player_data, covar = self.pull_covar()
            min_max = self.pull_model_predictions()[['player', 'min_score', 'max_score']]
        else: 
            player_data, covar, min_max = self.pull_model_predictions(), None, None

        player_data = self.join_salary(player_data)

        return player_data, covar, min_max


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

def get_display_data(player_data):
    
    display_data = player_data[['player', 'pos', 'salary', 'pred_fp_per_game']].sort_values(by='salary', ascending=False).reset_index(drop=True)
    
    display_data['my_team'] = False
    display_data['exclude'] = False
    display_data.my_team = display_data.my_team.astype(bool)
    display_data.exclude = display_data.exclude.astype(bool)

    display_data = display_data.rename(columns={'pred_fp_per_game': 'pred_pts'})
    display_data.pred_pts = display_data.pred_pts.round(1)
    
    return display_data

def create_interactive_grid(data):
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
            height=500
        )
    return selected

def create_plot(df):
    # Create a plot
    ax = df[['player', 'dk_salary']].set_index('player').plot(kind='barh')

    # Display the plot using Streamlit
    return st.write(ax.get_figure())

def download_saved_teams(deta_key, week, year, username):
    
    deta = deta_connect(deta_key)
    db_results = deta.Base('resultsdb')

    try:
        results = pd.DataFrame(db_results.fetch({'week': week, 'year': year, 'user': username}).items)    
        return results.to_csv().encode('utf-8')
    
    except: 
        st.write('No saved teams yet!')
        return pd.DataFrame().to_csv().encode('utf-8')
    

@st.cache_data
def init_sim(player_data, covar, min_max, use_covar, op_params, pos_require_start):
    num_iters = eval(op_params['num_iters'])
    use_ownership = eval(op_params['use_ownership'])
    salary_remain_max = eval(op_params['max_salary_remain'])

    sim = FootballSimulation(player_data, covar, min_max, week, year, salary_cap=50000, 
                             pos_require_start=pos_require_start, num_iters=num_iters, 
                             matchup_seed=False, use_covar=use_covar, use_ownership=use_ownership, 
                             salary_remain_max=salary_remain_max, db_name=db_name)
    return sim

@st.cache_data
def extract_params(op_params):

    ownership_vers = op_params['ownership_vers']
    adjust_pos_counts = eval(op_params['adjust_pos_counts'])
    matchup_drop = eval(op_params['matchup_drop'])
    max_team_type = eval(op_params['max_team_type'])
    min_player_same_team = eval(op_params['min_player_same_team'])
    min_player_opp_team = eval(op_params['min_players_opp_team'])
    num_top_players = eval(op_params['num_top_players'])
    own_neg_frac = eval(op_params['own_neg_frac'])
    player_drop_multiple = eval(op_params['player_drop_multiple'])
    qb_min_iter = eval(op_params['qb_min_iter'])
    qb_set_max_team = eval(op_params['qb_set_max_team'])
    qb_solo_start = eval(op_params['qb_solo_start'])
    static_top_players = eval(op_params['static_top_players'])

    try: min_player_opp_team = int(min_player_opp_team)
    except: pass
    try:min_player_same_team = int(min_player_same_team)
    except: pass

    return ownership_vers, adjust_pos_counts, matchup_drop, max_team_type, min_player_same_team, min_player_opp_team, num_top_players, own_neg_frac, player_drop_multiple, qb_min_iter, qb_set_max_team, qb_solo_start, static_top_players


def run_sim(df, sim, op_params, stack_team):
    to_add = list(df[df.my_team==True].player.values)
    to_drop = list(df[df.exclude==True].player.values)

    ownership_vers, adjust_pos_counts, matchup_drop, max_team_type, \
    min_player_same_team, min_player_opp_team, num_top_players, own_neg_frac,  \
    player_drop_multiple, qb_min_iter, qb_set_max_team, qb_solo_start, static_top_players = extract_params(op_params)
    
    if stack_team == 'Auto': 
        set_max_team = None
    else: 
        set_max_team = stack_team
        qb_set_max_team = False
        matchup_drop = 0
        qb_min_iter = 9

    st.write(set_max_team, adjust_pos_counts, matchup_drop, min_player_opp_team, min_player_same_team)

    results, team_cnts = sim.run_sim(to_add, to_drop, min_players_same_team_input=min_player_same_team, set_max_team=set_max_team, 
                                    min_players_opp_team_input=min_player_opp_team, adjust_select=adjust_pos_counts, 
                                    max_team_type=max_team_type, num_matchup_drop=matchup_drop, own_neg_frac=own_neg_frac, 
                                    n_top_players=num_top_players, ownership_vers=ownership_vers, static_top_players=static_top_players, 
                                    qb_min_iter=qb_min_iter, qb_set_max_team=qb_set_max_team, qb_solo_start=qb_solo_start)
    return results, team_cnts


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

def upload_results(deta_key, df):
    
    deta = deta_connect(deta_key)
    db_results = deta.Base('resultsdb')

    for _, row in df.iterrows():
        cur_row = {}
        for c in df.columns:
            cur_row[c] = row[c]
        db_results.put(cur_row)

    st.write('Lineup Saved')

def main():
    
    st.set_page_config(layout="wide")

    menu = ['SignUp', 'Login']
    with st.sidebar:
        st.header('User Authentication')
        choice = st.radio("Menu", menu, label_visibility='collapsed')
        
    if choice == "SignUp":
        signup_new_user(deta_key)
    
    elif choice == 'Login':
        
        credentials = pull_user_list(deta_key)
        name, authentication_status, username, authenticator = authenticate_user(credentials)

        if authentication_status == False:
            st.error('Username/password is incorrect')
        if authentication_status == None:
            st.warning('Please enter your username and password')

        if authentication_status:

            with st.sidebar:
                authenticator.logout('Logout', 'main')
                        
                st.header('Simulation Parameters')
                if st.button("Refresh Data"):
                    PullData.clear()
                    extract_params.clear()
                    init_sim.clear()
                    pull_op_params.clear()

                st.write('Week:', week)
                st.write('Year:', year)
            
            headings_text(name)
            col1, col2, col3 = st.columns([4, 3, 3])
            
            op_params = pull_op_params(db_name, week, year)
            pos_require_start, pos_require_flex, total_pos = pull_sim_requirements()
            
            team_display = init_my_team_df(pos_require_flex) 
            data_class = PullData(week, year, db_name, op_params)
            player_data, covar, min_max = data_class.pull_player_data()
            
            with st.sidebar:
                stack_team = st.selectbox('Stack Team', ['Auto']+sorted(list(player_data.team.unique())))

                st.download_button(
                        "Download Saved Teams",
                        download_saved_teams(deta_key, week, year, username),
                        f"week{week}_year{year}_saved_teams.csv",
                        "text/csv",
                        key='download-csv'
                    )
            
            display_data = get_display_data(player_data)
            sim = init_sim(player_data, covar, min_max, data_class.use_covar, op_params, pos_require_start)

            with col1:
                st.header('1. Choose Players')
                st.write('*Check **my_team** box to select a player* ✅')
                selected = create_interactive_grid(display_data)
                my_team = selected.loc[selected.my_team==True]

            results, team_cnts = run_sim(selected, sim, op_params, stack_team)
            results = results[results.SelectionCounts<100]
            
            with col2: 
                st.header('2. Review Top Choices')
                st.write('*These are the optimal players to choose from* ⬇️')

                st.dataframe(results, use_container_width=True, height=500)

            with col3:      
                st.header("⚡:green[Your Team]⚡")  
                st.write('*Players selected so far 🏈*')
                
                try: st.table(team_fill(team_display, my_team))
                except: st.table(team_display)

                subcol1, subcol2, subcol3 = st.columns(3)
                remaining_salary = 50000-my_team.salary.sum()
                subcol1.metric('Remaining Salary', remaining_salary)
                
                if total_pos-len(my_team) > 0: subcol2.metric('Per Player', int(remaining_salary / (total_pos-len(my_team))))
                else: subcol2.metric('Per Player', 'N/A')
                
                with subcol3:
                    if st.button("Save Team"):
                        my_team_upload = format_df_upload(my_team, username)
                        upload_results(deta_key, my_team_upload) 
                        st.text('Team Saved!')

if __name__ == '__main__':
    main()
# %%


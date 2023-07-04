#%%
import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
import sqlite3
from zSim_Helper_Covar import FootballSimulation

year = 2022
week = 2
num_iters = 150

total_lineups = 5
db_name = 'Simulation_App.sqlite3'

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

# def create_interactive_grid(data):
#     gb = GridOptionsBuilder.from_dataframe(data)
#     # add pagination
#     # gb.configure_pagination(paginationAutoPageSize=True)
#     gb.configure_side_bar()
#     gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
#     gb.configure_column('Include Player', editable=True, cellEditor='agSelectCellEditor', cellEditorParams={'values': ['Yes', 'No'] })
#     gridOptions = gb.build()

#     grid_response = AgGrid(
#         data,
#         gridOptions=gridOptions,
#         data_return_mode='AS_INPUT', 
#         update_mode='SELECTION_CHANGED', 
#         columns_auto_size=ColumnsAutoSizeMode.FIT_CONTENTS,
#         fit_columns_on_grid_load=True,
#         height=500, 
#         width='100%',
#         reload_data=False
#     )

#     data = grid_response['data']
#     selected = grid_response['selected_rows'] 
#     df = pd.DataFrame(selected) 

#     return df

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

@st.cache_data
def download_saved_teams():
    # conn = get_conn(db_name)
    # df = pd.read_sql_query('SELECT * FROM My_Team', conn)
    
    return st.session_state['lineups'].to_csv(index=False).encode('utf-8')
    # except: 
    #     st.write('No saved teams yet!')
    #     return pd.DataFrame().to_csv().encode('utf-8')
    

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


def run_sim(df, sim, op_params):
    to_add = list(df[df.my_team==True].player.values)
    to_drop = list(df[df.exclude==True].player.values)
    min_player_same_team_input = 2
    set_max_team = None
    results, team_cnts = sim.run_sim(to_add, to_drop, min_player_same_team_input, set_max_team,
                                        ownership_vers='mil_only')
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


#--------------------------


def main():
    # Set page configuration
    st.set_page_config(layout="wide")
    if 'lineups' not in st.session_state:
        st.session_state['lineups'] = pd.DataFrame()

    st.title('🏈 Fantasy Football Lineup Optimizer')
    st.subheader('Hello Spider! 😎')
    st.write('Welcome to the Fantasy Football Lineup Optimizer! This app will help you choose the optimal lineup for your DraftKings fantasy football team.')
    st.write('Follow the steps below to get started.')
    st.write(':red[**NOTE:**] *We recommend using desktop for the best experience.* 💻')

    col1, col2, col3 = st.columns([4, 3, 3])
    op_params = pull_op_params(db_name, week, year)
    pos_require_start, pos_require_flex, total_pos = pull_sim_requirements()
    team_display = init_my_team_df(pos_require_flex) 

    with st.sidebar:

        st.header('Simulation Parameters')
        st.write('Week:', week)
        st.write('Year:', year)


        if st.button("Refresh Data"):
            st.cache_resource.clear()
            op_params['New Data'] = True

        st.download_button(
                "Download Saved Teams",
                download_saved_teams(),
                f"week{week}_year{year}_saved_teams.csv",
                "text/csv",
                key='download-csv'
            )

    data_class = PullData(week, year, db_name, op_params)
    player_data, covar, min_max = data_class.pull_player_data()
    display_data = get_display_data(player_data)

    # st.write(data_class.pred_vers, data_class.ensemble_vers, data_class.std_dev_type)
    # st.write(data_class.covar_type, data_class.full_model_weight)
    # st.write(sim.num_iters, sim.use_ownership, sim.salary_remain_max)
    
    sim = init_sim(player_data, covar, min_max, data_class.use_covar, op_params, pos_require_start)

    with col1:
        st.header('1. Choose Players')
        st.write('*Check **my_team** box to select a player* ✅')

        selected = create_interactive_grid(display_data)
        my_team = selected.loc[selected.my_team==True]

    results, team_cnts = run_sim(selected, sim, op_params)
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
                st.session_state['lineups'] = pd.concat([st.session_state['lineups'], my_team], axis=0)
                #  my_team.to_sql('My_Team', if_exists='append', con=get_conn(db_name), index=False)
        
        st.write(st.session_state['lineups'])
        

if __name__ == '__main__':
    main()
# %%


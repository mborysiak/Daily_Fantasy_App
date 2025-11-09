#%%
import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt.glpk import ilp
import sqlite3
import gc

class FullLineupSim:
    def __init__(self, week, year, conn, pred_vers, reg_ens_vers, million_ens_vers,
                 use_covar, covar_type, print_results):

        self.week = week
        self.set_year = year
        self.pred_vers = pred_vers
        self.reg_ens_vers = reg_ens_vers
        self.million_ens_vers = million_ens_vers
        self.conn = conn
        self.use_covar = use_covar
        self.salary_cap = 50000
        self.covar_type = covar_type
        self.print_results = print_results
        self.num_pos = 9

    def get_model_predictions(self):

        if self.opt_metric == 'points':
            table_name = 'Points_Predictions'
            metric_name = 'pred_fp_per_game'
            metric_ens_vers = self.reg_ens_vers
            std_dev_query = f"AND std_dev_type='{self.std_dev_type}'"
        if self.opt_metric == 'etr_points':
            table_name = 'ETR_Predictions'
            metric_name = 'pred_fp_per_game'
            metric_ens_vers = self.reg_ens_vers
            std_dev_query = f"AND std_dev_type='{self.std_dev_type}'"
        elif self.opt_metric == 'roi':
            table_name = 'ROI_Predictions'
            metric_name = 'pred_roi'
            metric_ens_vers = self.million_ens_vers
            std_dev_query = ''
        elif self.opt_metric == 'roitop':
            table_name = 'ROI_Top_Predictions'
            metric_name = 'pred_roitop'
            metric_ens_vers = self.million_ens_vers
            std_dev_query = ''
        elif self.opt_metric == 'million':
            table_name = 'Million_Predictions'
            metric_name = 'pred_million'
            metric_ens_vers = self.million_ens_vers
            std_dev_query = ''

        df = pd.read_sql_query(f'''SELECT player, 
                                          pos, 
                                          {metric_name} pred_fp_per_game, 
                                          std_dev, 
                                          min_score, 
                                          max_score
                                    FROM {table_name}
                                    WHERE week={self.week}
                                        AND year={self.set_year}
                                        AND pred_vers='{self.pred_vers}'
                                        AND ens_vers='{metric_ens_vers}'
                                        {std_dev_query}
                                        AND pos !='K'
                                        AND pos IS NOT NULL
                                        AND player!='Ryan Griffin'
                    
                                ''', self.conn)

        df.loc[df.pos=='Defense', 'pos'] = 'DEF'
        teams = pd.read_sql_query(f'''SELECT player, team 
                                      FROM Player_Teams 
                                      WHERE week={self.week} 
                                            AND year={self.set_year}
                                    ''', 
                                    self.conn)
        df = pd.merge(df, teams, on=['player'])

        drop_teams = self.get_drop_teams()
        df = df[~df.team.isin(drop_teams)].reset_index(drop=True)

        for c in ['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']:
            min_value = df.loc[df[c] > 0, c].min()
            df.loc[df[c].isnull() | (df[c] <= 0), c] = min_value

        return df
    
    
            
    def get_covar_means(self):
        # pull in the player data (means, team, position) and covariance matrix
        player_data = pd.read_sql_query(f'''SELECT * 
                                            FROM Covar_Means
                                            WHERE week={self.week}
                                                    AND year={self.set_year}
                                                    AND pred_vers='{self.pred_vers}'
                                                    AND reg_ens_vers='{self.reg_ens_vers}'
                                                    AND std_dev_type='{self.std_dev_type}'
                                                    AND covar_type='{self.covar_type}' 
                                                    ''', 
                                             self.conn)
        return player_data

    def pull_covar(self):
        covar = pd.read_sql_query(f'''SELECT player, player_two, covar
                                      FROM Covar_Matrix
                                      WHERE week={self.week}
                                            AND year={self.set_year}
                                            AND pred_vers='{self.pred_vers}'
                                            AND reg_ens_vers='{self.reg_ens_vers}'
                                            AND std_dev_type='{self.std_dev_type}'
                                            AND covar_type='{self.covar_type}'
                                            ''', 
                                       self.conn)
        covar = pd.pivot_table(covar, index='player', columns='player_two').reset_index().fillna(0)
        covar.columns = [c[1] if i!=0 else 'player' for i, c in enumerate(covar.columns)]
        return covar
    

    def join_salary(self, df):

        # add salaries to the dataframe and set index to player
        salaries = pd.read_sql_query(f'''SELECT player, salary
                                         FROM Salaries
                                         WHERE year={self.set_year}
                                               AND week={self.week} ''', 
                                        self.conn)

        df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
        df.salary = df.salary.fillna(10000)

        return df
    
    def join_opp(self, df):
        matchups = pd.DataFrame(self.get_matchups(), index=[0]).T.reset_index()
        matchups.columns = ['team', 'opp']
        df = pd.merge(df, matchups, on='team')
        return df

    def get_drop_teams(self):

        df = pd.read_sql_query(f'''SELECT away_team, home_team, gametime 
                                   FROM Gambling_Lines 
                                   WHERE week={self.week} 
                                         AND year={self.set_year} 
                    ''', self.conn)
        df.gametime = pd.to_datetime(df.gametime)
        df['day_of_week'] = df.gametime.apply(lambda x: x.weekday())
        df['hour_in_day'] = df.gametime.apply(lambda x: x.hour)
        df = df[(df.day_of_week!=6) | (df.hour_in_day > 16) | (df.hour_in_day < 11)]
        drop_teams = list(df.away_team.values)
        drop_teams.extend(list(df.home_team.values))

        return drop_teams
    
    def get_matchups(self):
        df = pd.read_sql_query(f'''SELECT away_team, home_team
                                   FROM Gambling_Lines 
                                   WHERE week={self.week} 
                                         and year={self.set_year} 
                    ''', self.conn)

        df = df[(df.away_team.isin(self.player_data.team.unique())) & \
                (df.home_team.isin(self.player_data.team.unique()))]

        matchups = {}
        for away, home in df.values:
            matchups[away] = home
            matchups[home] = away

        return matchups
    
    def join_ownership_pred(self, df):

        # add salaries to the dataframe and set index to player
        ownership = pd.read_sql_query(f'''SELECT player, pred_ownership, std_dev, min_score, max_score
                                          FROM Predicted_Ownership
                                          WHERE year={self.set_year}
                                                AND week={self.week}
                                                AND ownership_vers='{self.ownership_vers}'
                                                AND pred_vers='{self.pred_vers}'
                                                AND million_ens_vers='{self.million_ens_vers}' ''', 
                                        self.conn)

        if self.use_covar: df = df.drop(['pred_fp_per_game'], axis=1)
        else: df = df.drop(['pred_fp_per_game', 'std_dev', 'min_score','max_score'], axis=1)

        df = pd.merge(df, ownership, how='left', on='player')

        for c in ['pred_ownership', 'std_dev', 'min_score', 'max_score']:
            if self.ownership_vers != 'standard_ln':
                min_value = df.loc[df[c] > 0, c].min()
                df.loc[df[c].isnull() | (df[c] <= 0), c] = min_value
            
            else:
                df[c] = df[c].fillna(df[c].min())

        return df

    def join_standard_ownership_pred(self, df):
        # Load standard_ln ownership data specifically for low ownership constraint
        ownership = pd.read_sql_query(f'''SELECT player, pred_ownership, std_dev, min_score, max_score
                                          FROM Predicted_Ownership
                                          WHERE year={self.set_year}
                                                AND week={self.week}
                                                AND ownership_vers='standard_ln'
                                                AND pred_vers='{self.pred_vers}'
                                                AND million_ens_vers='{self.million_ens_vers}' ''', 
                                        self.conn)

        if self.use_covar: df = df.drop(['pred_fp_per_game'], axis=1)
        else: df = df.drop(['pred_fp_per_game', 'std_dev', 'min_score','max_score'], axis=1)

        df = pd.merge(df, ownership, how='left', on='player')

        # For standard_ln, fill missing values with minimum
        for c in ['pred_ownership', 'std_dev', 'min_score', 'max_score']:
            df[c] = df[c].fillna(df[c].min())

        return df

    
    def get_predictions(self, col, ownership=False, num_options=500, use_standard=False):

        labels = self.player_data[['player', 'pos', 'team', 'salary']]

        if self.use_covar and not ownership: 
            predictions = self.covar_dist(num_options)
            predictions = pd.concat([labels, predictions], axis=1)

        else: 
            import time
            start = time.time()
            predictions = self.trunc_normal_dist(col, num_options, use_standard=use_standard)
            predictions = pd.concat([labels, predictions], axis=1)

        return predictions
    
    @staticmethod
    def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=500):

        import scipy.stats as stats

        # create truncated distribution
        lower_bound = (min_sc - mean_val) / sdev, 
        upper_bound = (max_sc - mean_val) / sdev
        trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
        
        estimates = trunc_dist.rvs(num_samples)

        return estimates


    def trunc_normal_dist(self, col, num_options=500, use_standard=False):
        import scipy.stats as stats
        
        if col=='pred_ownership' and use_standard: 
            df = self.standard_ownership_data
        elif col=='pred_ownership': 
            df = self.ownership_data
        elif col=='pred_fp_per_game': 
            df = self.player_data

        # Vectorized approach: extract all parameters as arrays
        means = df[col].values
        sdevs = df['std_dev'].values
        min_scores = df['min_score'].values
        max_scores = df['max_score'].values
        
        # Vectorized bounds calculation
        lower_bounds = (min_scores - means) / sdevs
        upper_bounds = (max_scores - means) / sdevs
        
        # Generate all samples in one vectorized call
        # Use broadcasting to create shape (num_players, num_options)
        all_samples = stats.truncnorm.rvs(
            lower_bounds[:, np.newaxis],  # Shape: (num_players, 1)
            upper_bounds[:, np.newaxis],  # Shape: (num_players, 1)
            loc=means[:, np.newaxis],     # Shape: (num_players, 1)
            scale=sdevs[:, np.newaxis],   # Shape: (num_players, 1)
            size=(len(means), num_options)
        )

        return pd.DataFrame(all_samples)
    
    
    @staticmethod
    def bounded_multivariate_dist(means, cov_matrix, dimenions_bounds, size):

        from numpy.random import multivariate_normal

        ndims = means.shape[0]
        return_samples = np.empty([0,ndims])
        local_size = size

        # generate new samples while the needed size is not reached
        while not return_samples.shape[0] == size:
            samples = multivariate_normal(means, cov_matrix, size=size*10)

            for dim, bounds in enumerate(dimenions_bounds):
                if not np.isnan(bounds[0]): # bounds[0] is the lower bound
                    samples = samples[(samples[:,dim] > bounds[0])]  # samples[:,dim] is the column of the dim

                if not np.isnan(bounds[1]): # bounds[1] is the upper bound
                    samples = samples[(samples[:,dim] < bounds[1])]   # samples[:,dim] is the column of the dim

            return_samples = np.vstack([return_samples, samples])

            local_size = size - return_samples.shape[0]
            if local_size < 0:
                return_samples = return_samples[np.random.choice(return_samples.shape[0], size, replace=False), :]

        return return_samples
    
    @staticmethod
    def unique_matchup_pairs(matchups):
        import itertools
        sorted_matchups = [sorted(m) for m in matchups.items()]
        return list(m for m, _ in itertools.groupby(sorted_matchups))
    
    
    def covar_dist(self, num_options=500):

        unique_matchups = self.unique_matchup_pairs(self.opponents)
        min_max = self.get_model_predictions()[['player', 'min_score', 'max_score']]
        
        dists = pd.DataFrame()
        for matchup in unique_matchups:

            means_sample = self.player_data[self.player_data.team.isin(matchup)].reset_index(drop=True)
            if len(means_sample) > 0:
                covar_sample = self.covar.loc[self.covar.player.isin(means_sample.player), means_sample.player]
                min_max_sample = pd.merge(means_sample[['player']], min_max, on='player', how='left')

                mean_vals = means_sample.pred_fp_per_game.values
                covar_vals = covar_sample.values
                bound_vals = min_max_sample[['min_score', 'max_score']].values

                results = self.bounded_multivariate_dist(mean_vals, covar_vals, bound_vals, size=num_options)
                results = pd.DataFrame(results, columns=means_sample.player).T
                dists = pd.concat([dists, results], axis=0)
            
        dists = dists.reset_index().rename(columns={'index': 'player'})
        dists = pd.merge(self.player_data[['player']], dists, on='player').drop('player', axis=1)

        return dists.reset_index(drop=True)

    
    def create_h_ownership(self):
        mean_own, std_own = pd.read_sql_query(f'''SELECT ownership_mean, ownership_std
                                                  FROM Mean_Ownership
                                                  WHERE week={self.week}
                                                        AND year={self.set_year}
                                                        AND ownership_vers='{self.ownership_vers}'
                                                        AND pred_vers='{self.pred_vers}'
                                                        AND million_ens_vers='{self.million_ens_vers}'
                                        ''', self.conn).values[0]

        sampled_mean = np.random.normal(mean_own, std_own, size=1).reshape(1, 1)
        
        # Only apply pos_or_neg multiplier for standard_ln ownership
        if self.ownership_vers == 'standard_ln' and self.pos_or_neg != 1:
            mean_own = self.pos_or_neg * sampled_mean / 1.1
        else:
            mean_own = sampled_mean
        
        return mean_own
    

    def get_team_game_mapping(self):
        """Create a mapping of teams to their games for hierarchical Gumbel noise"""
        team_to_game = {}
        processed_games = set()
        
        for team in self.unique_teams:
            if team in self.opponents:
                opp_team = self.opponents[team]
                # Create a consistent game identifier (sorted teams)
                game_id = tuple(sorted([team, opp_team]))
                
                if game_id not in processed_games:
                    processed_games.add(game_id)
                
                team_to_game[team] = game_id
        
        return team_to_game
    
    def calculate_position_scales(self, cur_pred_fps):
        """
        Calculate adaptive scaling factors for each position based on top-tier player ranges.
        This makes temperature scale-invariant across different optimization metrics (points, ROI, etc.)
        """
        # Position depth varies - these define how many top players to consider for each position
        top_n_by_position = {
            'QB': 4,   # Fewer QBs rostered
            'RB': 8,   # Deep position
            'WR': 12,   # Deepest position
            'TE': 4,   # Shallow position
            'DEF': 4   # Very shallow
        }
        
        position_scales = {}
        for pos in ['QB', 'RB', 'WR', 'TE', 'DEF']:
            pos_mask = np.array([p == pos for p in self.positions])
            if pos_mask.sum() > 0:
                pos_values = cur_pred_fps[pos_mask]
                sorted_vals = np.sort(pos_values)[::-1]  # Sort descending (best first)
                
                # Take top N players for this position
                n = min(top_n_by_position[pos], len(sorted_vals))
                top_tier = sorted_vals[:n]
                
                # Use range of competitive tier as scale
                scale = top_tier.max() - top_tier.min()
                
                # Fallback if scale is too small (avoid division issues)
                if scale < 0.01:
                    scale = np.std(pos_values) if len(pos_values) > 1 else 1.0
                
                position_scales[pos] = scale
        
        return position_scales

    def initialize_player_data(self):
        
        if self.use_covar: 
            player_data = self.get_covar_means()
            self.covar = self.pull_covar()
        else:
            player_data = self.get_model_predictions()

        self.player_data = self.join_salary(player_data)
        self.player_data = self.join_opp(self.player_data)
        self.opponents = self.get_matchups()


    def run_sim(self, conn, to_add, to_drop, num_iters, opt_metric, std_dev_type, ownership_vers, 
                num_options, num_avg_pts, pos_or_neg,
                min_pass_catchers, rb_in_stack, min_opp_team, max_teams_lineup,max_salary_remain,
                max_overlap, prev_qb_wt, prev_def_wt, prev_te_wt, previous_lineups, wr_flex_pct, rb_flex_pct,
                use_ownership, overlap_constraint, min_own_three_ten, min_own_less_three, player_gumbel_temp=0, team_gumbel_temp=0, game_gumbel_temp=0):
        
        self.conn = conn
        self.opt_metric = opt_metric
        self.std_dev_type = std_dev_type

        if self.opt_metric != 'points':
            self.use_covar = False

        self.initialize_player_data()

        self.prepare_data(use_ownership,ownership_vers, num_avg_pts, pos_or_neg, min_own_three_ten, min_own_less_three)
        if wr_flex_pct == 'auto': self.flex_mode = 'auto'
        else: self.flex_mode = 'fixed'

        player_counts = {player: 0 for player in self.player_data['player']}
        successful_iterations = 0
                
        for _ in range(num_iters):

            iters_run = 0
            success = False
            
            while not success and iters_run < 5:
                
                self.position_counts = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
                if self.flex_mode == 'fixed':
                    self.set_position_counts(float(wr_flex_pct), rb_flex_pct)
                
                iters_run += 1
                if iters_run >= 2: 
                    use_ownership = 0
                if iters_run >= 3:
                    min_own_three_ten = 0  # Disable low ownership constraints
                    min_own_less_three = 0
                    max_overlap = 8
                    overlap_constraint = 'standard'
                    

                cur_pred_fps = self.sample_points(num_avg_pts, num_avg_pts)
                if use_ownership == 1:
                    cur_ownership = self.sample_ownership(num_avg_pts, num_avg_pts)
                else:
                    cur_ownership = np.zeros(len(cur_pred_fps))
                # Sample standard ownership for low ownership constraints
                if min_own_three_ten > 0 or min_own_less_three > 0:
                    std_ownership_pcts = self.sample_standard_ownership(num_avg_pts, num_avg_pts)
                else:
                    std_ownership_pcts = None

                c, A, b, G, h = self.setup_optimization_problem(to_add, to_drop, cur_pred_fps, cur_ownership, min_pass_catchers, rb_in_stack,
                                                                min_opp_team, max_teams_lineup, max_salary_remain,
                                                                max_overlap, prev_qb_wt, prev_def_wt, prev_te_wt, previous_lineups,
                                                                use_ownership, overlap_constraint, std_ownership_pcts, min_own_three_ten, 
                                                                min_own_less_three, player_gumbel_temp, team_gumbel_temp, game_gumbel_temp)


                status, x = self.solve_optimization(c, G, h, A, b)
                if status != 'optimal':
                    print(f"Optimization failed in iteration {_ + 1}. Status: {status}")
                    continue

                selected_players = self.process_results(x)
                
                for player in selected_players:
                    player_counts[player] += 1
                
                successful_iterations += 1
                success = True
                
        if successful_iterations == 0:
            print("All optimization attempts failed.")
            return None, None
        
        player_percentages = self.calculate_player_perc(player_counts, successful_iterations)
        
        return selected_players, player_percentages
    
    def sample_points(self, num_options, num_avg_pts):
        current_points = self.pred_fps.iloc[:, np.random.choice(range(4, num_options+4), size=num_avg_pts)].mean(axis=1)
        return current_points
    
    def sample_ownership(self, num_options, num_avg_pts):
        current_ownership = self.ownerships.iloc[:, np.random.choice(range(4, num_options+4), size=num_avg_pts)].mean(axis=1)
        # Only apply pos_or_neg multiplier for standard_ln ownership
        if self.ownership_vers == 'standard_ln':
            return current_ownership * self.pos_or_neg
        else:
            return current_ownership
    
    def sample_standard_ownership(self, num_options, num_avg_pts):
        # Sample from standard_ln ownership and convert to percentages
        current_std_ownership = self.standard_ownerships.iloc[:, np.random.choice(range(4, num_options+4), size=num_avg_pts)].mean(axis=1)
        ownership_percentages = np.exp(current_std_ownership) * 100
        return ownership_percentages
        
    def prepare_data(self, use_ownership, ownership_vers, num_options, pos_or_neg, min_own_three_ten=0, min_own_less_three=0):
        
        self.df = self.player_data.copy()
        self.pos_or_neg = pos_or_neg
        self.ownership_vers = ownership_vers
        self.min_own_three_ten = min_own_three_ten
        self.min_own_less_three = min_own_less_three
        
        if use_ownership == 1:
            self.ownership_data = self.join_ownership_pred(self.df)
            self.min_ownership = self.create_h_ownership()[0][0]

        # Load standard_ln ownership data separately for low ownership constraints
        if min_own_three_ten > 0 or min_own_less_three > 0:
            self.standard_ownership_data = self.join_standard_ownership_pred(self.df)

        self.players = self.player_data['player'].values
        self.salaries = self.player_data['salary'].values
        self.positions = self.player_data['pos'].values
        self.teams = self.player_data['team'].values
        self.n_players = len(self.players)
        self.unique_teams = list(set(self.teams))
        self.n_teams = len(self.unique_teams)
        
        # Create team-to-game mapping after unique_teams is set
        self.team_to_game = self.get_team_game_mapping()

        self.pred_fps = self.get_predictions('pred_fp_per_game', num_options=num_options+5)
        if use_ownership == 1:
            self.ownerships = self.get_predictions('pred_ownership', ownership=True, num_options=num_options+5)
        if min_own_three_ten > 0 or min_own_less_three > 0:
            self.standard_ownerships = self.get_predictions('pred_ownership', ownership=True, num_options=num_options+5, use_standard=True)

    def setup_optimization_problem(self, to_add, to_drop, cur_pred_fps, cur_ownership, min_pass_catchers, rb_in_stack,
                                   min_opp_team, max_teams_lineup, max_salary_remain, max_overlap, prev_qb_wt, 
                                   prev_def_wt, prev_te_wt, previous_lineups, use_ownership, overlap_constraint, std_ownership_pcts=None, 
                                   min_own_three_ten=0, min_own_less_three=0, player_gumbel_temp=0, team_gumbel_temp=0, game_gumbel_temp=0):
        
        n_variables = self.n_players + len(self.unique_teams)
        
        c = self.create_objective_function(cur_pred_fps, player_gumbel_temp, team_gumbel_temp, game_gumbel_temp)
        A, b = self.create_equality_constraints()
        G, h = self.create_inequality_constraints(to_add, to_drop, cur_ownership, n_variables, min_pass_catchers, rb_in_stack,
                                                  min_opp_team, max_teams_lineup, max_salary_remain,
                                                  max_overlap, prev_qb_wt, prev_def_wt, prev_te_wt, previous_lineups,
                                                  use_ownership, overlap_constraint, std_ownership_pcts, min_own_three_ten, min_own_less_three)
        return c, A, b, G, h

    def create_objective_function(self, cur_pred_fps, player_gumbel_temp=0, team_gumbel_temp=0, game_gumbel_temp=0):
        """
        Apply hierarchical Gumbel noise at three levels with adaptive scaling:
        1. Game-level: Shootout vs defensive battle (both teams benefit/suffer together)
        2. Team-level: Winner vs loser (asymmetric within game)
        3. Player-level: Individual variance
        
        Temperatures are automatically scaled based on the competitive tier spread,
        making them work consistently across different optimization metrics (points, ROI, etc.)
        """
        if player_gumbel_temp > 0 or team_gumbel_temp > 0 or game_gumbel_temp > 0:
            
            # Calculate adaptive scaling factors based on top-tier player ranges
            position_scales = self.calculate_position_scales(cur_pred_fps)
            
            # Calculate overall scale for game/team level (use median of position scales)
            overall_scale = np.median(list(position_scales.values()))
            
            # 1. Game-level noise (shootout vs defensive battle)
            if game_gumbel_temp > 0:
                scaled_game_temp = game_gumbel_temp * overall_scale
                game_noise_dict = {}
                for game_id in set(self.team_to_game.values()):
                    game_noise_dict[game_id] = np.random.gumbel(0, scaled_game_temp)
                
                # Apply game noise - INVERTED for defenses (shootout hurts DEF, defensive battle helps DEF)
                game_noise_array = np.array([
                    -game_noise_dict[self.team_to_game[t]] if pos == 'DEF' 
                    else game_noise_dict[self.team_to_game[t]]
                    for t, pos in zip(self.teams, self.positions)
                ])
                cur_pred_fps = cur_pred_fps + game_noise_array
            
            # 2. Team-level noise (who wins/loses within the game)
            if team_gumbel_temp > 0:
                scaled_team_temp = team_gumbel_temp * overall_scale
                team_noise_dict = {team: np.random.gumbel(0, scaled_team_temp) 
                                  for team in self.unique_teams}
                team_noise_array = np.array([team_noise_dict[t] for t in self.teams])
                cur_pred_fps = cur_pred_fps + team_noise_array
            
            # 3. Player-level noise (individual variance, position-specific scaling)
            if player_gumbel_temp > 0:
                player_noise = np.array([
                    np.random.gumbel(0, player_gumbel_temp * position_scales[pos])
                    for pos in self.positions
                ])
                cur_pred_fps = cur_pred_fps + player_noise
        
        return matrix(list(-cur_pred_fps) + [0] * len(self.unique_teams), tc='d')
    

    def set_position_counts(self, wr_flex_pct, rb_flex_pct):
        
        rb_flex_pct = np.min([rb_flex_pct, 1-wr_flex_pct])
        te_flex_pct = np.max([1 - rb_flex_pct - wr_flex_pct, 0])
        flex_pos = np.random.choice(['RB', 'WR', 'TE'], 
                                p=[rb_flex_pct, wr_flex_pct, te_flex_pct])
        self.position_counts[flex_pos] += 1
        self.num_pos = 9.0

    
    def create_equality_constraints(self):
        A = []
        b = []

        # Constraint: Exact total number of players (always 9)
        A.append([1.0] * self.n_players + [0.0] * len(self.unique_teams))
        b.append(9.0)  # Explicitly set to 9 players

        if self.flex_mode == 'auto':
            # Exact position requirements for QB and DEF
            for pos in ['QB', 'DEF']:
                constraint = [1.0 if p == pos else 0.0 for p in self.positions] + [0.0] * len(self.unique_teams)
                A.append(constraint)
                b.append(float(self.position_counts[pos]))

            # Total RB/WR/TE must equal base requirements plus one (7 total)
            flex_constraint = []
            for pos in self.positions:
                if pos in ['RB', 'WR', 'TE']:
                    flex_constraint.append(1.0)
                else:
                    flex_constraint.append(0.0)
                    
            flex_constraint.extend([0.0] * len(self.unique_teams))
            A.append(flex_constraint)
            base_flex_total = sum([self.position_counts[pos] for pos in ['RB', 'WR', 'TE']])
            b.append(float(base_flex_total + 1))  # Should equal 7 (6 base + 1 flex)
        else:
            # Fixed position constraints
            for pos, count in self.position_counts.items():
                constraint = [1.0 if p == pos else 0.0 for p in self.positions] + [0.0] * len(self.unique_teams)
                A.append(constraint)
                b.append(float(count))

        return matrix(A, tc='d'), matrix(b, tc='d')
    
    def flex_constraint(self, G, h):
        for pos in ['RB', 'WR', 'TE']:
            constraint = [-1.0 if p == pos else 0.0 for p in self.positions] + [0.0] * len(self.unique_teams)
            G.append(constraint)
            h.append(-float(self.position_counts[pos]))
        return G, h


    def create_inequality_constraints(self, to_add, to_drop, cur_ownership, n_variables, min_pass_catchers, rb_in_stack,
                                      min_opp_team, max_teams_lineup, max_salary_remain, max_overlap, prev_qb_wt,
                                      prev_def_wt, prev_te_wt, previous_lineups, use_ownership, overlap_constraint, std_ownership_pcts=None, min_own_three_ten=0, min_own_less_three=0):
        G = []
        h = []

        self.add_salary_constraint(G, h, max_salary_remain)
        self.add_forced_players_constraint(G, h, n_variables, to_add)
        self.add_excluded_players_constraint(G, h, n_variables, to_drop)
        self.add_qb_dst_constraint(G, h, n_variables)
        if self.flex_mode == 'auto': self.flex_constraint(G, h)

        if len(to_add) <= 7:
            if use_ownership==1: 
                print('Using ownership')
                self.add_ownership_constraint(G, h, cur_ownership)
                # Add minimum player ownership constraint when using pos_or_neg = -1
                self.add_min_player_ownership_constraint(G, h, n_variables, min_own_threshold=-5)
            if min_own_three_ten > 0:
                self.add_three_ten_ownership_constraint(G, h, std_ownership_pcts, min_own_three_ten)
            if min_own_less_three > 0:
                self.add_less_three_ownership_constraint(G, h, std_ownership_pcts, min_own_less_three)
            self.add_stacking_constraints(G, h, n_variables, min_pass_catchers, rb_in_stack)
            self.add_opposing_team_constraint(G, h, n_variables, min_opp_team)
            self.add_max_teams_constraint(G, h, n_variables, max_teams_lineup)
            self.add_overlap_constraint(G, h, n_variables, max_overlap, previous_lineups, prev_qb_wt, prev_def_wt, prev_te_wt, overlap_constraint)
            
        return matrix(np.array(G), tc='d'), matrix(h, tc='d')
    

    def add_salary_constraint(self, G, h, max_salary_remain):
        G.append(list(self.salaries) + [0] * len(self.unique_teams))
        h.append(float(self.salary_cap))

        # New constraint: Total salary >= SALARY_CAP - max_salary_remain
        G.append([-s for s in self.salaries] + [0] * len(self.unique_teams))
        h.append(-float(self.salary_cap - max_salary_remain))

    def add_forced_players_constraint(self, G, h, n_variables, to_add):
        for player in to_add:
            if player in self.players:
                index = list(self.players).index(player)
                constraint = [0] * n_variables
                constraint[index] = -1  # Force this player to be selected
                G.append(constraint)
                h.append(-1.0)  # Must be less than or equal to -1, forcing selection

    
    def add_excluded_players_constraint(self, G, h, n_variables, to_drop):
        for player in to_drop:
            if player in self.players:
                index = list(self.players).index(player)
                constraint = [0] * n_variables
                constraint[index] = 1  # Prevent this player from being selected
                G.append(constraint)
                h.append(0.0)  # Must be less than or equal to 0, forcing non-selection

    def add_stacking_constraints(self, G, h, n_variables, min_pass_catchers, rb_in_stack):
        for team in self.unique_teams:
            self.add_pass_catcher_stack(G, h, n_variables, team, min_pass_catchers, rb_in_stack)

    def add_pass_catcher_stack(self, G, h, n_variables, team, min_pass_catchers, rb_in_stack):
        qb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'QB' and t == team]
        wr_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'WR' and t == team]
        te_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'TE' and t == team]
        
        # Get RB indices if rb_in_stack is enabled
        if rb_in_stack == 1:
            rb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'RB' and t == team]
        else:
            rb_indices = []
        
        # Combined pass-catcher stacking constraint: if QB selected, need minimum pass catchers (WR + TE + RB if enabled)
        all_pass_catcher_indices = wr_indices + te_indices + rb_indices
        if qb_indices and all_pass_catcher_indices and min_pass_catchers > 0:
            for qb_index in qb_indices:
                constraint = [0] * n_variables
                constraint[qb_index] = min_pass_catchers          # If QB selected, need N pass catchers
                for wr_index in wr_indices:
                    constraint[wr_index] = -1                     # Each WR counts as -1
                for te_index in te_indices:
                    constraint[te_index] = -1                     # Each TE counts as -1
                if rb_in_stack == 1:
                    for rb_index in rb_indices:
                        constraint[rb_index] = -1                 # Each RB counts as -1 when enabled
                G.append(constraint)
                h.append(0.0)                                     # Right hand side

    def add_qb_dst_constraint(self, G, h, n_variables):
        for team in self.unique_teams:
            qb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'QB' and t == team]
            opposing_team = self.opponents[team]
            opposing_dst_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'DEF' and t == opposing_team]
            
            if qb_indices and opposing_dst_indices:
                constraint = [0] * n_variables
                for qb_index in qb_indices:
                    constraint[qb_index] = 1
                for dst_index in opposing_dst_indices:
                    constraint[dst_index] = 1
                G.append(constraint)
                h.append(1.0)

    def add_opposing_team_constraint(self, G, h, n_variables, min_opp_team):
        for team in self.unique_teams:
            qb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'QB' and t == team]
            opposing_team = self.opponents[team]
            opposing_player_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if t == opposing_team and pos != 'DEF']
            
            if qb_indices and opposing_player_indices:
                constraint = [0] * n_variables
                for qb_index in qb_indices:
                    constraint[qb_index] = min_opp_team
                for opp_index in opposing_player_indices:
                    constraint[opp_index] = -1
                G.append(constraint)
                h.append(0.0)

    def add_max_teams_constraint(self, G, h, n_variables, max_teams_lineup):
        for i, team in enumerate(self.unique_teams):
            team_indices = [i for i, t in enumerate(self.teams) if t == team]
            
            constraint = [0] * n_variables
            for idx in team_indices:
                constraint[idx] = -1
            constraint[self.n_players + i] = 1
            G.append(constraint)
            h.append(0.0)

            constraint = [0] * n_variables
            for idx in team_indices:
                constraint[idx] = 1
            constraint[self.n_players + i] = -6
            G.append(constraint)
            h.append(0.0)

        G.append([0] * self.n_players + [1] * len(self.unique_teams))
        h.append(float(max_teams_lineup))

    def add_overlap_constraint(self, G, h, n_variables, max_overlap, previous_lineups, prev_qb_wt, prev_def_wt, prev_te_wt, overlap_constraint):
        for prev_lineup in previous_lineups:
            constraint = [0] * n_variables
            for i, (player, position) in enumerate(zip(self.players, self.positions)):
                if player in prev_lineup:
                    if position == 'QB':
                        constraint[i] = prev_qb_wt
                    elif position == 'DEF':
                        constraint[i] = prev_def_wt
                    elif position == 'TE':
                        constraint[i] = prev_te_wt
                    else:
                        constraint[i] = 1
            G.append(constraint)
            if overlap_constraint=='standard':
                h.append(float(max_overlap))
            elif overlap_constraint=='plus_wts':
                h.append(float(max_overlap+int(prev_qb_wt)+int(prev_def_wt)+int(prev_te_wt)))
            elif overlap_constraint=='minus_one':
                h.append(float(max_overlap-int(prev_qb_wt-1)-int(prev_def_wt-1)-int(prev_te_wt-1)))
            elif overlap_constraint=='div_two':
                h.append(float(max_overlap-int(prev_qb_wt/2)-int(prev_def_wt/2)-int(prev_te_wt/2)))

    def add_ownership_constraint(self, G, h, cur_ownership):
        G.append([-o for o in cur_ownership] + [0] * len(self.unique_teams))
        h.append(-self.min_ownership)

    def add_three_ten_ownership_constraint(self, G, h, std_ownership_pcts, min_own_three_ten):
        if std_ownership_pcts is not None and min_own_three_ten > 0:
            # Create constraint directly: sum of players in 3-10% bucket >= min_own_three_ten
            constraint = []
            for i, ownership_pct in enumerate(std_ownership_pcts):
                if 3 <= ownership_pct <= 10:
                    constraint.append(-1.0)  # This player counts toward the constraint
                else:
                    constraint.append(0.0)   # This player doesn't count
            
            constraint.extend([0.0] * len(self.unique_teams))  # Team variables
            G.append(constraint)
            h.append(-float(min_own_three_ten))  # Right-hand side

    def add_less_three_ownership_constraint(self, G, h, std_ownership_pcts, min_own_less_three):
        if std_ownership_pcts is not None and min_own_less_three > 0:
            # Create constraint directly: sum of players in <3% bucket >= min_own_less_three
            constraint = []
            for i, ownership_pct in enumerate(std_ownership_pcts):
                if ownership_pct < 3:
                    constraint.append(-1.0)  # This player counts toward the constraint
                else:
                    constraint.append(0.0)   # This player doesn't count
            
            constraint.extend([0.0] * len(self.unique_teams))  # Team variables
            G.append(constraint)
            h.append(-float(min_own_less_three))  # Right-hand side

    def add_min_player_ownership_constraint(self, G, h, n_variables, min_own_threshold=-5):
        """
        When using pos_or_neg=-1 with standard_ln ownership, prevent extremely
        low-owned players from being selected to avoid gaming the constraint.
        
        Parameters:
        - min_own_threshold: Minimum ownership threshold on ln scale (default -5 = ~0.67% owned)
                           -4.6 = ~1% owned, -3.0 = ~5% owned
        """
        if self.pos_or_neg == -1 and self.ownership_vers == 'standard_ln':
            # Check against ORIGINAL negative ln values in ownership_data (before pos_or_neg flip)
            for i, ownership_val in enumerate(self.ownership_data['pred_ownership'].values):
                if ownership_val < min_own_threshold:
                    constraint = [0] * n_variables
                    constraint[i] = 1  # Prevent this player from being selected
                    G.append(constraint)
                    h.append(0.0)

    def solve_optimization(self, c, G, h, A, b):
        return ilp(c, G, h, A.T, b, B=set(range(len(c))))

    def process_results(self, x):
        selected_indices = [i for i in range(self.n_players) if x[i] > 0.5]
        selected_players = [self.players[i] for i in selected_indices]
        selected_salaries = [self.salaries[i] for i in selected_indices]
        selected_teams = [self.teams[i] for i in selected_indices]
        selected_positions = [self.positions[i] for i in selected_indices]

        if self.print_results:
            self.print_out_results(selected_players, selected_salaries, 
                                   selected_teams, selected_positions)

        return selected_players
    
    def calculate_player_perc(self, player_counts, successful_iterations):
        # Convert counts to percentages
        player_percentages = {player: np.round((count / successful_iterations) * 100,1)
                            for player, count in player_counts.items() if count > 0}
        
        player_percentages = pd.DataFrame(player_percentages, index=[0]).T.reset_index()
        player_percentages.columns = ['player', 'SelectionCounts']
        player_percentages = pd.merge(player_percentages, self.player_data[['player', 'salary']], on='player')
        player_percentages = player_percentages.sort_values(by='SelectionCounts', ascending=False).reset_index(drop=True)
        return player_percentages

#%%

from joblib import Parallel, delayed

class RunSim:

    def __init__(self, db_path, week, year, pred_vers, reg_ens_vers, million_ens_vers, total_lineups, pull_stats=True):
        
        if '.sqlite3' not in db_path: self.db_path = f'{db_path}/Simulation/Simulation_{year}.sqlite3'
        else: self.db_path = db_path

        self.week = week
        self.year = year
        self.pred_vers = pred_vers
        self.reg_ens_vers = reg_ens_vers
        self.million_ens_vers = million_ens_vers
        self.total_lineups = total_lineups
        self.col_ordering = ['opt_metric', 'std_dev_type', 'covar_type', 'num_iters',
                             'ownership_vers',  'num_options', 'num_avg_pts',
                             'pos_or_neg', 'min_pass_catchers', 'rb_in_stack', 'min_opp_team', 'max_teams_lineup',
                             'max_salary_remain', 'max_overlap', 'prev_qb_wt', 'prev_def_wt', 'prev_te_wt', 'wr_flex_pct', 
                             'rb_flex_pct', 'use_ownership', 'overlap_constraint', 'min_own_three_ten', 'min_own_less_three', 'player_gumbel_temp', 'team_gumbel_temp', 'game_gumbel_temp']
        if pull_stats:
            try:
                results_conn = sqlite3.connect(f'{db_path}/DK_Results.sqlite3', timeout=60)
                self.player_stats = self.pull_past_points(results_conn, week, year)
                self.prizes = self.get_past_prizes(results_conn, week, year)
            except:
                print('No Stats or DK Results')

    def create_conn(self):
        return sqlite3.connect(self.db_path, timeout=60)
        

    @staticmethod
    def pull_past_points(results_conn, week, year):
        points = pd.read_sql_query(f'''SELECT player, 
                                              player_points as fantasy_pts
                                       FROM Contest_Points
                                        WHERE week={week}
                                               AND year={year}
                                               AND Contest='Million' ''', results_conn)
        return points


    def calc_winnings(self, to_add):
        results = pd.DataFrame(to_add, columns=['player'])
        results = pd.merge(results, self.player_stats, on='player', how='left').fillna(0)
        total_pts = results.fantasy_pts.sum()
        idx_match = np.argmin(abs(self.prizes.Points - total_pts))
        prize_money = self.prizes.loc[idx_match, 'prize']

        return prize_money, results

    @staticmethod
    def get_past_prizes(results_conn, week, year):
        prizes = pd.read_sql_query(f'''SELECT Rank, Points, prize
                                        FROM Contest_Results
                                        WHERE week={week}
                                            AND year={year}
                                            AND Contest='Million'
                                            AND `Rank`<75000 ''', results_conn)
        return prizes
    
    @staticmethod
    def adjust_high_winnings(tw, max_adjust=10000):
        tw = np.array(tw)
        tw[tw>max_adjust] = max_adjust
        return list(tw)
    
    def generate_param_list(self, d):
    
        extra_keys = [k for k,_ in d.items() if k not in self.col_ordering]
        missing_keys = [k for k in self.col_ordering if k not in d.keys()]

        if len(extra_keys) > 0:
            raise ValueError(f'Extra keys: {extra_keys}')
        if len(missing_keys) > 0:
            raise ValueError(f'Missing keys: {missing_keys}')

        d = {k: d[k] for k in self.col_ordering}
        params = []
        for i in range(self.total_lineups):
            cur_params = []
            for k, param_options in d.items():
                param_vars = list(param_options.keys())
                param_prob = list(param_options.values())
                cur_choice = np.random.choice(param_vars, p=param_prob)
                cur_params.append(cur_choice)


            cur_params.append(i)
            params.append(cur_params)

        return params
    
    def setup_sim(self, params):

        p = {k: v for k,v in zip(self.col_ordering, params)}
        print(p)
    
        if p['covar_type']=='no_covar': p['use_covar']=False
        else: p['use_covar']=True

        conn = self.create_conn()
        sim = FullLineupSim(self.week, self.year, conn, self.pred_vers, self.reg_ens_vers, 
                            self.million_ens_vers, p['use_covar'], p['covar_type'], print_results=False)
        conn.close()
        return sim, p
    
    
    def run_single_iter(self, sim, p, to_add, to_drop_selected, previous_lineups):

        to_drop = []
        to_drop.extend(to_drop_selected)

        conn = self.create_conn()
        last_lineup = None
        i = 0
        while last_lineup is None and i < 10:
            last_lineup, player_cnts = sim.run_sim(conn, to_add, to_drop, p['num_iters'], p['opt_metric'], p['std_dev_type'], p['ownership_vers'], 
                                                   p['num_options'], p['num_avg_pts'], p['pos_or_neg'], p['min_pass_catchers'], p['rb_in_stack'], p['min_opp_team'], p['max_teams_lineup'],
                                                   p['max_salary_remain'], p['max_overlap'], p['prev_qb_wt'], p['prev_def_wt'], p['prev_te_wt'], previous_lineups,
                                                   p['wr_flex_pct'], p['rb_flex_pct'], p['use_ownership'], p['overlap_constraint'], p['min_own_three_ten'], p['min_own_less_three'], p['player_gumbel_temp'], p['team_gumbel_temp'], p['game_gumbel_temp'])
            i += 1
        conn.close() 

        if player_cnts is not None:
            player_cnts = player_cnts[~player_cnts.player.isin(to_add)].reset_index(drop=True)
            self.player_data = sim.player_data.copy()
        return last_lineup, player_cnts
                    
    def run_full_lineup(self, params, to_add, to_drop, previous_lineups):

        sim, p = self.setup_sim(params)

        if p['num_iters'] == 1:
            to_add, _ = self.run_single_iter(sim, p, to_add, to_drop, previous_lineups)

        else:
            i = 0  # Initialize the iteration counter
            while len(to_add) < 9 and i < 15:  # Use a while loop to control iterations and break if necessary
                _, results = self.run_single_iter(sim, p, to_add, to_drop, previous_lineups)
                
                if results is not None:
                    prob = results.loc[:p['top_n_choices'], 'SelectionCounts'] / results.loc[:p['top_n_choices'], 'SelectionCounts'].sum()
            
                    try: 
                        selected_player = np.random.choice(results.loc[:p['top_n_choices'], 'player'], p=prob)
                        to_add.append(selected_player)
                    except: 
                        pass
                i += 1  # Increment the iteration counter    

        if to_add is None: to_add = []
        del sim; gc.collect()
        return to_add
    
    def run_multiple_lineups(self, params, calc_winnings=False, parallelize=False, n_jobs=-1, verbose=0):
        
        existing_players =[]
        if parallelize:
            all_lineups = Parallel(n_jobs=n_jobs, verbose=verbose)(
                                delayed(self.run_full_lineup)(cur_param, existing_players)
                                for cur_param in params
                                )
                                
        else:
            all_lineups = []
            for cur_params in params:
                to_add = self.run_full_lineup(cur_params, to_add=[], to_drop=[], previous_lineups=all_lineups)
                all_lineups.append(to_add)

        if calc_winnings:
            total_winnings = 0
            winnings_list = []
            player_results = pd.DataFrame()
            for i, lineup in enumerate(all_lineups):
                
                if len(lineup)<9: 
                    continue
                winnings, player_results_cur = self.calc_winnings(lineup)
                total_winnings += winnings
                winnings_list.append(winnings)

                player_results_cur = player_results_cur.assign(lineup_num=i, week=self.week, year=self.year)
                player_results = pd.concat([player_results, player_results_cur])

            return total_winnings, player_results, winnings_list
        
        else: 
            pass

        return all_lineups

#%%

# import warnings
# warnings.filterwarnings('ignore')

# week = 3
# year = 2025
# total_lineups = 50

# model_vers = {
#             'million_ens_vers': 'random_full_stack_matt0_brier1_include2_kfold3',
#             'pred_vers': 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_quick',
#             'reg_ens_vers': 'random_full_stack_sera0_rsq0_mse1_include2_kfold3',
#             }


# d = {'covar_type': {'kmeans_pred_trunc_new': 0.0,
#                 'no_covar': 1.0,
#                 'team_points_trunc': 0.0},
#  'game_gumbel_temp': {0: 0.5, 0.1: 0.0, 0.2: 0.5, 0.25: 0.0, 0.3: 0.0},
#  'max_overlap': {3: 0.0,
#                  5: 0.0,
#                  6: 0.0,
#                  7: 1.0,
#                  8: 0.0,
#                  9: 0.0,
#                  11: 0.0,
#                  13: 0.0},
#  'max_salary_remain': {500: 0.0, 750: 0.3, 1000: 0.7},
#  'max_teams_lineup': {4: 0.0, 5: 1.0, 6: 0.0, 8: 0.0, 9: 0.0},
#  'min_opp_team': {0: 1.0, 1: 0.0, 2: 0.0},
#  'min_own_less_three': {0: 1.0, 1: 0.0, 2: 0.0, 5: 0.0},
#  'min_own_three_ten': {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
#  'min_pass_catchers': {0: 0.0, 1: 0.0, 2: 0.9, 3: 0.1, 4: 0.0},
#  'num_avg_pts': {10: 0.0,
#                  25: 0.0,
#                  50: 0.0,
#                  100: 0.0,
#                  200: 0.5,
#                  500: 0.5,
#                  1000: 0.0},
#  'num_iters': {1: 1.0},
#  'num_options': {50: 0.0,
#                  200: 0.0,
#                  500: 0.0,
#                  1000: 0.0,
#                  2000: 0.0,
#                  3000: 0.0,
#                  5000: 1.0},
#  'opt_metric': {'etr_points': 0.0,
#                 'million': 0.0,
#                 'points': 0.4,
#                 'roi': 0.6,
#                 'roitop': 0.0},
#  'overlap_constraint': {'constraint_minus': 0.0, 'standard': 1.0},
#  'ownership_vers': {'mil_div_standard_ln': 0.0,
#                     'mil_only': 0.0,
#                     'mil_times_standard_ln': 0.0,
#                     'roi_only': 0.2,
#                     'roi_times_standard_ln': 0.2,
#                     'roitop_only': 0.2,
#                     'roitop_times_standard_ln': 0.2,
#                     'standard_ln': 0.2},
#  'player_gumbel_temp': {0: 1.0,
#                         0.025: 0.0,
#                         0.05: 0.0,
#                         0.1: 0.0,
#                         0.15: 0.0,
#                         0.2: 0.0},
#  'pos_or_neg': {1: 1.0},
#  'prev_def_wt': {1: 1.0, 2: 0.0},
#  'prev_qb_wt': {1: 1.0, 2: 0.0, 3: 0.0, 5: 0.0, 7: 0.0},
#  'prev_te_wt': {1: 1.0, 2: 0.0, 3: 0.0},
#  'rb_flex_pct': {0: 0.0, 0.3: 0.0, 0.35: 0.3, 0.4: 0.7, 0.5: 0.0},
#  'rb_in_stack': {0: 0.9, 1: 0.1},
#  'std_dev_type': {'spline_class80_q80_matt0_brier1_kfold3': 0.5,
#                   'spline_pred_class80_matt0_brier1_kfold3': 0.0,
#                   'spline_pred_class80_q80_matt0_brier1_kfold3': 0.5},
#  'team_gumbel_temp': {0: 1.0},
#  'use_ownership': {0: 0.7, 1: 0.3},
#  'wr_flex_pct': {0.5: 0.0, 0.6: 1.0, 0.65: 0.0, 'auto': 0.0}}

# print(f'Running week {week} for year {year}')

# pred_vers = model_vers['pred_vers']
# reg_ens_vers = model_vers['reg_ens_vers']
# million_ens_vers = model_vers['million_ens_vers']

# path = 'C:/Users/borys/OneDrive/Documents/Github/Daily_Fantasy_Data/Databases/'
# rs = RunSim(path, week, year, pred_vers, reg_ens_vers, million_ens_vers, total_lineups)
# params = rs.generate_param_list(d)

# #%%

# sim, p = rs.setup_sim(params[np.random.randint(0, len(params))])
# # sim, p = rs.setup_sim(params[0])

# to_add = []
# to_drop = []
# previous_lineups = []
# l1, l2 = rs.run_single_iter(sim, p, to_add, to_drop,previous_lineups)
# l2 = pd.merge(l2, sim.player_data[['player', 'pos']], on='player')
# print(l2.groupby('pos').size())
# l2

# #%%

# total_winnings, player_results, winnings_list = rs.run_multiple_lineups(params, calc_winnings=True)
# print(total_winnings)
# print(winnings_list)
# player_results.groupby('lineup_num').agg({'fantasy_pts': 'sum'}).sort_values(by='fantasy_pts', ascending=False)


# # %%
# player_results.groupby('player').agg({'fantasy_pts': 'mean', 'lineup_num': 'count'}).sort_values(by='lineup_num', ascending=False).iloc[:50]


# # %%

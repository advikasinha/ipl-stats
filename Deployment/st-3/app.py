import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple

class EnhancedIPLTeamPlanner:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.total_budget = 90.0  # crores
        self.retention_budget = 42.0  # max for 4 players
        self.min_players = 18
        self.max_players = 25
        self.max_overseas = 8
        self.playing_xi_overseas = 4
        
    def calculate_player_score(self, row: pd.Series) -> float:
        batting_score = 0
        if row['InningsBatted'] > 0:
            batting_score = (
                row['BattingAVG'] * 0.15 +
                row['BattingS/R'] * 0.25 +
                (row['4s'] + row['6s']) * 0.1 +
                row['RunsScored'] * 0.2 +
                (row['50s'] * 0.5 + row['100s']) * 0.2 +
                (1 - row['Ducks']/max(row['InningsBatted'], 1)) * 0.1
            )
        
        bowling_score = 0
        if row['InningsBowled'] > 0:
            bowling_score = (
                (300 - row.get('BowlingAVG', 300)) * 0.25 +
                (12 - row['EconomyRate']) * 0.25 +
                row['Wickets'] * 0.2 +
                (row['3s'] * 0.5 + row['5s']) * 0.2 +
                row['Maidens'] * 0.1
            )
        
        fielding_score = (row['CatchesTaken'] + row['StumpingsMade']) * 0.5
        
        experience_factor = min(1.2, 1 + (row['IPL Matches'] / 100) * 0.2)
        
        batting_score = np.clip(batting_score / 100, 0, 1)
        bowling_score = np.clip(bowling_score / 100, 0, 1)
        fielding_score = np.clip(fielding_score / 20, 0, 1)
        
        if row['Type'] == 'Batsman':
            final_score = (batting_score * 0.8 + bowling_score * 0.05 + fielding_score * 0.15) * experience_factor
        elif row['Type'] == 'Bowler':
            final_score = (batting_score * 0.05 + bowling_score * 0.8 + fielding_score * 0.15) * experience_factor
        elif row['Type'] == 'Wicket-Keeper':
            final_score = (batting_score * 0.6 + fielding_score * 0.4) * experience_factor
        else:
            final_score = (batting_score * 0.4 + bowling_score * 0.4 + fielding_score * 0.2) * experience_factor
            
        return final_score

    def generate_auction_strategy(self, retained_players: pd.DataFrame) -> pd.DataFrame:
        required_roles = {
            'Batsman': (5, 7),
            'Bowler': (5, 7),
            'All-Rounder': (3, 5),
            'Wicket-Keeper': (2, 3)
        }
        
        role_budgets = self.optimize_budget(retained_players, required_roles)
        available_players = self.data[~self.data['Name'].isin(retained_players['Name'])].copy()

        if available_players.empty:
            st.warning("No players are available for auction. Please check the input dataset.")
            return pd.DataFrame()

        available_players['Performance_Score'] = available_players.apply(
            self.calculate_player_score, axis=1
        )

        if available_players['Performance_Score'].isnull().all():
            st.warning("Performance scores could not be calculated. Check the input data.")
            return pd.DataFrame()

        available_players['Value_Score'] = available_players['Performance_Score'] / available_players['ValueinCR']

        auction_strategy_list = []
        
        for role, (min_needed, max_needed) in required_roles.items():
            retained_count = len(retained_players[retained_players['Type'] == role])
            still_needed = max(0, min_needed - retained_count)
            role_budget = role_budgets.get(role, 0)

            role_players = available_players[available_players['Type'] == role].copy()
            
            if len(role_players) > 0:
                avg_budget_per_player = role_budget / max(still_needed, 1)
                mean_performance = role_players['Performance_Score'].mean()
                
                for idx, player in role_players.iterrows():
                    suggested_bid = min(
                        avg_budget_per_player * (player['Performance_Score'] / mean_performance),
                        player['ValueinCR'] * 1.5
                    ) if mean_performance > 0 else avg_budget_per_player
                    
                    auction_strategy_list.append({
                        'Name': player['Name'],
                        'Type': player['Type'],
                        'National Side': player['National Side'],
                        'Performance_Score': player['Performance_Score'],
                        'Value_Score': player['Value_Score'],
                        'Suggested_Bid': suggested_bid,
                        'Base_Price': player['ValueinCR'],
                        'IPL_Experience': player['IPL Matches'],
                        'Priority': 'High' if len([x for x in auction_strategy_list if x['Type'] == role]) < still_needed else 'Medium'
                    })
        
        auction_strategy_df = pd.DataFrame(auction_strategy_list)

        required_columns = [
            'Name', 'Type', 'National Side', 'Performance_Score', 'Value_Score',
            'Suggested_Bid', 'Base_Price', 'IPL_Experience', 'Priority'
        ]
        for col in required_columns:
            if col not in auction_strategy_df.columns:
                auction_strategy_df[col] = None
        
        return auction_strategy_df

    def optimize_budget(self, retained_players: pd.DataFrame, target_roles: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        remaining_budget = self.total_budget - retained_players['retention_cost'].sum()
        players_needed = self.min_players - len(retained_players)
        
        role_budgets = {}
        total_priority = 0

        for role, (min_req, _) in target_roles.items():
            retained_count = len(retained_players[retained_players['Type'] == role])
            still_needed = max(0, min_req - retained_count)
            
            priority_weights = {
                'Batsman': 1.2,
                'Bowler': 1.2,
                'All-Rounder': 1.5,
                'Wicket-Keeper': 1.0
            }
            
            priority = still_needed * priority_weights[role]
            total_priority += priority
            role_budgets[role] = {'priority': priority, 'needed': still_needed}
        
        for role in role_budgets:
            if total_priority > 0:
                role_budgets[role] = (role_budgets[role]['priority'] / total_priority) * remaining_budget
            else:
                role_budgets[role] = remaining_budget / len(role_budgets)

        return role_budgets

    # Add other methods as needed

def main():
    st.title("Enhanced IPL Franchise Planning System")
    
    uploaded_file = st.file_uploader("Upload IPL dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        planner = EnhancedIPLTeamPlanner(data)

        franchise = st.selectbox("Select Your Franchise", sorted(data['Team'].unique()))

        team_data = data[data['Team'] == franchise]

        if team_data.empty:
            st.warning("No players found for the selected franchise.")
            return

        st.header("Retention Strategy")
        retention_suggestions = planner.suggest_retentions(franchise)

        if retention_suggestions.empty:
            st.warning("No retention suggestions available for the selected franchise.")
            return

        st.subheader("Suggested Retentions")
        st.dataframe(retention_suggestions)

        selected_retentions = []
        for _, player in retention_suggestions.iterrows():
            if st.checkbox(f"Retain {player['Name']} (₹{player['retention_cost']} cr)"):
                selected_retentions.append(player)

        selected_retentions_df = pd.DataFrame(selected_retentions)

        if selected_retentions_df.empty:
            st.warning("No players retained. Please select players for retention to proceed.")
            return

        st.header("Auction Strategy")
        auction_strategy = planner.generate_auction_strategy(selected_retentions_df)

        if auction_strategy.empty:
            st.warning("No target players identified. Review your retention and budget strategy.")
            return

        st.dataframe(auction_strategy)

if __name__ == "__main__":
    main()

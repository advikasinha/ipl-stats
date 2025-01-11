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
        self.retention_budget = 42.0  # max for retention
        self.min_players = 18
        self.max_players = 25
        self.max_overseas = 8
        self.playing_xi_overseas = 4
        self.current_budget = self.total_budget

    def calculate_player_score(self, row: pd.Series) -> float:
        """Enhanced player scoring incorporating more metrics"""
        # Batting metrics
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
        
        # Bowling metrics
        bowling_score = 0
        if row['InningsBowled'] > 0:
            bowling_score = (
                (300 - row.get('BowlingAVG', 300)) * 0.25 +
                (12 - row['EconomyRate']) * 0.25 +
                row['Wickets'] * 0.2 +
                (row['3s'] * 0.5 + row['5s']) * 0.2 +
                row['Maidens'] * 0.1
            )
        
        # Fielding metrics
        fielding_score = (row['CatchesTaken'] + row['StumpingsMade']) * 0.5
        
        # Experience metrics
        experience_factor = min(1.2, 1 + (row['IPL Matches'] / 100) * 0.2)
        
        # Normalize scores
        batting_score = np.clip(batting_score / 100, 0, 1)
        bowling_score = np.clip(bowling_score / 100, 0, 1)
        fielding_score = np.clip(fielding_score / 20, 0, 1)
        
        # Calculate final score based on player type
        if row['Type'] == 'Batsman':
            final_score = (batting_score * 0.8 + bowling_score * 0.05 + fielding_score * 0.15) * experience_factor
        elif row['Type'] == 'Bowler':
            final_score = (batting_score * 0.05 + bowling_score * 0.8 + fielding_score * 0.15) * experience_factor
        elif row['Type'] == 'Wicket-Keeper':
            final_score = (batting_score * 0.6 + fielding_score * 0.4) * experience_factor
        else:  # All-rounder
            final_score = (batting_score * 0.4 + bowling_score * 0.4 + fielding_score * 0.2) * experience_factor
            
        return final_score

    def suggest_retentions(self, current_team: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retention strategy for top Indian batsmen and overseas players.
        """
        team_players = self.data[self.data['Team'] == current_team].copy()
        team_players['performance_score'] = team_players.apply(self.calculate_player_score, axis=1)
        
        # Filter Indian batsmen and overseas players
        indian_players = team_players[(team_players['National Side'] == 'India')]
        overseas_players = team_players[team_players['National Side'] != 'India']
        
        # Select top players
        top_indian_players= indian_players.nlargest(5, 'performance_score')[['Name', 'Type', 'performance_score','National Side', 'Team']]
        top_overseas_players = overseas_players.nlargest(3, 'performance_score')[['Name', 'Type', 'performance_score',"National Side", 'Team']]
        
        return top_indian_players, top_overseas_players

    def categorize_priority(self, threshold_high: float = 0.7, threshold_medium: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Categorize players into high and medium priority based on performance thresholds.
        """
        self.data['performance_score'] = self.data.apply(self.calculate_player_score, axis=1)
        
        # High-priority players
        high_priority = self.data[self.data['performance_score'] >= threshold_high][['Name', 'Type', 'performance_score', 'Team']]
        
        # Medium-priority players
        medium_priority = self.data[(self.data['performance_score'] >= threshold_medium) &
                                     (self.data['performance_score'] < threshold_high)][['Name', 'Type', 'performance_score', 'Team']]
        
        return high_priority.sort_values('performance_score', ascending=False), medium_priority.sort_values('performance_score', ascending=False)

    def update_budget(self, selected_players: pd.DataFrame):
        """Update the budget based on selected players."""
        total_cost = selected_players['estimated_price'].sum()
        self.current_budget = self.total_budget - total_cost

    def get_player_replacements(self, player: pd.Series) -> pd.DataFrame:
        """Find replacement players for a given player."""
        available_players = self.data[self.data['Type'] == player['Type']].copy()
        available_players['similarity_score'] = available_players.apply(
            lambda x: 1 - abs(player['performance_score'] - x['performance_score']), axis=1
        )
        
        return available_players.nlargest(5, 'similarity_score')[['Name', 'Type', 'performance_score', 'similarity_score', 'Team']]

# Main Application
if __name__ == "__main__":
    st.title("Enhanced IPL Franchise Planning System")

    uploaded_file = st.file_uploader("Upload IPL dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        planner = EnhancedIPLTeamPlanner(data)

        # Franchise Selection
        franchise = st.selectbox("Select Your Franchise", sorted(data['Team'].unique()))

        # Retention Suggestions
        st.header("Retention Suggestions")
        top_indian_batsmen, top_overseas_players = planner.suggest_retentions(franchise)
        
        st.subheader("Top Indian Batsmen")
        st.dataframe(top_indian_batsmen)
        
        st.subheader("Top Overseas Players")
        st.dataframe(top_overseas_players)

        # High and Medium Priority Players
        st.header("Target Players")
        high_priority, medium_priority = planner.categorize_priority()

        tabs = st.tabs(["High Priority", "Medium Priority"])
        with tabs[0]:
            st.subheader("High Priority Players")
            st.dataframe(high_priority)
        with tabs[1]:
            st.subheader("Medium Priority Players")
            st.dataframe(medium_priority)

        # Replacement Analysis
        st.header("Player Replacement Analysis")
        selected_player = st.selectbox("Select Player to Find Replacements", data['Name'])

        if selected_player:
            player_data = data[data['Name'] == selected_player].iloc[0]
            replacements = planner.get_player_replacements(player_data)
            st.subheader(f"Replacements for {selected_player}")
            st.dataframe(replacements)

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
        self.current_retention_budget = self.retention_budget

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
        team_players['estimated_price'] = (team_players['performance_score'] * self.retention_budget / 4).round(2)
        team_players['estimated_price'] = team_players['estimated_price'].clip(upper=self.retention_budget / 2)
        
        # Filter Indian batsmen and overseas players
        indian_players = team_players[(team_players['National Side'] == 'India')]
        overseas_players = team_players[team_players['National Side'] != 'India']
        
        # Select top players
        top_indian_players= indian_players.nlargest(5, 'performance_score')[['Name', 'Type', 'performance_score','National Side', 'Team', 'estimated_price']]
        top_overseas_players = overseas_players.nlargest(3, 'performance_score')[['Name', 'Type', 'performance_score',"National Side", 'Team', 'estimated_price']]
        
        return top_indian_players, top_overseas_players
    
    def calculate_similarity_score(self, player1: pd.Series, player2: pd.Series) -> float:
        """Calculate similarity between two players based on performance, batting and Bowling"""
        # Performance similarity (50% weight)
        perf_similarity = 1 - abs(player1['performance_score'] - player2['performance_score'])
        
        # Batting style similarity (25% weight)
        batting_similarity = 1.0 if player1['Batting Style'] == player2['Batting Style'] else 0.0
        
        # Bowling similarity (25% weight)
        bowling_similarity = 1.0 if player1['Bowling'] == player2['Bowling'] else 0.0
        
        # Calculate weighted similarity
        total_similarity = (perf_similarity * 0.5 + 
                          batting_similarity * 0.25 + 
                          bowling_similarity * 0.25)
        
        return total_similarity
    
    def categorize_priority(self, threshold_high: float = 0.7, threshold_medium: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Categorize players into high and medium priority based on performance thresholds."""
            self.data['performance_score'] = self.data.apply(self.calculate_player_score, axis=1)
            self.data['estimated_price'] = (self.data['performance_score'] * self.retention_budget / 4).round(2)
            self.data['estimated_price'] = self.data['estimated_price'].clip(upper=self.retention_budget / 2)
            
            # Select relevant columns for display
            display_columns = [
                'Name', 'Type', 'performance_score', 'Team', 'estimated_price',
                'BattingS/R', 'Wickets', 'BattingAVG', 'BowlingAVG', 'EconomyRate',
                'Batting Style', 'Bowling'
            ]
            
            # High-priority players
            high_priority = self.data[self.data['performance_score'] >= threshold_high][display_columns].copy()
            
            # Medium-priority players
            medium_priority = self.data[
                (self.data['performance_score'] >= threshold_medium) & 
                (self.data['performance_score'] < threshold_high)
            ][display_columns].copy()
            
            # Add selection column
            high_priority['Select'] = False
            medium_priority['Select'] = False
            
            return high_priority.sort_values('performance_score', ascending=False), medium_priority.sort_values('performance_score', ascending=False)

    def update_budget(self, selected_players: pd.DataFrame):
        """Update the budget based on selected players."""
        total_cost = selected_players['estimated_price'].sum()
        self.current_budget = self.total_budget - total_cost

    def get_player_replacements(self, player: pd.Series) -> pd.DataFrame:
        """Find replacement players for a given player based on enhanced similarity."""
        available_players = self.data[self.data['Type'] == player['Type']].copy()
        available_players['performance_score'] = available_players.apply(self.calculate_player_score, axis=1)
        available_players['similarity_score'] = available_players.apply(
            lambda x: self.calculate_similarity_score(player, x), axis=1
        )
        
        display_columns = [
            'Name', 'Type', 'performance_score', 'similarity_score', 'Team',
            'BattingS/R', 'Wickets', 'BattingAVG', 'BowlingAVG', 'EconomyRate',
            'Batting Style', 'Bowling'
        ]
        
        return available_players.nlargest(5, 'similarity_score')[display_columns]
    
# Main Application
if __name__ == "__main__":
    # st.set_page_config(layout='wide')
    st.title("Enhanced IPL Franchise Planning System")

    if 'current_budget' not in st.session_state:
        st.session_state.current_budget = 90.0
    if 'current_retention_budget' not in st.session_state:
        st.session_state.current_retention_budget = 42.0

    uploaded_file = st.file_uploader("Upload IPL dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        planner = EnhancedIPLTeamPlanner(data)

        # Sidebar for budget tracking
        st.sidebar.header("Budget Tracker")
        st.sidebar.metric("Total Budget Remaining", 
                         f"₹{st.session_state.current_budget:.2f}Cr",
                         f"₹{90.0 - st.session_state.current_budget:.2f}Cr used")
        st.sidebar.metric("Retention Budget Remaining", 
                         f"₹{st.session_state.current_retention_budget:.2f}Cr",
                         f"₹{42.0 - st.session_state.current_retention_budget:.2f}Cr used")

        # Franchise Selection
        franchise = st.selectbox("Select Your Franchise", sorted(data['Team'].unique()))
        
        # Team Analysis Dashboard
        st.header("Current Team Analysis:")
        team_data = data[data['Team'] == franchise]
        
        # Team composition visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Player type distribution
            fig_composition = px.pie(
                team_data,
                names='Type',
                title='Team Composition',
                color_discrete_sequence=px.colors.cyclical.Edge
            )
            st.plotly_chart(fig_composition)
            
        with col2:
            # Experience level distribution
            team_data['Experience_Level'] = pd.cut(
                team_data['IPL Matches'],
                bins=[-1, 10, 30, 60, float('inf')],
                labels=['Rookie', 'Developing', 'Experienced', 'Veteran']
            )
            fig_experience = px.pie(
                team_data,
                names='Experience_Level',
                title='Team Experience Distribution',
                color_discrete_sequence=px.colors.cyclical.Edge
            )
            st.plotly_chart(fig_experience)

        # Retention Suggestions
        st.header("Retention Suggestions")
        top_indian_batsmen, top_overseas_players = planner.suggest_retentions(franchise)
        
        cb_in=[False, False, False, False, False]
        st.subheader(f"Top Indian Players for {franchise}")
        ret_ind_df=pd.DataFrame(top_indian_batsmen)
        ret_ind_df['Retain']=cb_in
        st.data_editor(ret_ind_df,
        column_config={
        "Retain": st.column_config.CheckboxColumn(
            "Retain?",
            help="Choose players you want to retain",
            default=False,
        ),
        "estimated_price": st.column_config.NumberColumn(
                        "Estimated Price (Cr)",
                        format="₹%.2f"
                    )
                },
                hide_index=True
            )

        cb_intl=[False, False, False]
        st.subheader(f"Top Overseas Players for {franchise}")
        ret_intl_df=pd.DataFrame(top_overseas_players)
        ret_intl_df['Retain']=cb_intl
        st.data_editor(ret_intl_df,
        column_config={
        "Retain": st.column_config.CheckboxColumn(
            "Select",
            help="Choose players you want to retain",
            default=False,
        ),"estimated_price": st.column_config.NumberColumn(
                        "Estimated Price (Cr)",
                        format="₹%.2f"
                    )
                },
                hide_index=True
            )

        # High and Medium Priority Players
        st.header("Target Players")
        high_priority, medium_priority = planner.categorize_priority()

        def update_budgets(df):
            selected_players = df[df['Select']].copy()
            if not selected_players.empty:
                total_cost = selected_players['estimated_price'].sum()
                st.session_state.current_budget -= total_cost
                st.session_state.current_retention_budget -= total_cost
                st.experimental_rerun()

        tabs = st.tabs(["High Priority", "Medium Priority"])
        with tabs[0]:
            st.subheader("High Priority Players")
            edited_high = st.data_editor(
                high_priority,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select player for retention",
                        default=False
                    ),
                    "estimated_price": st.column_config.NumberColumn(
                        "Estimated Price (Cr)",
                        format="₹%.2f"
                    )
                },
                hide_index=True
            )
            if st.button("Update Budget (High Priority)"):
                update_budgets(edited_high)

        with tabs[1]:
            st.subheader("Medium Priority Players")
            edited_medium = st.data_editor(
                medium_priority,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select player for retention",
                        default=False
                    ),
                    "estimated_price": st.column_config.NumberColumn(
                        "Estimated Price (Cr)",
                        format="₹%.2f"
                    )
                },
                hide_index=True
            )
            if st.button("Update Budget (Medium Priority)"):
                update_budgets(edited_medium)

        # Enhanced Replacement Analysis
        st.header("Player Replacement Analysis")
        selected_player = st.selectbox("Select Player to Find Replacements", data['Name'])

        if selected_player:
            player_data = data[data['Name'] == selected_player].iloc[0]
            player_data['performance_score'] = planner.calculate_player_score(player_data)
            replacements = planner.get_player_replacements(player_data)
            st.subheader(f"Potential Replacements for {selected_player}")
            st.dataframe(replacements)


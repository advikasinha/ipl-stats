import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple

class IPLTeamPlanner:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.total_budget = 90.0  # crores
        self.retention_budget = 42.0  # max for 4 players
        self.min_players = 18
        self.max_players = 25
        self.max_overseas = 8
        self.playing_xi_overseas = 4
        
    def calculate_player_score(self, row: pd.Series) -> float:
        """Calculate comprehensive player score based on all-round performance"""
        batting_score = (
            row.get('BattingAVG', 0) * 0.2 +
            row.get('BattingS/R', 0) * 0.3 +
            row.get('Boundaries', 0) * 0.1 +
            row.get('RunsScored', 0) * 0.4
        ) if row.get('BattingAVG', 0) > 0 else 0
        
        bowling_score = (
            (300 - row.get('BowlingAvg', 300)) * 0.3 +
            (12 - row.get('EconomyRate', 12)) * 0.3 +
            row.get('Wickets', 0) * 0.4
        ) if row.get('Wickets', 0) > 0 else 0
        
        # Normalize scores
        batting_score = np.clip(batting_score / 100, 0, 1)
        bowling_score = np.clip(bowling_score / 100, 0, 1)
        
        # Calculate final score based on player type
        if row['Type'] == 'Batsman':
            return batting_score * 0.9 + bowling_score * 0.1
        elif row['Type'] == 'Bowler':
            return batting_score * 0.1 + bowling_score * 0.9
        else:  # All-rounder or Wicket-keeper
            return batting_score * 0.6 + bowling_score * 0.4
            
    def suggest_retentions(self, current_team: str) -> pd.DataFrame:
        """Suggest optimal retention strategy"""
        team_players = self.data[self.data['Team'] == current_team].copy()
        team_players['performance_score'] = team_players.apply(self.calculate_player_score, axis=1)
        
        # Separate Indian and overseas players
        indian_players = team_players[team_players['National Side'] == 'India']
        overseas_players = team_players[team_players['National Side'] != 'India']
        
        # Get top 3 Indian players and top 1 overseas player
        top_indian = indian_players.nlargest(3, 'performance_score')
        top_overseas = overseas_players.nlargest(1, 'performance_score')
        
        retention_suggestions = pd.concat([top_indian, top_overseas])
        
        # Calculate retention costs
        retention_costs = [15, 11, 7, 4]  # crores, in order of retention
        retention_suggestions['retention_cost'] = retention_costs[:len(retention_suggestions)]
        
        return retention_suggestions
    
    def get_player_replacements(self, player: pd.Series, excluded_players: List[str]) -> pd.DataFrame:
        """Find similar players as replacement options"""
        available_players = self.data[~self.data['Name'].isin(excluded_players)].copy()
        
        # Calculate similarity scores
        available_players['similarity_score'] = available_players.apply(
            lambda x: self._calculate_similarity(player, x), axis=1
        )
        
        return available_players.nlargest(5, 'similarity_score')
    
    def _calculate_similarity(self, player1: pd.Series, player2: pd.Series) -> float:
        """Calculate similarity between two players"""
        if player1['Type'] != player2['Type']:
            return 0
            
        metrics = ['BattingAVG', 'BattingS/R', 'Wickets', 'EconomyRate']
        similarity = 0
        
        for metric in metrics:
            if metric in player1 and metric in player2:
                val1 = player1[metric]
                val2 = player2[metric]
                if pd.notnull(val1) and pd.notnull(val2):
                    similarity += 1 - abs(val1 - val2) / max(val1, val2)
                    
        return similarity / len(metrics)
    
    def generate_auction_strategy(self, retained_players: pd.DataFrame) -> pd.DataFrame:
        """Generate auction strategy based on team needs"""
        remaining_budget = self.total_budget - retained_players['retention_cost'].sum()
        required_roles = {
            'Batsman': (5, 7),
            'Bowler': (5, 7),
            'All-Rounder': (3, 5),
            'Wicket-Keeper': (2, 3)
        }
        
        # Adjust required roles based on retentions
        for _, player in retained_players.iterrows():
            role = player['Type']
            if role in required_roles:
                required_roles[role] = (
                    max(0, required_roles[role][0] - 1),
                    max(0, required_roles[role][1] - 1)
                )
        
        # Get available players
        available_players = self.data[~self.data['Name'].isin(retained_players['Name'])].copy()
        available_players['performance_score'] = available_players.apply(
            self.calculate_player_score, axis=1
        )
        
        # Calculate suggested bid amounts
        available_players['suggested_bid'] = available_players['performance_score'] * (
            remaining_budget / (self.min_players - len(retained_players))
        )
        
        # Prioritize players based on team needs
        auction_strategy = []
        
        for role, (min_needed, max_needed) in required_roles.items():
            if min_needed > 0:
                role_players = available_players[
                    available_players['Type'] == role
                ].nlargest(max_needed * 2, 'performance_score')
                
                for _, player in role_players.iterrows():
                    auction_strategy.append({
                        'Name': player['Name'],
                        'Type': player['Type'],
                        'National Side': player['National Side'],
                        'Performance_Score': player['performance_score'],
                        'Suggested_Bid': player['suggested_bid'],
                        'Priority': 'High' if len(auction_strategy) < min_needed else 'Medium'
                    })
        
        return pd.DataFrame(auction_strategy)

def main():
    st.title("IPL Franchise Planning System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload IPL 2022 dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        planner = IPLTeamPlanner(data)
        
        # Franchise selection
        franchise = st.selectbox(
            "Select Your Franchise",
            sorted(data['Team'].unique())
        )
        
        # Show retention suggestions
        st.header("Retention Strategy")
        retention_suggestions = planner.suggest_retentions(franchise)
        
        # Display retention suggestions with metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Suggested Retentions")
            st.dataframe(
                retention_suggestions[['Name', 'Type', 'National Side', 'retention_cost']]
            )
            
        with col2:
            # Retention cost visualization
            fig = px.pie(
                retention_suggestions,
                values='retention_cost',
                names='Name',
                title='Retention Budget Distribution'
            )
            st.plotly_chart(fig)
        
        # Allow user to select retentions
        st.subheader("Select Players to Retain")
        selected_retentions = []
        
        for _, player in retention_suggestions.iterrows():
            if st.checkbox(f"Retain {player['Name']} (₹{player['retention_cost']} cr)"):
                selected_retentions.append(player)
        
        selected_retentions_df = pd.DataFrame(selected_retentions)
        
        if not selected_retentions_df.empty:
            # Generate auction strategy
            st.header("Auction Strategy")
            auction_strategy = planner.generate_auction_strategy(selected_retentions_df)
            
            # Display auction strategy
            st.subheader("Priority Targets")
            
            # Filter by priority
            high_priority = auction_strategy[auction_strategy['Priority'] == 'High']
            medium_priority = auction_strategy[auction_strategy['Priority'] == 'Medium']
            
            st.subheader("High Priority Targets")
            st.dataframe(high_priority)
            
            st.subheader("Backup Options")
            st.dataframe(medium_priority)
            
            # Show replacement options
            st.header("Player Replacements")
            selected_player = st.selectbox(
                "Select player to find replacements for",
                auction_strategy['Name']
            )
            
            if selected_player:
                player_data = auction_strategy[
                    auction_strategy['Name'] == selected_player
                ].iloc[0]
                replacements = planner.get_player_replacements(
                    player_data,
                    auction_strategy['Name'].tolist()
                )
                
                st.subheader(f"Replacement options for {selected_player}")
                st.dataframe(replacements[
                    ['Name', 'Type', 'National Side', 'similarity_score', 'Suggested_Bid']
                ])

if __name__ == "__main__":
    main()
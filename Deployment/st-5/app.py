import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple

class TeamCompositionRules:
    def __init__(self):
        self.composition_rules = {
            'Batsman': {'min': 5, 'max': 7, 'current': 0},
            'Bowler': {'min': 5, 'max': 7, 'current': 0},
            'All-Rounder': {'min': 3, 'max': 5, 'current': 0},
            'Wicket-Keeper': {'min': 2, 'max': 3, 'current': 0}
        }
        self.total_players_range = (18, 25)
        
    def get_needed_roles(self) -> Dict[str, int]:
        return {role: data['min'] - data['current'] 
                for role, data in self.composition_rules.items() 
                if data['current'] < data['min']}
    
    def can_add_role(self, role: str) -> bool:
        return self.composition_rules[role]['current'] < self.composition_rules[role]['max']
    
    def add_player(self, role: str):
        if self.can_add_role(role):
            self.composition_rules[role]['current'] += 1
            return True
        return False

class PricePredictionModel:
    def predict_price(self, player_data: pd.Series) -> float:
        try:
            base_price = 0.5  # Base price in crores
            
            # Role-based adjustments
            role_multipliers = {
                'Batsman': 1.2,
                'Bowler': 1.1,
                'All-Rounder': 1.4,
                'Wicket-Keeper': 1.3
            }
            
            # Performance-based adjustments
            performance_score = float(player_data.get('performance_score', 0))
            performance_multiplier = 1 + (performance_score * 3)
            
            # Experience adjustment
            matches = float(player_data.get('IPL Matches', 0))
            experience_multiplier = min(1.5, 1 + (matches / 100))
            
            # International status adjustment
            international_multiplier = 1.3 if player_data.get('National Side') != 'India' else 1.0
            
            # Calculate final price
            predicted_price = (base_price * 
                             performance_multiplier * 
                             role_multipliers.get(player_data.get('Type', 'Batsman'), 1.0) * 
                             experience_multiplier * 
                             international_multiplier)
            
            return round(predicted_price, 2)
        except Exception as e:
            st.error(f"Error in price prediction: {e}")
            return 0.5  # Return base price if calculation fails

class EnhancedIPLTeamPlanner:
    def __init__(self, data: pd.DataFrame):
        self.data = data.fillna(0)
        self.total_budget = 90.0
        self.retention_budget = 42.0
        self.current_budget = self.total_budget
        self.retained_players = pd.DataFrame()
        self.auctioned_players = pd.DataFrame()
        self.composition_rules = TeamCompositionRules()
        self.price_predictor = PricePredictionModel()

    def calculate_player_score(self, row: pd.Series) -> float:
        try:
            # Batting metrics
            batting_score = 0
            if row['InningsBatted'] > 0:
                batting_score = (
                    float(row['BattingAVG']) * 0.15 +
                    float(row['BattingS/R']) * 0.25 +
                    (float(row['4s']) + float(row['6s'])) * 0.1 +
                    float(row['RunsScored']) * 0.2 +
                    (float(row['50s']) * 0.5 + float(row['100s'])) * 0.2 +
                    (1 - float(row['Ducks'])/max(float(row['InningsBatted']), 1)) * 0.1
                )
            
            # Bowling metrics
            bowling_score = 0
            if row['InningsBowled'] > 0:
                bowling_score = (
                    (300 - float(row.get('BowlingAVG', 300))) * 0.25 +
                    (12 - float(row['EconomyRate'])) * 0.25 +
                    float(row['Wickets']) * 0.2 +
                    (float(row['3s']) * 0.5 + float(row['5s'])) * 0.2 +
                    float(row['Maidens']) * 0.1
                )
            
            # Normalize scores
            batting_score = np.clip(batting_score / 100, 0, 1)
            bowling_score = np.clip(bowling_score / 100, 0, 1)
            
            # Calculate final score based on player type
            if row['Type'] == 'Batsman':
                final_score = batting_score * 0.8 + bowling_score * 0.2
            elif row['Type'] == 'Bowler':
                final_score = batting_score * 0.2 + bowling_score * 0.8
            elif row['Type'] == 'Wicket-Keeper':
                final_score = batting_score
            else:  # All-rounder
                final_score = (batting_score + bowling_score) / 2
                
            return float(final_score)
        except Exception as e:
            st.error(f"Error calculating player score: {e}")
            return 0.0

    def categorize_retention_priority(self, players_df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = players_df.copy()
            
            # Calculate performance scores first
            df['performance_score'] = df.apply(self.calculate_player_score, axis=1)
            
            # Calculate predicted prices
            df['predicted_price'] = df.apply(self.price_predictor.predict_price, axis=1)
            
            # Initialize priority column
            df['priority'] = 'Low'
            
            # High priority conditions
            high_priority_mask = (
                (df['performance_score'] > 0.7) & 
                (df['IPL Matches'] > 30) & 
                (df['predicted_price'] <= self.retention_budget / 3)
            )
            df.loc[high_priority_mask, 'priority'] = 'High'
            
            # Medium priority conditions
            medium_priority_mask = (
                (df['performance_score'] > 0.5) & 
                (df['IPL Matches'] > 15) & 
                (df['predicted_price'] <= self.retention_budget / 2)
            )
            df.loc[medium_priority_mask & ~high_priority_mask, 'priority'] = 'Medium'
            
            return df[['Name', 'Type', 'performance_score', 'predicted_price', 'priority', 'IPL Matches', 'National Side']]
        except Exception as e:
            st.error(f"Error in categorizing retention priority: {e}")
            return pd.DataFrame()

    def suggest_retentions(self, current_team: str) -> Dict[str, pd.DataFrame]:
        try:
            team_players = self.data[self.data['Team'] == current_team].copy()
            
            # Separate Indian and overseas players
            indian_players = team_players[team_players['National Side'] == 'India']
            overseas_players = team_players[team_players['National Side'] != 'India']
            
            # Categorize players by priority
            indian_recommendations = self.categorize_retention_priority(indian_players)
            overseas_recommendations = self.categorize_retention_priority(overseas_players)
            
            return {
                'indian_high': indian_recommendations[indian_recommendations['priority'] == 'High'],
                'indian_medium': indian_recommendations[indian_recommendations['priority'] == 'Medium'],
                'overseas_high': overseas_recommendations[overseas_recommendations['priority'] == 'High'],
                'overseas_medium': overseas_recommendations[overseas_recommendations['priority'] == 'Medium']
            }
        except Exception as e:
            st.error(f"Error in suggesting retentions: {e}")
            return {'indian_high': pd.DataFrame(), 'indian_medium': pd.DataFrame(),
                   'overseas_high': pd.DataFrame(), 'overseas_medium': pd.DataFrame()}

def main():
    st.title("Enhanced IPL Franchise Planning System")

    uploaded_file = st.file_uploader("Upload IPL dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        planner = EnhancedIPLTeamPlanner(data)
        
        # Franchise Selection
        franchise = st.selectbox("Select Your Franchise", sorted(data['Team'].unique()))
        
        # Display current budget
        st.sidebar.header("Budget Overview")
        st.sidebar.metric("Total Budget", f"₹{planner.total_budget}Cr")
        st.sidebar.metric("Available Budget", f"₹{planner.current_budget}Cr")
        
        # Display required team composition
        st.sidebar.header("Required Team Composition")
        for role, rules in planner.composition_rules.composition_rules.items():
            st.sidebar.write(f"{role}: {rules['min']}-{rules['max']} players")
        
        # Retention Recommendations
        st.header("Retention Recommendations")
        retention_suggestions = planner.suggest_retentions(franchise)
        
        cols = st.columns(2)
        
        with cols[0]:
            st.subheader("Indian Players")
            
            st.write("High Priority Targets")
            high_priority_indian = retention_suggestions['indian_high']
            if not high_priority_indian.empty:
                st.dataframe(high_priority_indian)
                selected_indian_high = st.multiselect(
                    "Select High Priority Indian Players",
                    high_priority_indian['Name'].tolist()
                )
                
            st.write("Medium Priority Targets")
            medium_priority_indian = retention_suggestions['indian_medium']
            if not medium_priority_indian.empty:
                st.dataframe(medium_priority_indian)
                selected_indian_medium = st.multiselect(
                    "Select Medium Priority Indian Players",
                    medium_priority_indian['Name'].tolist()
                )
        
        with cols[1]:
            st.subheader("Overseas Players")
            
            st.write("High Priority Targets")
            high_priority_overseas = retention_suggestions['overseas_high']
            if not high_priority_overseas.empty:
                st.dataframe(high_priority_overseas)
                selected_overseas_high = st.multiselect(
                    "Select High Priority Overseas Players",
                    high_priority_overseas['Name'].tolist()
                )
                
            st.write("Medium Priority Targets")
            medium_priority_overseas = retention_suggestions['overseas_medium']
            if not medium_priority_overseas.empty:
                st.dataframe(medium_priority_overseas)
                selected_overseas_medium = st.multiselect(
                    "Select Medium Priority Overseas Players",
                    medium_priority_overseas['Name'].tolist()
                )

        # Display needed roles based on current composition
        st.header("Required Roles")
        needed_roles = planner.composition_rules.get_needed_roles()
        for role, count in needed_roles.items():
            st.write(f"Need {count} more {role}(s)")

if __name__ == "__main__":
    main()
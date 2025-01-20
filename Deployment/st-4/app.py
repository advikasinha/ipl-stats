import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import os
from PIL import Image

def set_custom_style():
    """Set custom CSS styles for the app"""
    st.set_page_config(
        page_title="IPL Team Planner",
        page_icon="🏏",
        layout="wide"
    )
    
def load_team_logos():
    """Load team logos from Deployment/assets directory"""
    team_logos = {
        'MI': 'Deployment/st-4/assets/team_logos/mi.png',
        'CSK': 'Deployment/st-4/assets/team_logos/csk.png',
        'RCB': 'Deployment/st-4/assets/team_logos/rcb.png',
        'KKR': 'Deployment/st-4/assets/team_logos/kkr.png',
        'DC': 'Deployment/st-4/assets/team_logos/dc.png',
        'PBKS': 'Deployment/st-4/assets/team_logos/pbks.png',
        'RR': 'Deployment/st-4/assets/team_logos/rr.png',
        'SRH': 'Deployment/st-4/assets/team_logos/srh.png',
        'GT': 'Deployment/st-4/assets/team_logos/gt.png',
        'LSG': 'Deployment/st-4/assets/team_logos/lsg.png'
    }
    return team_logos

def display_banner():
    """Display IPL banner from local Deployment/assets"""
    banner_path = 'Deployment/st-4/assets/ipl-banner.jpg'
    if os.path.exists(banner_path):
        banner_img = Image.open(banner_path)
        st.image(banner_img, use_container_width=True, caption="")
    else:
        st.error(f"Banner image not found at {banner_path}")

def display_team_grid(data: pd.DataFrame, team_logos: dict):
    """Display all teams in a grid layout for selection"""
    st.markdown("### Select Your Team")
    
    # Create 2 rows of 5 teams each
    teams = sorted(data['Team'].unique())
    cols = st.columns(5)
    
    for idx, team in enumerate(teams):
        # Calculate which row and column this team should be in
        col_index = idx % 5
        
        with cols[col_index]:
            logo_path = team_logos.get(team)
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                # Create clickable team logo
                st.markdown(f"""
                    <div class="team-grid-item">
                        <center>
                """, unsafe_allow_html=True)
                if st.button(
                    label="",
                    key=f"team_{team}",
                    help=f"Select {team}",
                    use_container_width=True
                ):
                    st.session_state.selected_team = team
                    st.session_state.show_analysis = True
                st.image(logo_img, caption=team, use_container_width=True)
                st.markdown("</center></div>", unsafe_allow_html=True)

def show_team_analysis(data: pd.DataFrame, planner: 'EnhancedIPLTeamPlanner', franchise: str):
    """Display all analysis sections for selected team"""
    # Sidebar metrics
    with st.sidebar:
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        st.header("💰 Budget Tracker")
        #Sidebar "Update Budget" button
        if st.button("Update Budget", key="update_budget"):
        #Logic to update budget after selecting players
            total_spent = sum(player['price'] for player in st.session_state.selected_players)
            retention_spent = sum(
                player['price'] for player in st.session_state.selected_players if player.get('is_retention')
                    )
            st.session_state.current_budget = 90.0 - total_spent
            st.session_state.current_retention_budget = 42.0 - retention_spent
            st.success("Budget updated!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Budget",
                f"₹{st.session_state.current_budget:.2f}Cr",
                f"₹{90.0 - st.session_state.current_budget:.2f}Cr used",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Retention Budget",
                f"₹{st.session_state.current_retention_budget:.2f}Cr",
                f"₹{42.0 - st.session_state.current_retention_budget:.2f}Cr used",
                delta_color="inverse"
            )

        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        st.header("🏏 Team Composition")
        for player_type in st.session_state.player_counts:
            current = st.session_state.player_counts[player_type]
            maximum = st.session_state.max_players_by_type[player_type]
            st.progress(current / maximum, text=f"{player_type}: {current}/{maximum}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Selected Players Display
    st.sidebar.header("Selected Players")
    if st.session_state.selected_players:
        selected_df = pd.DataFrame(st.session_state.selected_players)
        st.sidebar.dataframe(selected_df, hide_index=True)

    # Team Analysis Dashboard
    st.markdown('<div class="stHeader">', unsafe_allow_html=True)
    st.header(f"📊 {franchise} Team Analysis")
    st.markdown('</div>', unsafe_allow_html=True)
    
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
        st.plotly_chart(fig_composition, use_container_width=True)
    
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
        st.plotly_chart(fig_experience, use_container_width=True)

    # Retention Suggestions
    st.markdown('<div class="stHeader">', unsafe_allow_html=True)
    st.header("🌟 Retention Suggestions")
    st.markdown('</div>', unsafe_allow_html=True)
    
    top_indian_players, top_overseas_players = planner.suggest_retentions(franchise)
    
    if 'Select' not in top_indian_players.columns:
        top_indian_players['Select'] = False
    
    if 'Select' not in top_overseas_players.columns:
        top_overseas_players['Select'] = False

    st.subheader("Top Indian Players")
    edited_indian = st.data_editor(
        top_indian_players,
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

    st.subheader("Top Overseas Players")
    edited_overseas = st.data_editor(
        top_overseas_players,
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

    # High and Medium Priority Players
    st.markdown('<div class="stHeader">', unsafe_allow_html=True)
    st.header("🎯 Target Players")
    st.markdown('</div>', unsafe_allow_html=True)
    
    high_priority, medium_priority = planner.categorize_priority()
    
    tabs = st.tabs(["High Priority", "Medium Priority"])
    with tabs[0]:
        st.subheader("High Priority Players")
        edited_high = st.data_editor(
            high_priority,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select player",
                    default=False
                ),
                "estimated_price": st.column_config.NumberColumn(
                    "Estimated Price (Cr)",
                    format="₹%.2f"
                )
            },
            hide_index=True
        )
    
    with tabs[1]:
        st.subheader("Medium Priority Players")
        edited_medium = st.data_editor(
            medium_priority,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select player",
                    default=False
                ),
                "estimated_price": st.column_config.NumberColumn(
                    "Estimated Price (Cr)",
                    format="₹%.2f"
                )
            },
            hide_index=True
        )

    # Enhanced Replacement Analysis
    st.markdown('<div class="stHeader">', unsafe_allow_html=True)
    st.header("🔄 Player Replacement Analysis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    selected_player = st.selectbox("Select Player to Find Replacements", data['Name'])
    
    if selected_player:
        player_data = data[data['Name'] == selected_player].iloc[0]
        replacements = planner.get_player_replacements(player_data)
        st.subheader(f"Potential Replacements for {selected_player}")
        st.dataframe(replacements)

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

        #initialize session state
        if 'current_budget' not in st.session_state:
            st.session_state.current_budget = self.total_budget
        if 'current_retention_budget' not in st.session_state:
            st.session_state.current_retention_budget = self.retention_budget
        if 'selected_players' not in st.session_state:
            st.session_state.selected_players = []
        if 'player_counts' not in st.session_state:
            st.session_state.player_counts = {
                 'Batsman': 0,
                 'Bowler': 0,
                'Wicket-Keeper': 0,
                'All-rounder': 0
         }
        if 'max_players_by_type' not in st.session_state:
            st.session_state.max_players_by_type = {
                'Batsman': 7,
                'Bowler': 7,
                'Wicket-Keeper': 3,
                'All-rounder': 8
            }


    def calculate_player_score(self, row: pd.Series) -> float:
        """Calculate player performance score"""
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
        
        ## Bowling metrics
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
        """Suggest players for retention"""
        team_players = self.data[self.data['Team'] == current_team].copy()
        team_players['performance_score'] = team_players.apply(self.calculate_player_score, axis=1)
        team_players['estimated_price'] = (team_players['performance_score'] * self.retention_budget / 4).round(2)
        team_players['estimated_price'] = team_players['estimated_price'].clip(upper=self.retention_budget / 2)
        
        # Filter Indian and overseas players
        indian_players = team_players[team_players['National Side'] == 'India']
        overseas_players = team_players[team_players['National Side'] != 'India']
        
        # Select top players
        top_indian_players = indian_players.nlargest(5, 'performance_score')[
            ['Name', 'Type', 'performance_score', 'National Side', 'Team', 'estimated_price']
        ]
        top_overseas_players = overseas_players.nlargest(3, 'performance_score')[
            ['Name', 'Type', 'performance_score', 'National Side', 'Team', 'estimated_price']
        ]
        
        return top_indian_players, top_overseas_players

    def clean_player_type(self, player_type: str) -> str:
        """Standardize player type strings"""
        return player_type.strip()

    def can_add_player_type(self, player_type: str) -> bool:
        """Check if another player of this type can be added"""
        clean_type = self.clean_player_type(player_type)
        return st.session_state.player_counts[clean_type] < st.session_state.max_players_by_type[clean_type]

    def update_team_composition(self, player: dict, adding: bool = True):
        """Update team composition counters"""
        player_type = self.clean_player_type(player['Type'])
        if adding:
            if not self.can_add_player_type(player_type):
                st.error(f"Maximum number of {player_type}s reached!")
                return False
            st.session_state.player_counts[player_type] += 1
        else:
            st.session_state.player_counts[player_type] -= 1
        return True

    def update_budgets_and_composition(self, player: dict, is_retention: bool = False):
        """Update budgets and team composition"""
        if player['Select']:  # Player is being selected
            if not self.can_add_player_type(player['Type']):
                return False
                
            if is_retention:
                if st.session_state.current_retention_budget >= player['estimated_price']:
                    st.session_state.current_retention_budget -= player['estimated_price']
                else:
                    st.error(f"Not enough retention budget for {player['Name']}")
                    return False
            
            if st.session_state.current_budget >= player['estimated_price']:
                if self.update_team_composition(player, adding=True):
                    st.session_state.current_budget -= player['estimated_price']
                    st.session_state.selected_players.append({
                        'name': player['Name'],
                        'type': player['Type'],
                        'price': player['estimated_price'],
                        'is_retention': is_retention
                    })
                    # st.experimental_rerun()
                    return True
            else:
                st.error(f"Not enough budget for {player['Name']}")
                return False
        else:  # Player is being deselected
            for i, selected in enumerate(st.session_state.selected_players):
                if selected['name'] == player['Name']:
                    if selected['is_retention']:
                        st.session_state.current_retention_budget += player['estimated_price']
                    st.session_state.current_budget += player['estimated_price']
                    self.update_team_composition(selected, adding=False)
                    st.session_state.selected_players.pop(i)
                    # st.experimental_rerun()
                    break
            return True

    def categorize_priority(self, threshold_high: float = 0.7, threshold_medium: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Categorize players into priority levels"""
        self.data['performance_score'] = self.data.apply(self.calculate_player_score, axis=1)
        self.data['estimated_price'] = (self.data['performance_score'] * self.retention_budget / 4).round(2)
        self.data['estimated_price'] = self.data['estimated_price'].clip(upper=self.retention_budget / 2)
        
        display_columns = [
            'Name', 'Type', 'performance_score', 'Team', 'estimated_price',
            'BattingS/R', 'Wickets', 'BattingAVG', 'BowlingAVG', 'EconomyRate',
            'Batting Style', 'Bowling'
        ]
        
        high_priority = self.data[self.data['performance_score'] >= threshold_high][display_columns].copy()
        medium_priority = self.data[
            (self.data['performance_score'] >= threshold_medium) & 
            (self.data['performance_score'] < threshold_high)
        ][display_columns].copy()
        
        high_priority['Select'] = False
        medium_priority['Select'] = False
        
        return high_priority.sort_values('performance_score', ascending=False), medium_priority.sort_values('performance_score', ascending=False)

    def calculate_similarity_score(self, player1: pd.Series, player2: pd.Series) -> float:
        """Calculate similarity between two players"""
        perf_similarity = 1 - abs(player1['performance_score'] - player2['performance_score'])
        batting_similarity = 1.0 if player1['Batting Style'] == player2['Batting Style'] else 0.0
        bowling_similarity = 1.0 if player1['Bowling'] == player2['Bowling'] else 0.0
        
        total_similarity = (perf_similarity * 0.5 + 
                          batting_similarity * 0.25 + 
                          bowling_similarity * 0.25)
        
        return total_similarity

    def get_player_replacements(self, player: pd.Series) -> pd.DataFrame:
        """Find replacement players"""
        available_players = self.data[self.data['Type'] == player['Type']].copy()
        available_players['performance_score'] = available_players.apply(self.calculate_player_score, axis=1)
        available_players['similarity_score'] = available_players.apply(
            lambda x: self.calculate_similarity_score(player, x), axis=1
        )
        
        display_columns = [
            'Name', 'Type', 'Batting Style', 'Bowling', 'performance_score', 'Team', 'National Side',
            'BattingS/R', 'Wickets', 'BattingAVG', 'BowlingAVG', 'EconomyRate',
        ]
        
        return available_players.nlargest(5, 'similarity_score')[display_columns]

# Main Application
if __name__ == "__main__":
    set_custom_style()    
    st.title("_IPL Franchise Planning System_")
    display_banner()

    # Initialize session state for team selection
    if 'selected_team' not in st.session_state:
        st.session_state.selected_team = None
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False

    uploaded_file = st.file_uploader("Upload IPL dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        planner = EnhancedIPLTeamPlanner(data)

        # If no team is selected, show the team selection grid
        if not st.session_state.show_analysis or st.session_state.selected_team is None:
            team_logos = load_team_logos()
            display_team_grid(data, team_logos)
        else:
            # Show back button
            if st.button("← Back to Team Selection"):
                st.session_state.show_analysis = False
                st.session_state.selected_team = None
                # st.experimental_rerun()
            
            # Show analysis for selected team
            if st.session_state.selected_team:
                show_team_analysis(data, planner, st.session_state.selected_team)

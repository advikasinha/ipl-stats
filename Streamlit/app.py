# app.py
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import openai
from typing import Dict, List
import json
import plotly.express as px

class IPLTeamAnalyzer:
    def __init__(self, data: pd.DataFrame, budget: float = 90.0):
        self.data = data
        self.budget = budget
        self.min_players = 18
        self.max_players = 25
        
    def calculate_player_score(self, row: pd.Series) -> float:
        """Calculate a composite score for a player based on their stats"""
        batting_score = (
            row.get('BattingAvg', 0) * 0.3 +
            row.get('StrikeRate', 0) * 0.3 +
            row.get('RunsScored', 0) * 0.2 +
            row.get('HighestScore', 0) * 0.2
        )
        
        bowling_score = (
            row.get('BowlingAvg', 0) * 0.3 +
            row.get('EconomyRate', 0) * 0.3 +
            row.get('Wickets', 0) * 0.2 +
            row.get('MaidenOvers', 0) * 0.2
        )
        
        return batting_score + bowling_score
    
    def form_optimal_team(self, franchise: str) -> Dict:
        """Form optimal team within budget constraints"""
        available_players = self.data.copy()
        selected_players = []
        remaining_budget = self.budget
        
        # Calculate composite scores
        available_players['composite_score'] = available_players.apply(
            self.calculate_player_score, axis=1
        )
        
        # Ensure minimum requirements
        min_requirements = {
            'Batsman': 6,
            'Bowler': 6,
            'All-Rounder': 3,
            'Wicket-Keeper': 2
        }
        
        for role, count in min_requirements.items():
            role_players = available_players[
                available_players['Type'] == role
            ].nlargest(count, 'composite_score')
            
            for _, player in role_players.iterrows():
                if remaining_budget >= player['ValueInCR']:
                    selected_players.append(player)
                    remaining_budget -= player['ValueInCR']
                    available_players = available_players[
                        available_players['Name'] != player['Name']
                    ]
        
        # Fill remaining slots optimally
        while (
            len(selected_players) < self.max_players and 
            remaining_budget > 0 and 
            not available_players.empty
        ):
            best_player = available_players.nlargest(1, 'composite_score').iloc[0]
            if best_player['ValueInCR'] <= remaining_budget:
                selected_players.append(best_player)
                remaining_budget -= best_player['ValueInCR']
                available_players = available_players[
                    available_players['Name'] != best_player['Name']
                ]
            else:
                break
        
        return {
            'team': pd.DataFrame(selected_players),
            'remaining_budget': remaining_budget,
            'total_players': len(selected_players)
        }

def create_network_graph(data: pd.DataFrame):
    """Create and return network graph data"""
    G = nx.Graph()
    
    # Add player nodes
    for _, player in data.iterrows():
        G.add_node(player['Name'], 
                  type='player',
                  team=player['Team'],
                  value=player.get('ValueInCR', 1))
    
    # Add team nodes and connect players
    teams = data['Team'].unique()
    for team in teams:
        G.add_node(team, type='team')
        team_players = data[data['Team'] == team]['Name'].tolist()
        
        # Connect players to their team
        for player in team_players:
            G.add_edge(team, player, type='team_player')
            
        # Connect players within same team
        for i in range(len(team_players)):
            for j in range(i+1, len(team_players)):
                G.add_edge(team_players[i], team_players[j], 
                          type='teammate')
    
    return G

def main():
    st.title("IPL 2022 Team Analysis Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload IPL 2022 dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Sidebar
        st.sidebar.header("Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Team Network", "Team Formation", "Player Statistics"]
        )
        
        if analysis_type == "Team Network":
            st.header("Team Network Analysis")
            
            # Create network graph
            G = create_network_graph(data)
            
            # Create network visualization using plotly
            pos = nx.spring_layout(G)
            
            edge_trace = go.Scatter(
                x=[], y=[], mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            )
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            node_trace = go.Scatter(
                x=[], y=[], mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                )
            )
            
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                          ))
            
            st.plotly_chart(fig)
            
        elif analysis_type == "Team Formation":
            st.header("Optimal Team Formation")
            
            selected_franchise = st.selectbox(
                "Select Franchise",
                data['Team'].unique()
            )
            
            analyzer = IPLTeamAnalyzer(data)
            result = analyzer.form_optimal_team(selected_franchise)
            
            st.subheader("Selected Team")
            st.dataframe(result['team'])
            
            st.metric(
                "Remaining Budget (Crores)", 
                f"₹{result['remaining_budget']:.2f}"
            )
            
            # Show team composition
            team_composition = result['team']['Type'].value_counts()
            fig = px.pie(
                values=team_composition.values,
                names=team_composition.index,
                title="Team Composition"
            )
            st.plotly_chart(fig)
            
        else:  # Player Statistics
            st.header("Player Statistics")
            
            selected_player = st.selectbox(
                "Select Player",
                data['Name'].unique()
            )
            
            player_data = data[data['Name'] == selected_player].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Team", player_data['Team'])
                st.metric("Type", player_data['Type'])
                st.metric("Value (Crores)", f"₹{player_data['ValueInCR']}")
            
            with col2:
                st.metric("Batting Average", f"{player_data.get('BattingAvg', 0):.2f}")
                st.metric("Strike Rate", f"{player_data.get('StrikeRate', 0):.2f}")
                st.metric("Wickets", player_data.get('Wickets', 0))

if __name__ == "__main__":
    main()
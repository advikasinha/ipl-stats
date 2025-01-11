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
        """Enhanced player scoring incorporating more metrics"""
        # Batting metrics
        batting_score = 0
        if row['InningsBatted'] > 0:
            batting_score = (
                row['BattingAVG'] * 0.15 +  # Added weight to average
                row['BattingS/R'] * 0.25 +  # Strike rate importance
                (row['4s'] + row['6s']) * 0.1 +  # Boundary hitting ability
                row['RunsScored'] * 0.2 +  # Total runs
                (row['50s'] * 0.5 + row['100s']) * 0.2 +  # Impact innings
                (1 - row['Ducks']/max(row['InningsBatted'], 1)) * 0.1  # Consistency
            )
        
        # Bowling metrics
        bowling_score = 0
        if row['InningsBowled'] > 0:
            bowling_score = (
                (300 - row.get('BowlingAVG', 300)) * 0.25 +
                (12 - row['EconomyRate']) * 0.25 +
                row['Wickets'] * 0.2 +
                (row['3s'] * 0.5 + row['5s']) * 0.2 +  # Impact spells
                row['Maidens'] * 0.1  # Control metric
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

    # def suggest_retentions(self, current_team: str) -> pd.DataFrame:
    #     """Enhanced retention strategy considering pre-auction status"""
    #     team_players = self.data[self.data['Team'] == current_team].copy()
        
    #     # Consider pre-auction status
    #     previously_retained = team_players[team_players['Pre-Auction'] == 'Retained']
        
    #     # Calculate performance scores
    #     team_players['performance_score'] = team_players.apply(self.calculate_player_score, axis=1)
        
    #     # Separate Indian and overseas players
    #     indian_players = team_players[team_players['National Side'] == 'India']
    #     overseas_players = team_players[team_players['National Side'] != 'India']
        
    #     # Get top players considering both performance and previous retention status
    #     def get_top_players(players_df: pd.DataFrame, n: int) -> pd.DataFrame:
    #         # Boost score for previously retained players
    #         players_df = players_df.copy()
    #         players_df.loc[players_df['Pre-Auction'] == 'Retained', 'performance_score'] *= 1.2
    #         return players_df.nlargest(n, 'performance_score')
        
    #     top_indian = get_top_players(indian_players, 3)
    #     top_overseas = get_top_players(overseas_players, 1)
        
    #     retention_suggestions = pd.concat([top_indian, top_overseas])
        
    #     # Calculate retention costs
    #     retention_costs = [15, 11, 7, 4]  # crores, in order of retention
    #     retention_suggestions['retention_cost'] = retention_costs[:len(retention_suggestions)]
        
    #     return retention_suggestions
    # def suggest_retentions(self, current_team: str) -> pd.DataFrame:
    #     """
    #     Enhanced retention strategy with flexible retention count based on performance.
    #     Maximum 4 players can be retained, with maximum 1 overseas player.
    #     Retention is not mandatory - only suggests players if their performance justifies it.
    #     """
    #     team_players = self.data[self.data['Team'] == current_team].copy()
        
    #     # Calculate performance scores for all players
    #     team_players['performance_score'] = team_players.apply(self.calculate_player_score, axis=1)
        
    #     # Set performance thresholds for retention
    #     RETENTION_THRESHOLD = 0.6  # Only retain players above this performance score
        
    #     # Separate Indian and overseas players and sort by performance
    #     indian_players = team_players[team_players['National Side'] == 'India'].copy()
    #     overseas_players = team_players[team_players['National Side'] != 'India'].copy()
        
    #     indian_players = indian_players[indian_players['performance_score'] >= RETENTION_THRESHOLD]
    #     overseas_players = overseas_players[overseas_players['performance_score'] >= RETENTION_THRESHOLD]
        
    #     # Sort both by performance score
    #     indian_players = indian_players.sort_values('performance_score', ascending=False)
    #     overseas_players = overseas_players.sort_values('performance_score', ascending=False)
        
    #     # Select retentions based on performance
    #     retentions = []
        
    #     # Add top overseas player if performance is good enough
    #     if not overseas_players.empty and overseas_players.iloc[0]['performance_score'] >= RETENTION_THRESHOLD:
    #         retentions.append(overseas_players.iloc[0])
        
    #     # Add top Indian players (up to 3, only if performance is good enough)
    #     for _, player in indian_players.head(3).iterrows():
    #         if len(retentions) < 4:  # Check total retention limit
    #             retentions.append(player)
        
    #     if not retentions:
    #         return pd.DataFrame()  # Return empty DataFrame if no players meet retention criteria
        
    #     retention_suggestions = pd.DataFrame(retentions)
        
    #     # Sort final suggestions by performance score
    #     retention_suggestions = retention_suggestions.sort_values('performance_score', ascending=False)
        
    #     # Assign retention costs based on number of players retained
    #     retention_costs = [15, 11, 7, 4]  # crores, in order of retention
    #     retention_suggestions['retention_cost'] = retention_costs[:len(retention_suggestions)]
        
    #     # Add a column explaining why each player was selected
    #     retention_suggestions['retention_reasoning'] = retention_suggestions.apply(
    #         lambda x: (
    #             f"Top {'overseas' if x['National Side'] != 'India' else 'Indian'} performer "
    #             f"with performance score of {x['performance_score']:.3f}"
    #         ),
    #         axis=1
    #     )
        
    #     return retention_suggestions
    def suggest_retentions(self, current_team: str) -> pd.DataFrame:
        """
        Retention strategy considering pre-auction status, performance, and flexibility in retention count.
        Maximum 4 players can be retained, with a maximum of 1 overseas player.
        Players are selected based on performance, previous retention status, and overall contribution.
        """
        # Filter players of the given team
        team_players = self.data[self.data['Team'] == current_team].copy()
        
        # Calculate performance scores
        team_players['performance_score'] = team_players.apply(self.calculate_player_score, axis=1)
        
        # Boost score for previously retained players
        team_players.loc[team_players['Pre-Auction'] == 'Retained', 'performance_score'] *= 1.2
        
        # Set performance threshold for retention
        RETENTION_THRESHOLD = 0.6  # Retain only players with performance score above this
        
        # Separate Indian and overseas players
        indian_players = team_players[team_players['National Side'] == 'India'].copy()
        overseas_players = team_players[team_players['National Side'] != 'India'].copy()
        
        # Filter players based on performance threshold
        indian_players = indian_players[indian_players['performance_score'] >= RETENTION_THRESHOLD]
        overseas_players = overseas_players[overseas_players['performance_score'] >= RETENTION_THRESHOLD]
        
        # Sort players by performance score
        indian_players = indian_players.sort_values('performance_score', ascending=False)
        overseas_players = overseas_players.sort_values('performance_score', ascending=False)
        
        # Select retentions
        retentions = []
        
        # Add top overseas player if they meet the threshold
        if not overseas_players.empty:
            top_overseas = overseas_players.iloc[0]
            retentions.append(top_overseas)
        
        # Add top Indian players (up to 3)
        for _, player in indian_players.head(3).iterrows():
            if len(retentions) < 4:  # Check total retention limit
                retentions.append(player)
        
        # Return an empty DataFrame if no players meet retention criteria
        if not retentions:
            return pd.DataFrame()
        
        # Create a DataFrame for selected retentions
        retention_suggestions = pd.DataFrame(retentions)
        
        # Sort final suggestions by performance score
        retention_suggestions = retention_suggestions.sort_values('performance_score', ascending=False)
        
        # Assign retention costs
        retention_costs = [15, 11, 7, 4]  # crores, in order of retention
        retention_suggestions['retention_cost'] = retention_costs[:len(retention_suggestions)]
        
        # Add reasoning for retention
        retention_suggestions['retention_reasoning'] = retention_suggestions.apply(
            lambda x: (
                f"Top {'overseas' if x['National Side'] != 'India' else 'Indian'} performer "
                f"with performance score of {x['performance_score']:.3f} "
                f"{'(previously retained)' if x['Pre-Auction'] == 'Retained' else ''}"
            ),
            axis=1
        )
        
        return retention_suggestions


    def optimize_budget(self, retained_players: pd.DataFrame, target_roles: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """Optimize budget allocation for different player roles"""
        remaining_budget = self.total_budget - retained_players['retention_cost'].sum()
        players_needed = self.min_players - len(retained_players)
        
        # Calculate role-wise requirements and allocate budget
        role_budgets = {}
        total_priority = 0
        
        for role, (min_req, _) in target_roles.items():
            retained_count = len(retained_players[retained_players['Type'] == role])
            still_needed = max(0, min_req - retained_count)
            
            # Priority based on role and current team composition
            priority_weights = {
                'Batsman': 1.2,
                'Bowler': 1.2,
                'All-Rounder': 1.5,
                'Wicket-Keeper': 1.0
            }
            
            priority = still_needed * priority_weights[role]
            total_priority += priority
            role_budgets[role] = {'priority': priority, 'needed': still_needed}
        
        # Allocate budget based on priorities
        for role in role_budgets:
            if total_priority > 0:
                role_budgets[role] = (role_budgets[role]['priority'] / total_priority) * remaining_budget
            else:
                role_budgets[role] = remaining_budget / len(role_budgets)
                
        return role_budgets

    def get_player_replacements(self, player: pd.Series, excluded_players: List[str]) -> pd.DataFrame:
        """Enhanced player replacement suggestions"""
        available_players = self.data[~self.data['Name'].isin(excluded_players)].copy()
        
        # Calculate similarity scores with more metrics
        available_players['similarity_score'] = available_players.apply(
            lambda x: self._calculate_enhanced_similarity(player, x), axis=1
        )
        
        # Add value for money metric
        available_players['value_score'] = available_players.apply(
            lambda x: self.calculate_player_score(x) / max(x['ValueinCR'], 1), axis=1
        )
        
        # Combine similarity and value scores
        available_players['final_score'] = (
            available_players['similarity_score'] * 0.7 +
            available_players['value_score'] * 0.3
        )
        
        return available_players.nlargest(5, 'final_score')

    def _calculate_enhanced_similarity(self, player1: pd.Series, player2: pd.Series) -> float:
        """Calculate enhanced similarity between two players"""
        if player1['Type'] != player2['Type']:
            return 0
            
        # Define metric groups and their weights
        metric_groups = {
            'batting': {
                'metrics': ['BattingAVG', 'BattingS/R', 'RunsScored', '4s', '6s'],
                'weight': 0.4
            },
            'bowling': {
                'metrics': ['BowlingAVG', 'EconomyRate', 'Wickets', 'Maidens'],
                'weight': 0.4
            },
            'experience': {
                'metrics': ['IPL Matches', 'MatchPlayed'],
                'weight': 0.2
            }
        }
        
        total_similarity = 0
        total_weight = 0
        
        for group, config in metric_groups.items():
            group_similarity = 0
            valid_metrics = 0
            
            for metric in config['metrics']:
                if metric in player1 and metric in player2:
                    val1 = player1[metric]
                    val2 = player2[metric]
                    if pd.notnull(val1) and pd.notnull(val2) and val1 != 0:
                        similarity = 1 - abs(val1 - val2) / max(val1, val2)
                        group_similarity += similarity
                        valid_metrics += 1
            
            if valid_metrics > 0:
                total_similarity += (group_similarity / valid_metrics) * config['weight']
                total_weight += config['weight']
        
        return total_similarity / total_weight if total_weight > 0 else 0

    def generate_auction_strategy(self, retained_players: pd.DataFrame) -> pd.DataFrame:
        required_roles = {
        'Batsman': (5, 7),
        'Bowler': (5, 7),
        'All-Rounder': (3, 5),
        'Wicket-Keeper': (2, 3)
        }
    
    # Get role-wise budgets
        role_budgets = self.optimize_budget(retained_players, required_roles)
    
    # Get available players
        available_players = self.data[~self.data['Name'].isin(retained_players['Name'])].copy()
    
    # Calculate performance scores
        available_players['Performance_Score'] = available_players.apply(
            self.calculate_player_score, axis=1
        )
    
    # Calculate value for money
        available_players['Value_Score'] = available_players['Performance_Score'] / available_players['ValueinCR']
    
        auction_strategy_list = []
        
        for role, (min_needed, max_needed) in required_roles.items():
            retained_count = len(retained_players[retained_players['Type'] == role])
            still_needed = max(0, min_needed - retained_count)
            role_budget = role_budgets.get(role, 0)
            
            # Get players for this role
            role_players = available_players[available_players['Type'] == role].copy()
            
            if len(role_players) > 0:
                # Calculate suggested bid amounts
                avg_budget_per_player = role_budget / max(still_needed, 1)
                mean_performance = role_players['Performance_Score'].mean()
                
                for idx, player in role_players.iterrows():
                    suggested_bid = min(
                        avg_budget_per_player * (player['Performance_Score'] / mean_performance),
                        player['ValueinCR'] * 1.5  # Cap at 150% of base price
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
        
        # Ensure all required columns exist
        required_columns = [
            'Name', 'Type', 'National Side', 'Performance_Score', 'Value_Score',
            'Suggested_Bid', 'Base_Price', 'IPL_Experience', 'Priority'
        ]
        for col in required_columns:
            if col not in auction_strategy_df.columns:
                auction_strategy_df[col] = None
        
        return auction_strategy_df
    
    def display_auction_strategy(auction_strategy: pd.DataFrame, st) -> None:
        display_columns = [
            'Name', 'Type', 'National Side', 'Performance_Score',
            'Value_Score', 'Suggested_Bid', 'Base_Price', 'IPL_Experience'
        ]
        
        # Ensure all display columns exist
        for col in display_columns:
            if col not in auction_strategy.columns:
                auction_strategy[col] = None
        
        # Filter by priority
        high_priority = auction_strategy[auction_strategy['Priority'] == 'High'].copy()
        medium_priority = auction_strategy[auction_strategy['Priority'] == 'Medium'].copy()
        
        tabs = st.tabs(["High Priority", "Medium Priority", "All Targets"])
        
        with tabs[0]:
            if not high_priority.empty:
                st.dataframe(high_priority[display_columns])
                
                fig_priority = px.scatter(
                    high_priority,
                    x='Performance_Score',
                    y='Suggested_Bid',
                    size='IPL_Experience',
                    color='Type',
                    hover_data=['Name', 'National Side'],
                    title='High Priority Targets Analysis'
                )
                st.plotly_chart(fig_priority)
            else:
                st.info("No high priority targets available.")
        
        with tabs[1]:
            if not medium_priority.empty:
                st.dataframe(medium_priority[display_columns])
            else:
                st.info("No medium priority targets available.")
        
        with tabs[2]:
            if not auction_strategy.empty:
                fig_targets = px.scatter(
                    auction_strategy,
                    x='Performance_Score',
                    y='Suggested_Bid',
                    size='IPL_Experience',
                    color='Priority',
                    hover_data=['Name', 'Type', 'National Side'],
                    title='All Targets Analysis'
                )
                st.plotly_chart(fig_targets)
            else:
                st.info("No targets available.")


    
    
def main():
    st.title("Enhanced IPL Franchise Planning System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload IPL dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        planner = EnhancedIPLTeamPlanner(data)
        
        # Franchise selection
        franchise = st.selectbox(
            "Select Your Franchise",
            sorted(data['Team'].unique())
        )
        
        # Team Analysis Dashboard
        st.header("Team Analysis Dashboard")
        team_data = data[data['Team'] == franchise]
        
        # Team composition visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Player type distribution
            fig_composition = px.pie(
                team_data,
                names='Type',
                title='Current Team Composition'
            )
            st.plotly_chart(fig_composition)
            
        with col2:
            # Experience level distribution
            team_data['Experience_Level'] = pd.cut(
                team_data['IPL Matches'],
                bins=[0, 20, 50, 100, float('inf')],
                labels=['Rookie', 'Developing', 'Experienced', 'Veteran']
            )
            fig_experience = px.pie(
                team_data,
                names='Experience_Level',
                title='Team Experience Distribution'
            )
            st.plotly_chart(fig_experience)
        
        # Show retention suggestions
        st.header("Retention Strategy")
        retention_suggestions = planner.suggest_retentions(franchise)
        
        # Display retention suggestions with metrics
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Suggested Retentions")
            retention_display = retention_suggestions[[
                'Name', 'Type', 'National Side', 'retention_cost', 'performance_score'
            ]].copy()
            retention_display['performance_score'] = retention_display['performance_score'].round(3)
            st.dataframe(retention_display)
            
        with col4:
            # Retention budget visualization
            fig_budget = go.Figure(go.Waterfall(
                name="Retention Budget",
                orientation="v",
                measure=["relative"] * len(retention_suggestions) + ["total"],
                x=list(retention_suggestions['Name']) + ["Remaining"],
                y=list(-retention_suggestions['retention_cost']) + 
                  [planner.retention_budget - retention_suggestions['retention_cost'].sum()]
            ))
            fig_budget.update_layout(title="Retention Budget Allocation")
            st.plotly_chart(fig_budget)
        
        # Allow user to select retentions
        st.subheader("Select Players to Retain")
        selected_retentions = []
        
        for _, player in retention_suggestions.iterrows():
            if st.checkbox(f"Retain {player['Name']} (₹{player['retention_cost']} cr)"):
                selected_retentions.append(player)
        
        selected_retentions_df = pd.DataFrame(selected_retentions)
        
        if not selected_retentions_df.empty:
            st.header("Auction Strategy")
            auction_strategy = planner.generate_auction_strategy(selected_retentions_df)
            
            # Display budget optimization
            st.subheader("Budget Optimization")
            required_roles = {
                'Batsman': (5, 7),
                'Bowler': (5, 7),
                'All-Rounder': (3, 5),
                'Wicket-Keeper': (2, 3)
            }
            role_budgets = planner.optimize_budget(selected_retentions_df, required_roles)
            
            # Display auction strategy
            if not auction_strategy.empty:
                display_auction_strategy(auction_strategy, st)
        
            # Budget allocation visualization
            budget_data = pd.DataFrame({
                'Role': list(role_budgets.keys()),
                'Budget': list(role_budgets.values())
            })
            
            fig_role_budget = px.bar(
                budget_data,
                x='Role',
                y='Budget',
                title='Recommended Budget Allocation by Role'
            )
            st.plotly_chart(fig_role_budget)
            
            # Display auction strategy with tabs
            st.subheader("Target Players")
            tabs = st.tabs(["High Priority", "Medium Priority", "All Targets"])
            
            with tabs[0]:
                high_priority = auction_strategy[auction_strategy['Priority'] == 'High'].copy()
                high_priority = high_priority.sort_values('Value_Score', ascending=False)
                st.dataframe(high_priority[[
                    'Name', 'Type', 'National Side', 'Performance_Score',
                    'Value_Score', 'Suggested_Bid', 'Base_Price', 'IPL_Experience'
                ]])
                
                # Visualize high priority targets
                if not high_priority.empty:
                    fig_priority = px.scatter(
                        high_priority,
                        x='Performance_Score',
                        y='Suggested_Bid',
                        size='IPL_Experience',
                        color='Type',
                        hover_data=['Name', 'National Side'],
                        title='High Priority Targets Analysis'
                    )
                    st.plotly_chart(fig_priority)
            
            with tabs[1]:
                medium_priority = auction_strategy[auction_strategy['Priority'] == 'Medium'].copy()
                medium_priority = medium_priority.sort_values('Value_Score', ascending=False)
                st.dataframe(medium_priority[[
                    'Name', 'Type', 'National Side', 'Performance_Score',
                    'Value_Score', 'Suggested_Bid', 'Base_Price', 'IPL_Experience'
                ]])
            
            with tabs[2]:
                # Overall target analysis
                fig_targets = px.scatter(
                    auction_strategy,
                    x='Performance_Score',
                    y='Suggested_Bid',
                    size='IPL_Experience',
                    color='Priority',
                    hover_data=['Name', 'Type', 'National Side'],
                    title='All Targets Analysis'
                )
                st.plotly_chart(fig_targets)
            
            # Player replacement analysis
            st.header("Player Replacement Analysis")
            col5, col6 = st.columns([1, 2])
            
            with col5:
                selected_player = st.selectbox(
                    "Select player to find replacements for",
                    auction_strategy['Name']
                )
                
                filtered_players = auction_strategy[auction_strategy['Name'] == selected_player]

                if not filtered_players.empty:
                    selected_type = filtered_players['Type'].iloc[0]
                    st.info(f"Finding replacements for: {selected_player} ({selected_type})")
                else:
                    st.warning(f"No data available for the selected player: {selected_player}")


            if selected_player:
                player_data = auction_strategy[
                    auction_strategy['Name'] == selected_player
                ].iloc[0]
                replacements = planner.get_player_replacements(
                    player_data,
                    auction_strategy['Name'].tolist()
                )
                
                with col6:
                    st.subheader("Replacement Options")
                    replacement_display = replacements[[
                        'Name', 'Type', 'National Side', 'similarity_score',
                        'value_score', 'ValueinCR', 'IPL Matches'
                    ]].copy()
                    replacement_display = replacement_display.round(3)
                    st.dataframe(replacement_display)
                
                # Comparison visualization
                st.subheader("Replacement Comparison")
                selected_metrics = st.multiselect(
                    "Select metrics for comparison",
                    ['BattingAVG', 'BattingS/R', 'RunsScored', 'Wickets', 'EconomyRate', 'IPL Matches'],
                    default=['BattingAVG', 'BattingS/R', 'IPL Matches']
                )
                
                if selected_metrics:
                    comparison_data = pd.concat([
                        player_data[selected_metrics].to_frame().T,
                        replacements[selected_metrics]
                    ])
                    comparison_data['Player'] = [player_data['Name']] + replacements['Name'].tolist()
                    
                    fig_comparison = go.Figure()
                    for metric in selected_metrics:
                        fig_comparison.add_trace(go.Bar(
                            name=metric,
                            x=comparison_data['Player'],
                            y=comparison_data[metric],
                            text=comparison_data[metric].round(2),
                            textposition='auto',
                        ))
                    
                    fig_comparison.update_layout(
                        title="Metric Comparison with Replacement Options",
                        barmode='group'
                    )
                    st.plotly_chart(fig_comparison)
            
            # Team Balance Analysis
            st.header("Projected Team Balance")
            if not selected_retentions_df.empty and not high_priority.empty:
                projected_team = pd.concat([
                    selected_retentions_df[['Name', 'Type', 'National Side']],
                    high_priority[['Name', 'Type', 'National Side']]
                ])
                
                col7, col8 = st.columns(2)
                
                with col7:
                    # Player type distribution
                    fig_projected = px.pie(
                        projected_team,
                        names='Type',
                        title='Projected Team Composition'
                    )
                    st.plotly_chart(fig_projected)
                
                with col8:
                    # Overseas-domestic balance
                    overseas_count = len(projected_team[projected_team['National Side'] != 'India'])
                    domestic_count = len(projected_team[projected_team['National Side'] == 'India'])
                    
                    balance_data = pd.DataFrame({
                        'Category': ['Overseas', 'Domestic'],
                        'Count': [overseas_count, domestic_count]
                    })
                    
                    fig_balance = px.pie(
                        balance_data,
                        names='Category',
                        values='Count',
                        title='Overseas-Domestic Balance'
                    )
                    st.plotly_chart(fig_balance)
                
                # Show warnings if any
                if overseas_count > planner.max_overseas:
                    st.warning(f"⚠️ Projected overseas players ({overseas_count}) exceed the limit of {planner.max_overseas}")
                
                if len(projected_team) < planner.min_players:
                    st.warning(f"⚠️ Need at least {planner.min_players - len(projected_team)} more players")

if __name__ == "__main__":
    main()  
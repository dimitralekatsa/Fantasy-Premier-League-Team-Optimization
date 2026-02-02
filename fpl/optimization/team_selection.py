import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

class WildcardOptimization:
    def __init__(
            self,
            predictions_df: pd.DataFrame,
            points_col: str = "predicted_points",
            budget: float = 1000,
            starters_budget_pct: float = 0.83,
            valid_status: list[str] = ["a", "d"],
            num_starters: int = 11,
            max_per_team: int = 3,
            future_weights: dict | None = None
    ):
        """
        Parameters:
            predictions_df: DataFrame with predicted players for upcoming gameweeks
            points_col: Name of the column that contains the predictions
            budget: Budget we have available for the whole 15-player squad (default 1000)
            starters_budget_pct: Percentage of the total budget to use for starters (default 83%)
            valid_status: Selection of which statuses we allow for the players of out team
            num_starters: Number of starter players we want to select (if num_starters = 12, better to raise the budget above 830)
            max_per_team: Players that are allowed from same team (in case the FPL changes its rules)
            future_weights: Weights for each upcoming gameweek
        """
        self.predictions_df = predictions_df
        self.points_col = points_col
        self.budget = budget
        self.starters_budget = budget * starters_budget_pct
        self.valid_status = valid_status
        self.num_starters = num_starters
        self.max_per_team = max_per_team
        
        if future_weights is None:
            gameweeks = sorted(predictions_df.gameweek.unique())
            self.future_weights = {i: max(0, 1-(0.2*i)) for i in range(len(gameweeks))}
        else:
            self.future_weights = future_weights

    def select_wildcard_team(self) -> tuple[pd.DataFrame, int, pd.DataFrame]:
        """
        Select the optimal 15-player team based on FPL constrictions
        Two-step optimizations -> first step: select starters, second step: select bench players with the budget that is left
        
        Returns:
            team_df: DataFrame with only selected for the team players
            total_cost: Cost of the selected optimal team
            updated_predictions_df: DataFrame with in_team column (0: not selected, 1: selected)
        """
        data = self.predictions_df.copy()
        id_cols = ["player_id", "team_id", "opponent_team_id"]
        for col in id_cols:
            data[col] = data[col].astype("Int16")
            
        self.predictions_df = data

        data = data[data.status.isin(self.valid_status)].reset_index(drop=True)
        data[self.points_col] = data[self.points_col].fillna(0)

        min_gameweek = data.gameweek.min()
        data["weighted_points"] = data.apply(
            lambda row: row[self.points_col] * self.future_weights.get(row.gameweek - min_gameweek, 0),
            axis=1
        )
        
        # Group by player and sum wieghted predicted points across all gameweeks
        data = data[["player_id", "player_name", "team_id", "now_cost", "position", "weighted_points"]].groupby(
            ["player_id", "player_name", "team_id", "now_cost", "position"]).sum().reset_index()
            
        # Step 1: Select the Best Starting Players
        model1 = LpProblem(name="Starting_Team_Selection", sense=LpMaximize)
        player_vars1 = {player: LpVariable(name=f"starter_{player}", cat="Binary") for player in data["player_id"]}
        
        # Objective: Maximize predicted points
        model1 += lpSum(player_vars1[player] * data.loc[i, "weighted_points"] for i, player in enumerate(data["player_id"]))
        
        # Budget constraint
        model1 += lpSum(player_vars1[player] * data.loc[i, "now_cost"] for i, player in enumerate(data["player_id"])) <= self.starters_budget
        
        # Position constraints for starting players
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Goalkeeper") == 1
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Defender") >= 3
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Defender") <= 5
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Midfielder") >= 2
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Midfielder") <= 5
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Forward") >= 1
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Forward") <= 3
        
        # Total number of starting players constraint
        model1 += lpSum(player_vars1[player] for player in player_vars1) == self.num_starters
        
        # Max 3 players per team (currently in fpl)
        teams = data["team_id"].unique()
        for team in teams:
            model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"])
                            if data.loc[i, "team_id"] == team) <= self.max_per_team
        
        model1.solve(PULP_CBC_CMD(msg=0))

        selected_players1 = [player for player in player_vars1 if player_vars1[player].value() == 1]
        
        # Step 2: Select Bench Players
        model2 = LpProblem(name="Bench_Team_Selection", sense=LpMaximize)
        player_vars2 = {player: LpVariable(name=f"bench_{player}", cat="Binary") for player in data["player_id"]}
        
        # Objective: Maximize predicted points for bench players
        model2 += lpSum(player_vars2[player] * data.loc[i, "weighted_points"] for i, player in enumerate(data["player_id"]))
        
        # Budget constraint for bench (remaining budget)
        starter_cost = sum(data.loc[data["player_id"] == p, "now_cost"].iloc[0] for p in selected_players1)
        model2 += lpSum(player_vars2[player] * data.loc[i, "now_cost"] 
                       for i, player in enumerate(data["player_id"])) <= self.budget - starter_cost
        
        # Position constraints to complete the 15-man squad
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                        if data.loc[i, "position"] == "Goalkeeper") == 1
        
        num_def_selected = sum(1 for p in selected_players1 if data.loc[data["player_id"] == p, "position"].iloc[0] == "Defender")
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Defender") == 5 - num_def_selected
        
        num_mid_selected = sum(1 for p in selected_players1 if data.loc[data["player_id"] == p, "position"].iloc[0] == "Midfielder")
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Midfielder") == 5 - num_mid_selected
        
        num_fwd_selected = sum(1 for p in selected_players1 if data.loc[data["player_id"] == p, "position"].iloc[0] == "Forward")
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Forward") == 3 - num_fwd_selected
        
        # Ensure exactly (15 - num_starters) bench players are selected
        model2 += lpSum(player_vars2[player] for player in player_vars2) == 15 - self.num_starters
        
        # Max 3 players per team (considering model1 selections)
        for team in teams:
            num_from_team = sum(1 for p in selected_players1 
                               if data.loc[data["player_id"] == p, "team_id"].iloc[0] == team)
            model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                           if data.loc[i, "team_id"] == team) <= self.max_per_team - num_from_team
        
        # Ensure bench does not include starters
        for player in selected_players1:
            model2 += player_vars2[player] == 0
        
        # Solve model2
        model2.solve(PULP_CBC_CMD(msg=0))
        
        # Extract selected bench players
        selected_players2 = [player for player in player_vars2 if player_vars2[player].value() == 1]
        
        # Combine Results
        final_team = selected_players1 + selected_players2
        final_team_df = data[data["player_id"].isin(final_team)].copy()
        final_team_df = final_team_df.sort_values(by="weighted_points", ascending=False).reset_index(drop=True)
        
        total_cost = final_team_df["now_cost"].sum()
        
        # Update predictions dataframe with in_team flag
        updated_predictions_df = self.predictions_df.copy()
        updated_predictions_df["in_team"] = updated_predictions_df["player_id"].apply(
            lambda x: 1 if x in final_team else 0)
        
        # Store results
        self.selected_team_df = final_team_df
        self.total_cost = total_cost
        self.updated_predictions_df = updated_predictions_df
        
        # Print wildcard team
        player_entries = [f"{row['player_name']} ({row['player_id']})" 
                         for _, row in final_team_df.iterrows()]
        team_string = ", ".join(player_entries)
        
        print("-"*40)
        print("WILDCARD TEAM SELECTION")
        print("-"*40)
        print(f"Team: {team_string}")
        print(f"\nTotal Squad Cost: £{self.total_cost/10:.1f}m")
        print(f"Budget Remaining: £{(self.budget - self.total_cost)/10:.1f}m")
        
        return final_team_df, total_cost, updated_predictions_df
    
    def select_best_starting_xi(self, target_gameweek: int | None = None) -> pd.DataFrame:
        """
        Select the best starting 11 from the 15-player squad for a specific gameweek
        
        Parameters:
            target_gameweek: optional. Specific gameweek to optimize for. If None, uses the minimum gameweek
        
        Returns:
            best_xi_df: Dataframe with starting XI information
        """
        if self.selected_team_df is None:
            raise ValueError("Must run select_wildcard_team() first")
        
        if target_gameweek is None:
            target_gameweek = self.predictions_df["gameweek"].min()
        
        selected_player_ids = self.selected_team_df["player_id"].tolist()
        gw_data = self.predictions_df[self.predictions_df["player_id"].isin(selected_player_ids)]
                
        # Filter for target gameweek
        upcoming_gameweek_df = gw_data[gw_data.gameweek == target_gameweek].copy()
        upcoming_gameweek_df[self.points_col] = upcoming_gameweek_df[self.points_col].fillna(0)
        
        # Optimization Model to Select Best 11
        model = LpProblem(name="Best_Starting_11_Selection", sense=LpMaximize)
        
        # Decision variables for each player
        player_vars = {row["player_id"]: LpVariable(name=f"in_team_{row['player_id']}", cat="Binary") 
                      for _, row in upcoming_gameweek_df.iterrows()}
        
        # Objective: Maximize predicted points
        model += lpSum(player_vars[row["player_id"]] * row[self.points_col] 
                      for _, row in upcoming_gameweek_df.iterrows())
        
        # Position constraints
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows() 
                      if row["position"] == "Goalkeeper") == 1
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows() 
                      if row["position"] == "Defender") >= 3
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows() 
                      if row["position"] == "Defender") <= 5
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows() 
                      if row["position"] == "Midfielder") >= 2
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows() 
                      if row["position"] == "Midfielder") <= 5
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows() 
                      if row["position"] == "Forward") >= 1
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows() 
                      if row["position"] == "Forward") <= 3
        
        # Total 11 players constraint
        model += lpSum(player_vars[row["player_id"]] for _, row in upcoming_gameweek_df.iterrows()) == 11
        
        model.solve(PULP_CBC_CMD(msg=0))
        
        # Extract selected players
        selected_players = [player for player in player_vars if player_vars[player].value() == 1]
        
        # Create the final best XI DataFrame
        best_xi_df = upcoming_gameweek_df[upcoming_gameweek_df["player_id"].isin(selected_players)].copy()
        best_xi_df = best_xi_df.sort_values(by=self.points_col, ascending=False)
        
        # Identify captain and vice-captain (top 2 predicted points)
        captain_id = best_xi_df.iloc[0]["player_id"]
        vice_captain_id = best_xi_df.iloc[1]["player_id"]
        
        total_xi_points = best_xi_df[self.points_col].where(
            best_xi_df["player_id"] != captain_id,
            best_xi_df[self.points_col] * 2
        ).sum()
        
        # Print starting XI details
        print(f"\nStarters (Gameweek {target_gameweek}):")
                
        positions_order = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        for position in positions_order:
            pos_players = best_xi_df[best_xi_df["position"] == position].sort_values(
                self.points_col, ascending=False
            )
            if not pos_players.empty:
                pos_upper = position.upper()
                players_list = []
                for _, row in pos_players.iterrows():
                    player_name = f"{row['player_name']} ({row['player_id']})"
                    if row["player_id"] == captain_id:
                        player_name += "--CAPTAIN"
                    elif row["player_id"] == vice_captain_id:
                        player_name += "--VICE CAPTAIN"
                    players_list.append(player_name)
                players_str = ", ".join(players_list)
                print(f"{pos_upper}: {players_str}")
        
        print(f"\nTotal Starting 11 Predicted Points: {total_xi_points:.1f}")
        
        bench_players = upcoming_gameweek_df[~upcoming_gameweek_df["player_id"].isin(selected_players)].copy()

        print(f"\nBench (sorted by predicted points):")
        bench_sorted = bench_players.sort_values(by=self.points_col, ascending=False)
        bench_names = ", ".join([f"{row["player_name"]} ({row["player_id"]})" for _, row in bench_sorted.iterrows()])
        print(bench_names)
        
        return best_xi_df
    
    def optimize_wildcard(self, target_gameweek: int | None = None) -> tuple:
        """
        Complete wildcard optimization workflow with automatic budget reduction until feasible solution is found
        
        Parameters:
            target_gameweek: optional. Gameweek for starting 11 selection, if None uses the minimum
        
        Returns:
            team_df: DataFrame with only selected players
            updated_df: DataFrame with added in_team column (0: if player is not selected, 1: if he is selected)
        """
        original_starters_budget = self.starters_budget
        min_budget = 750
        budget_step = 5
        
        team_df, _, updated_df = self.select_wildcard_team()

        if len(team_df) == 15:
            self.select_best_starting_xi(target_gameweek=target_gameweek)
            return team_df, updated_df
        
        print(f"WARNING: Only got {len(team_df)} starters with budget £{self.starters_budget/10:.1f}m")
        current_starters_budget = self.starters_budget - budget_step

        while current_starters_budget >= min_budget and len(team_df) != 15:
            print(f"Trying with reduced budget: £{current_starters_budget/10:.1f}m")
            
            # Create new optimizer with reduced budget
            optimizer = WildcardOptimization(
                predictions_df=self.predictions_df,
                points_col=self.points_col,
                budget=self.budget,
                starters_budget_pct=current_starters_budget / self.budget,
                valid_status=self.valid_status,
                num_starters=self.num_starters,
                max_per_team=self.max_per_team,
                future_weights=self.future_weights
            )
            
            team_df, _, updated_df = optimizer.select_wildcard_team()
            
            if len(team_df) == 15:                    
                # Update our state to match the successful optimizer
                self.selected_team_df = optimizer.selected_team_df
                self.total_cost = optimizer.total_cost
                self.updated_predictions_df = optimizer.updated_predictions_df
                self.starters_budget = current_starters_budget
                break
            else:
                print(f"Got {len(team_df)} players, continuing...")
            
            # Reduce budget for next attempt
            current_starters_budget -= budget_step
        
        if len(team_df) != 15:
            self.starters_budget = original_starters_budget  # Restore original starters budget
            raise ValueError(f"Could not find valid 15-player team with any budget from £{original_starters_budget/10:.1f}m down to £{min_budget/10:.1f}m")
        
        # Show budget reduction info
        if self.starters_budget != original_starters_budget:
            print(f"Note: Starters budget was reduced from £{original_starters_budget/10:.1f}m to £{self.starters_budget/10:.1f}m")
        
        # Select best starting XI
        best_xi_df = self.select_best_starting_xi(target_gameweek)
        
        return team_df, best_xi_df, updated_df


class FreeHitOptimization:
    def __init__(
            self,
            predictions_df: pd.DataFrame,
            points_col: str = "predicted_points",
            budget: float = 1000,
            starters_budget_pct: float = 0.83,
            valid_status: list[str] = ["a", "d"],
            max_per_team: int = 3,
    ):
        """
        Initialize the Free Hit Optimization class
        
        Parameters:
            predictions_df: DataFrame with player predictions
            points_col: Name of the wanted points column for optimization
            budget: Budget we have available for the whole 15-player squad (default 1000)
            starters_budget_pct: Percentage of the total budget to use for starters (default 83%)
            valid_status: Selection of which statuses we allow for the players of out team
            max_per_team: Players that are allowed from same team
        """
        self.predictions_df = predictions_df
        self.points_col = points_col
        self.budget = budget
        self.starters_budget = budget * starters_budget_pct
        self.valid_status = valid_status
        self.max_per_team = max_per_team
        
        self.selected_team_df = None
        self.total_cost = None
        self.total_predicted_points = None
        self.updated_predictions_df = None
        self.target_gameweek = None
        
    def select_freehit_team(self, target_gameweek: int | None = None) -> tuple[pd.DataFrame, int, float]:
        """
        Select the optimal 15-player team for the upcoming gameweek using Free Hit chip
        
        Parameters:
            target_gameweek: optional. Specific gameweek to optimize for. If None, uses the minimum gameweek.
        
        Returns:
            team_df: DataFrame with only selected for the team players
            total_cost: Cost of the selected optimal team
            total_predicted_points: Sum of the optimal teams predicted points (only for starters)
        """
        predictions_df = self.predictions_df.copy()
        id_cols = ["player_id", "team_id", "opponent_team_id"]
        for col in id_cols:
            predictions_df[col] = predictions_df[col].astype("Int16")
        
        # Filter for target gameweek
        if target_gameweek is None:
            target_gameweek = predictions_df["gameweek"].min()
        
        self.target_gameweek = target_gameweek
        print(f"Optimizing Free Hit team for Gameweek {target_gameweek}")
        
        # Filter data for the specific gameweek and valid players
        data = predictions_df[
            (predictions_df["gameweek"] == target_gameweek) &
            (predictions_df["status"].isin(self.valid_status))
        ].copy().reset_index(drop=True)
        
        if data.empty:
            raise ValueError(f"No valid players found for gameweek {target_gameweek}")
        
        data[self.points_col] = data[self.points_col].fillna(0)
        
        # Step 1: Select the Best Starting Players
        model1 = LpProblem(name="FreeHit_Starters_Selection", sense=LpMaximize)
        player_vars1 = {player: LpVariable(name=f"starter_{player}", cat="Binary") 
                        for player in data["player_id"]}
        
        # Objective: Maximize predicted points for starting XI
        model1 += lpSum(player_vars1[player] * data.loc[i, self.points_col] 
                       for i, player in enumerate(data["player_id"]))
        
        # Budget constraint for starting XI
        model1 += lpSum(player_vars1[player] * data.loc[i, "now_cost"] 
                       for i, player in enumerate(data["player_id"])) <= self.starters_budget
        
        # Position constraints for starting XI
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Goalkeeper") == 1
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Defender") >= 3
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Defender") <= 5
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Midfielder") >= 2
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Midfielder") <= 5
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Forward") >= 1
        model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Forward") <= 3
        
        # Total 11 starting players constraint
        model1 += lpSum(player_vars1[player] for player in player_vars1) == 11
        
        # Max players per team
        teams = data["team_id"].unique()
        for team in teams:
            model1 += lpSum(player_vars1[player] for i, player in enumerate(data["player_id"]) 
                           if data.loc[i, "team_id"] == team) <= self.max_per_team
        
        # Solve model
        model1.solve(PULP_CBC_CMD(msg=0))
        
        # Extract selected starting players
        selected_players1 = [player for player in player_vars1 if player_vars1[player].value() == 1]
        
        # Step 2: Select Bench Players
        model2 = LpProblem(name="FreeHit_Bench_Selection", sense=LpMaximize)
        player_vars2 = {player: LpVariable(name=f"bench_{player}", cat="Binary") 
                        for player in data["player_id"]}
        
        # Objective: Maximize predicted points for bench players
        model2 += lpSum(player_vars2[player] * data.loc[i, self.points_col] 
                       for i, player in enumerate(data["player_id"]))
        
        # Budget constraint for bench (remaining budget)
        starter_cost = sum(data.loc[data["player_id"] == p, "now_cost"].iloc[0] for p in selected_players1)
        model2 += lpSum(player_vars2[player] * data.loc[i, "now_cost"] 
                       for i, player in enumerate(data["player_id"])) <= self.budget - starter_cost
        
        # Position constraints to complete the 15-man squad
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Goalkeeper") == 1
        
        num_def_selected = sum(1 for p in selected_players1 
                              if data.loc[data["player_id"] == p, "position"].iloc[0] == "Defender")
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Defender") == 5 - num_def_selected
        
        num_mid_selected = sum(1 for p in selected_players1 
                              if data.loc[data["player_id"] == p, "position"].iloc[0] == "Midfielder")
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Midfielder") == 5 - num_mid_selected
        
        num_fwd_selected = sum(1 for p in selected_players1 
                              if data.loc[data["player_id"] == p, "position"].iloc[0] == "Forward")
        model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                       if data.loc[i, "position"] == "Forward") == 3 - num_fwd_selected
        
        # Total 15 players constraint
        model2 += lpSum(player_vars2[player] for player in player_vars2) == 4

        # Max players per team
        for team in teams:
            num_from_team = sum(1 for p in selected_players1 
                               if data.loc[data["player_id"] == p, "team_id"].iloc[0] == team)
            model2 += lpSum(player_vars2[player] for i, player in enumerate(data["player_id"]) 
                           if data.loc[i, "team_id"] == team) <= self.max_per_team - num_from_team
        
        # Ensure bench does not include starters
        for player in selected_players1:
            model2 += player_vars2[player] == 0
        
        # Solve model
        model2.solve(PULP_CBC_CMD(msg=0))
        
        # Extract selected bench players
        selected_players2 = [player for player in player_vars2 if player_vars2[player].value() == 1]
        
        # Combine Results
        final_team = selected_players1 + selected_players2
        starters_df = data[data["player_id"].isin(selected_players1)].copy()
        bench_df = data[data["player_id"].isin(selected_players2)].copy()
        final_team_df = data[data["player_id"].isin(final_team)].copy()
        final_team_df = final_team_df.sort_values(by=self.points_col, ascending=False).reset_index(drop=True)

        total_cost = final_team_df["now_cost"].sum()

        # Identify captain and vice-captain (top 2 predicted points)
        starters_df = starters_df.sort_values(by=self.points_col, ascending=False)
        captain_id = starters_df.iloc[0]["player_id"]
        vice_captain_id = starters_df.iloc[1]["player_id"]

        total_predicted_points = starters_df[self.points_col].where(
            starters_df["player_id"] != captain_id,
            starters_df[self.points_col] * 2
        ).sum()

        self.selected_team_df = final_team_df
        self.total_cost = total_cost
        self.total_predicted_points = total_predicted_points
        
        # Print starting XI details
        print("-"*40)
        print(f"FREE HIT TEAM SELECTION (Gameweek {self.target_gameweek})")
        print("-"*40)
        print(f"\nStarters:")
        
        # Group by position
        positions_order = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
        for position in positions_order:
            pos_players = starters_df[starters_df["position"] == position].sort_values(
                self.points_col, ascending=False
            )
            if not pos_players.empty:
                pos_upper = position.upper()
                players_list = []
                for _, row in pos_players.iterrows():
                    player_name = f"{row["player_name"]} ({row["player_id"]})"
                    if row["player_id"] == captain_id:
                        player_name += "--CAPTAIN"
                    elif row["player_id"] == vice_captain_id:
                        player_name += "--VICE CAPTAIN"
                    players_list.append(player_name)
                players_str = ", ".join(players_list)
                print(f"{pos_upper}: {players_str}")
        
        print(f"\nTotal Starting 11 Predicted Points: {total_predicted_points:.1f}")

        print(f"\nBench (sorted by predicted points):")
        bench_sorted = bench_df.sort_values(by=self.points_col, ascending=False)
        bench_names = ", ".join([f"{row["player_name"]} ({row["player_id"]})" for _, row in bench_sorted.iterrows()])
        print(bench_names)
        
        return final_team_df, total_cost, total_predicted_points
    
    def optimize_freehit(self, target_gameweek: int | None = None):
        """
        Complete Free Hit optimization workflow.
        
        Parameters:
            target_gameweek: optional. Gameweek for Free Hit optimization. If None, uses minimum gameweek.
        
        Returns:
            team_df: DataFrame with only selected players
        """
        # Select the 15-player Free Hit team
        team_df, _, _ = self.select_freehit_team(
            target_gameweek=target_gameweek
        )
        
        return team_df

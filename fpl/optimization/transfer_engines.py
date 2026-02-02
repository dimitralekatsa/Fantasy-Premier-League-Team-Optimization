import pandas as pd
import numpy as np
import itertools
from itertools import combinations
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD, LpStatusOptimal

from fpl.utils.utils import (
    backfill_missing_players,
)

class BeamSearchFPLOptimizer:
    def __init__(
            self,
            players_df: pd.DataFrame,
            current_team: list,
            points_prefix: str = "Points_GW", 
            team_size: int = 15,
            max_transfers: int = 5,
            max_per_team: int = 3,
            budget: float = 1000, 
            initial_transfers: int = 1,
            beam_width: int = 50,
            future_weights: dict | None = None,
            debug: bool = False,
            valid_status: list[str] = ["a", "d"]):
        """      
        Parameters:
            players_df: wide-format dataframe with player information and predicted points
            current_team: list of player_ids in the current team
            points_prefix: prefix for the predicted points columns in the dataframe
            team_size: number of players in a team (default: 15)
            max_transfers: maximum number of transfers allowed at once (default: 5)
            max_per_team: maximum players allowed from a single team (default: 3)
            budget: total budget constraint (default: 1000, which is £100.0m in FPL)
            initial_transfers: number of transfers available at the start
            beam_width: number of states to maintain in the beam (default: 50)
            future_weights: weights for future gameweeks (default: linearly decreasing by 0.2)
            debug: whether to print debug information
        """
        self.players_df = players_df
        self.points_prefix = points_prefix
        self.team_size = team_size
        self.max_transfers = max_transfers
        self.max_per_team = max_per_team
        self.budget = budget
        self.initial_transfers = initial_transfers
        self.beam_width = beam_width
        self.debug = debug
        self.valid_status = valid_status
        
        # Extract all gameweeks in the prediction horizon
        self.gameweeks = sorted([int(col.replace(points_prefix, '')) 
                                for col in players_df.columns 
                                if col.startswith(points_prefix)])
        
        # Set future weights if not provided
        if future_weights is None:
            # Default to linearly decreasing weights with 0.2 steps
            max_future_gws = len(self.gameweeks) - 1
            self.future_weights = {i: max(0, 1.0 - (0.2 * i)) for i in range(max_future_gws + 1)}
        else:
            self.future_weights = future_weights
            
        if self.debug:
            print(f"DEBUG: Future weights: {self.future_weights}")
        
        # Create a player lookup dictionary for quick access
        self.player_lookup = {}
        for player_id, row in players_df.set_index("player_id").iterrows():
            player_dict = row.to_dict()
            player_dict["player_id"] = player_id
            self.player_lookup[player_id] = player_dict

        self.current_team = current_team.copy()

        if len(self.current_team) != 15:
            raise ValueError(f"Current team must have exactly 15 players, got {len(self.current_team)}")
        
        # Validate all current team players exist in the data
        missing_players = [int(p) for p in self.current_team if p not in self.player_lookup]
        if missing_players:
            self.players_df, found_all = backfill_missing_players(
                df=self.players_df,
                missing_players=missing_players,
                points_prefix=self.points_prefix
            )

            if not found_all:
                raise ValueError(f"Players not found in data: {missing_players}")

            # Add backfilled players to the lookup
            new_rows = self.players_df[self.players_df["player_id"].isin(missing_players)]
            for player_id, row in new_rows.set_index("player_id").iterrows():
                player_dict = row.to_dict()
                player_dict["player_id"] = player_id
                self.player_lookup[player_id] = player_dict

        # When finding valid players for transfers, filter by status
        self.valid_transfer_players = {
            player_id: player for player_id, player in self.player_lookup.items()
            if player.get("status", "u") in valid_status
        }

        # Pre-filter by position
        self.players_by_position = {
            "Goalkeeper": [],
            "Defender": [],
            "Midfielder": [],
            "Forward": [],
        }
        for player_id, player in self.valid_transfer_players.items():
            self.players_by_position[player["position"]].append((player_id, player))
        
        # Detect the minimum available gameweek
        self.current_gw = min(self.gameweeks)
        
        # Calculate initial team value
        self.initial_team_value = sum(self.player_lookup[p]["now_cost"] for p in self.current_team)
        if self.debug:
            print(f"DEBUG: Initial team value: {self.initial_team_value}")
            print(f"DEBUG: Budget available: {self.budget}")
            
            if self.initial_team_value > self.budget:
                print(f"DEBUG: Warning - Initial team value ({self.initial_team_value}) exceeds budget ({self.budget})")

    def select_best_starting_11(self, team: list, gameweek: int) -> tuple[list, float, tuple, int]:
        """
        Selects the best starting 11 players and captain based on valid FPL formations

        Parameters:
            team: list of player_ids in the current 15-man squad            
            gameweek: the gameweek for which we are selecting the best team
        
        Returns:
            - List of the best 11 player_ids (with first player as captain)
            - Total predicted points for this best 11 (including doubled captain points)
            - Formation used (Defender, Midfielder, Forward)
            - Vice-captain player_id
        """
        valid_formations = [
            (5, 4, 1), (5, 3, 2),  # Five defenders formations
            (4, 4, 2), (4, 3, 3),  # Four defenders formations
            (3, 5, 2), (3, 4, 3)   # Three defenders formations
        ]
        
        points_column = f"{self.points_prefix}{gameweek}"
        
        # Extract all players and treat NaN points as 0
        all_players = []
        players_with_nan = []
        
        for p in team:
            player_data = self.player_lookup[p].copy()
            points_value = player_data.get(points_column)
            
            if pd.isna(points_value) or points_value is None:
                player_data[points_column] = 0.0
                players_with_nan.append(player_data["player_name"])
                if self.debug:
                    print(f"DEBUG: Player {player_data['player_name']} has NaN points, treating as 0")
            
            all_players.append(player_data)
        
        if self.debug and players_with_nan:
            print(f"DEBUG: {len(players_with_nan)} players have NaN points for GW{gameweek}: {players_with_nan}")
        
        # Separate players by position
        gks = sorted([p for p in all_players if p["position"] == "Goalkeeper"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        defs = sorted([p for p in all_players if p["position"] == "Defender"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        mids = sorted([p for p in all_players if p["position"] == "Midfielder"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        fwds = sorted([p for p in all_players if p["position"] == "Forward"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        
        if self.debug:
            print(f"DEBUG: Position breakdown - GK: {len(gks)}, DEF: {len(defs)}, MID: {len(mids)}, FWD: {len(fwds)}")

        best_team = []
        best_points = float("-inf")
        best_formation = None
        vice_captain_id = None

        formation_points = {}
        
        for formation in valid_formations:
            num_def, num_mid, num_fwd = formation

            # Ensure we have enough players for the formation
            if len(gks) < 1 or len(defs) < num_def or len(mids) < num_mid or len(fwds) < num_fwd:
                if self.debug:
                    print(f"DEBUG: Skipping formation {formation} - not enough players")
                continue  # Skip invalid formations

            # Select the best players for this formation
            selected_gk = [gks[0]]
            selected_defs = defs[:num_def]
            selected_mids = mids[:num_mid]
            selected_fwds = fwds[:num_fwd]
            
            # Combine all selected players
            selected_players = selected_gk + selected_defs + selected_mids + selected_fwds
            
            # Sort players by points to get captain and vice captain
            sorted_by_points = sorted(selected_players, 
                                    key=lambda p: p.get(f"{self.points_prefix}{gameweek}", 0), 
                                    reverse=True)

            # Get captain (highest points) and vice captain (second highest)
            captain = sorted_by_points[0]
            vice_captain = sorted_by_points[1] if len(sorted_by_points) > 1 else None
            
            # Calculate total predicted points (with captain doubled)
            total_points = 0
            for player in selected_players:
                points = player.get(f"{self.points_prefix}{gameweek}", 0)
                if player == captain:
                    points *= 2
                total_points += points
            
            # Store formation points for debugging
            formation_str = f"{num_def}-{num_mid}-{num_fwd}"
            formation_points[formation_str] = total_points
                
            # Extract player_ids
            selected_team = [p["player_id"] for p in selected_players]
            
            # Update the best team if this formation gives more points
            if total_points > best_points:
                best_points = total_points
                captain_id = captain["player_id"]
                vice_captain_id = vice_captain["player_id"] if vice_captain else None

                selected_team = [p["player_id"] for p in selected_players]
                selected_team.remove(captain_id)
                if vice_captain_id and vice_captain_id in selected_team:
                    selected_team.remove(vice_captain_id)
                    best_team = [captain_id, vice_captain_id] + selected_team
                else:
                    best_team = [captain_id] + selected_team
                best_formation = formation
        
        # Debug output for formation comparison
        if self.debug and formation_points:
            print("\nDEBUG: Points by formation (including captain bonus):")
            for form, points in sorted(formation_points.items(), key=lambda x: x[1], reverse=True):
                star = " *" if form == f"{best_formation[0]}-{best_formation[1]}-{best_formation[2]}" else ""
                print(f"DEBUG:   {form}: {points:.2f} pts{star}")

        # If we couldn't form a valid team , return an empty team
        if not best_team:
            best_team = [None] * 11
            best_points = 0
            best_formation = (0, 0, 0)

        return best_team, best_points, best_formation, vice_captain_id
    
    def find_optimal_team_for_gameweek(self, gameweek: int) -> tuple[list, float]:
        """
        Use PuLP to find the optimal team for a gameweek from all available players
        
        Parameters:
            gameweek: the gameweek to optimize for
            
        Returns:
            (list of player_ids in the optimal team, total predicted points)
        """
        model = LpProblem(name="Optimal_FPL_Team", sense=LpMaximize)
        
        # Filter players with valid points for this gameweek - only consider transferable players
        valid_players = []
        for _, player in self.valid_transfer_players.items():
            if pd.notna(player.get(f"{self.points_prefix}{gameweek}", 0)):
                valid_players.append(player)
        
        player_vars = {}
        for player in valid_players:
            player_vars[player["player_id"]] = LpVariable(f"player_{player['player_id']}", cat="Binary")
        
        # Objective: Maximize total expected points (with captain consideration)
        model += lpSum([player_vars[player["player_id"]] * player.get(f"{self.points_prefix}{gameweek}", 0) 
                         for player in valid_players])

        # Budget constraint
        model += lpSum([player_vars[player["player_id"]] * player["now_cost"] for player in valid_players]) <= self.budget
        
        # Team size constraint (only 11 for the optimal team as we only care about starters)
        model += lpSum([player_vars[player["player_id"]] for player in valid_players]) == 11
        
        # Position constraints
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Goalkeeper"]) == 1
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Defender"]) >= 3
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Defender"]) <= 5
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Midfielder"]) >= 2
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Midfielder"]) <= 5
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Forward"]) >= 1
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Forward"]) <= 3
        
        # Team constraint
        for team_id in set(p["team_id"] for p in valid_players):
            model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                            if player["team_id"] == team_id]) <= self.max_per_team

        model.solve(PULP_CBC_CMD(msg=False))
        
        # Get the selected players
        optimal_team = [player["player_id"] for player in valid_players 
                        if player_vars[player["player_id"]].value() == 1]
        
        # Calculate the predicted points
        total_points = 0
        for p in optimal_team:
            points = self.player_lookup[p].get(f"{self.points_prefix}{gameweek}", 0)
            if pd.isna(points):
                points = 0.0
            total_points += points
        optimal_team_points = total_points
        
        if optimal_team:
            captain_id = max(optimal_team, key=lambda p: self.player_lookup[p].get(f"{self.points_prefix}{gameweek}", 0))
            captain_points = self.player_lookup[captain_id].get(f"{self.points_prefix}{gameweek}", 0)
            optimal_team_points += captain_points
        else:
            optimal_team_points = 0
            
        if self.debug:
            print(f"\nOptimal team found by PuLP for GW {gameweek} (Points: {optimal_team_points:.2f}):")
            for pid in optimal_team:
                player = self.player_lookup[pid]
                pts = player.get(f"{self.points_prefix}{gameweek}", 0)
                captain_mark = " (C)" if pid == captain_id else ""
                print(f"  {player['player_name']} ({player['position']}){captain_mark} - {pts:.2f} pts")
        
        return optimal_team, optimal_team_points
    
    def get_valid_transfers(self, team: list, optimal_team: list, available_transfers: int, gameweek: int) -> list:
        """
        Generate all valid transfers for the current team, sorted by weighted impact
        Only considers transferring out players who are not in the optimal team
        
        Parameters:
            team: list of player_ids in the current team
            optimal_team: list of player_ids in the optimal team for this gameweek
            available_transfers: number of transfers available
            gameweek: current gameweek
        
        Returns:
            list of valid transfers (player out, player in) tuples, sorted by weighted impact
        """
        # If no transfers available, return empty list
        if available_transfers == 0:
            if self.debug:
                print("DEBUG: No transfers available for this gameweek")
            return []
        
        # Identify players in common (these we won't prioritize transferring out)
        common_players = set(team).intersection(set(optimal_team))
        
        if self.debug:
            print(f"DEBUG: Players in common with optimal team: {len(common_players)}")
            print(f"DEBUG: Common players: {[self.player_lookup[p]['player_name'] for p in common_players]}")
        
        # Get current team composition
        team_composition = {}
        for player_id in team:
            team_id = self.player_lookup[player_id]["team_id"]
            team_composition[team_id] = team_composition.get(team_id, 0) + 1
        
        # Calculate current team value for total budget check
        current_team_value = sum(self.player_lookup[p]["now_cost"] for p in team)
        
        # Find all valid player swaps
        valid_transfers = []
        invalid_count = 0
        
        # First, prioritize transferring out players not in the optimal team
        for out_player in team:
            # Skip players that are in the optimal team
            if out_player in common_players:
                continue
                
            out_position = self.player_lookup[out_player]["position"]
            out_team = self.player_lookup[out_player]["team_id"]
            out_cost = self.player_lookup[out_player]["now_cost"]
            out_points = self.player_lookup[out_player].get(f"{self.points_prefix}{gameweek}", 0)
            
            # Filter for potential replacements in the same position
            for in_player_id, in_player in self.players_by_position[out_position]:
                if in_player_id in team:
                    continue
                    
                in_points = in_player.get(f"{self.points_prefix}{gameweek}", 0)
                if pd.isna(in_points):
                    in_points = 0.0

                in_points = in_player.get(f"{self.points_prefix}{gameweek}", 0)
                in_team = in_player["team_id"]
                in_cost = in_player["now_cost"]
                
                # Check team constraint
                new_team_composition = team_composition.copy()
                new_team_composition[out_team] = new_team_composition.get(out_team, 0) - 1
                new_team_composition[in_team] = new_team_composition.get(in_team, 0) + 1
                
                if new_team_composition[in_team] > self.max_per_team:
                    invalid_count += 1
                    continue
                
                # Check total budget constraint
                new_team_value = current_team_value - out_cost + in_cost
                if new_team_value > self.budget:
                    if self.debug:
                        invalid_count += 1
                        print(f"DEBUG: Transfer {self.player_lookup[out_player]['player_name']} → {in_player['player_name']} invalid - budget exceeded: {new_team_value} > {self.budget}")
                    continue
                
                # Valid transfer
                valid_transfers.append((out_player, in_player_id))
        
        # If we still have room for transfers and haven't found any valid ones yet,
        # consider transferring players who are in the optimal team but might still be improved
        if len(valid_transfers) == 0 and available_transfers > 0:
            for out_player in common_players:
                out_position = self.player_lookup[out_player]["position"]
                out_team = self.player_lookup[out_player]["team_id"]
                out_cost = self.player_lookup[out_player]["now_cost"]
                out_points = self.player_lookup[out_player].get(f"{self.points_prefix}{gameweek}", 0)
                
                # Filter for potential replacements in the same position
                for in_player_id, in_player in self.players_by_position[out_position]:

                    # Skip if player is already in team
                    if in_player_id in team:
                        continue
                        
                    in_points = in_player.get(f"{self.points_prefix}{gameweek}", 0)
                    if pd.isna(in_points):
                        in_points = 0.0
                    
                    # Get in_player points (changes each iteration)
                    in_points = in_player.get(f"{self.points_prefix}{gameweek}", 0)

                    # Only skip if the new player is worse both now and in the future
                    if in_points < out_points:
                        continue
                    
                    in_team = in_player["team_id"]
                    in_cost = in_player["now_cost"]
                    
                    # Check team constraint
                    new_team_composition = team_composition.copy()
                    new_team_composition[out_team] = new_team_composition.get(out_team, 0) - 1
                    new_team_composition[in_team] = new_team_composition.get(in_team, 0) + 1
                    
                    if new_team_composition[in_team] > self.max_per_team:
                        invalid_count += 1
                        continue
                    
                    # Check total budget constraint
                    new_team_value = current_team_value - out_cost + in_cost
                    if new_team_value > self.budget:
                        invalid_count += 1
                        continue
                    
                    # Valid transfer
                    valid_transfers.append((out_player, in_player_id))
        
        if self.debug:
            print(f"DEBUG: Found {len(valid_transfers)} valid transfers and rejected {invalid_count} invalid transfers for gameweek {gameweek}")
        
        # Sort transfers by weighted total impact (current + weighted future)
        transfer_impacts = []
        
        for out_player, in_player in valid_transfers:
            total_weighted_impact = 0
            
            current_gw_index = self.gameweeks.index(gameweek)
            
            # Calculate impact for current and future gameweeks with weights
            for i, gw in enumerate(self.gameweeks[current_gw_index:]):
                weight_index = i
                weight = self.future_weights.get(weight_index, 0)
                
                if weight <= 0:
                    break  # No point calculating if weight is 0
                
                # Calculate points difference for this gameweek
                out_points = self.player_lookup[out_player].get(f"{self.points_prefix}{gw}", 0)
                in_points = self.player_lookup[in_player].get(f"{self.points_prefix}{gw}", 0)
                
                # Skip if either player has NaN points for this gameweek
                if pd.isna(out_points):
                    out_points = 0.0
                if pd.isna(in_points):
                    in_points = 0.0
                    
                    
                points_diff = in_points - out_points
                
                # Add weighted contribution
                total_weighted_impact += weight * points_diff
            
            transfer_impacts.append(((out_player, in_player), total_weighted_impact))
        
        # Sort by total weighted impact
        transfer_impacts.sort(key=lambda x: x[1], reverse=True)
        
        if self.debug:
            print(f"DEBUG: Top 5 transfers by weighted impact for GW{gameweek}:")
            for i, ((out_id, in_id), impact) in enumerate(transfer_impacts[:5]):
                out_name = self.player_lookup[out_id]["player_name"]
                in_name = self.player_lookup[in_id]["player_name"]
                print(f"DEBUG:   {i+1}. {out_name} → {in_name} (Weighted impact: {impact:.2f})")
        
        # Return sorted transfers
        return [transfer for transfer, _ in transfer_impacts]
    
    def generate_transfer_combinations(self, team, optimal_team, available_transfers, gameweek, max_combinations=1000):
        """
        Generate all possible combinations of transfers up to available_transfers.
        
        Parameters:
        -----------
        team : list
            List of player_ids in the current team.
        optimal_team : list
            List of player_ids in the optimal team for this gameweek.
        available_transfers : int
            Number of transfers available.
        gameweek : int
            Current gameweek.
        max_combinations : int
            Maximum number of combinations to generate (for performance control).
            
        Returns:
        --------
        list
            List of transfer combinations, where each combination is a list of (out, in) tuples.
        """
        valid_single_transfers = self.get_valid_transfers(team, optimal_team, available_transfers, gameweek)
        
        if not valid_single_transfers:
            return [[]]  # Return just the empty combination (no transfers)
        
        # Limit the number of single transfers to consider (take the best ones)
        max_single_transfers = min(100, len(valid_single_transfers))
        limited_transfers = valid_single_transfers[:max_single_transfers]

        if self.debug:
            print(f"DEBUG: Limited to top {len(limited_transfers)} transfers out of {len(valid_single_transfers)} total")
            
        # Group transfers by outgoing player to avoid replacing the same player twice
        transfers_by_out = {}
        for out_id, in_id in limited_transfers:
            if out_id not in transfers_by_out:
                transfers_by_out[out_id] = []
            transfers_by_out[out_id].append(in_id)
        
        # Generate all possible combinations of transfers
        all_combinations = [[]]
        combination_count = 1
        
        for num_transfers in range(1, min(available_transfers, len(transfers_by_out)) + 1):
            if self.debug:
                print(f"DEBUG: Generating combinations of {num_transfers} transfers...")
            
            # Choose which players to transfer out
            for out_players in itertools.combinations(transfers_by_out.keys(), num_transfers):
                # For each player to transfer out, consider all possible replacements
                transfer_options = []
                for out_player in out_players:
                    in_options = [(out_player, in_player) for in_player in transfers_by_out[out_player]]
                    transfer_options.append(in_options)
                
                # Generate all combinations of transfers
                for transfers in itertools.product(*transfer_options):
                    # Check combination size limit
                    if combination_count >= max_combinations:
                        if self.debug:
                            print(f"DEBUG: Reached max combinations limit of {max_combinations}")
                        return all_combinations
                    
                    in_players = [in_id for out_id, in_id in transfers]
                    if len(in_players) != len(set(in_players)):
                        continue  # Skip combinations with duplicate incoming players

                    # Verify the combination is valid
                    if self.is_valid_transfer_combination(team, list(transfers)):
                        all_combinations.append(list(transfers))
                        combination_count += 1
        
        if self.debug:
            print(f"DEBUG: Generated {len(all_combinations)} valid transfer combinations")
        
        return all_combinations
    
    def is_valid_transfer_combination(self, original_team: list, transfers: list) -> bool:
        """
        Check if a combination of transfers is valid
        
        Parameters:
            original_team: list of player_ids in the original team
            transfers: list of (out, in) tuples representing transfers
        
        Returns:
            True if the combination is valid, False otherwise
        """
        if not transfers:
            return True
        
        # Create a new team after transfers
        new_team = original_team.copy()
        for out_id, in_id in transfers:
            new_team = [p if p != out_id else in_id for p in new_team]
        
        # Check position constraints
        positions = {"Goalkeeper": 0, "Defender": 0, "Midfielder": 0, "Forward": 0}
        for player_id in new_team:
            pos = self.player_lookup[player_id]["position"]
            positions[pos] = positions.get(pos, 0) + 1
        
        if positions.get("Goalkeeper", 0) != 2 or positions.get("Defender", 0) != 5 or \
           positions.get("Midfielder", 0) != 5 or positions.get("Forward", 0) != 3:
            return False
        
        # Check team composition
        team_composition = {}
        for player_id in new_team:
            team_id = self.player_lookup[player_id]["team_id"]
            team_composition[team_id] = team_composition.get(team_id, 0) + 1
            if team_composition[team_id] > self.max_per_team:
                return False
        
        # Check budget constraint
        total_cost = sum(self.player_lookup[p]["now_cost"] for p in new_team)
        if total_cost > self.budget:
            return False
        
        return True
    
    def optimize_transfers(self, max_combinations_per_gw: int = 2000) -> dict:
        """
        Optimize transfers using beam search algorithm with optimized search space
        
        Returns:
            Optimization results with transfer plan
        """
        beam = []
        initial_state = {
            "team": self.current_team.copy(),
            "transfers_available": self.initial_transfers,
            "total_points": 0,
            "history": []
        }
        beam.append(initial_state)
        
        for i, gw in enumerate(self.gameweeks):
            weight = self.future_weights.get(i, 0)
            
            if self.debug:
                print(f"\nDEBUG: Processing gameweek {gw} (weight: {weight:.2f})...")
                print(f"DEBUG: Beam size: {len(beam)}")
            
            next_beam = []
            
            # Process each state in the current beam
            for state in beam:
                current_team = state["team"]
                available_transfers = state["transfers_available"]
                
                if self.debug:
                    print(f"DEBUG: Available transfers: {available_transfers}")
                    print(f"DEBUG: Team value: {sum(self.player_lookup[p]['now_cost'] for p in current_team)}")
                
                # Find optimal team for this gameweek
                optimal_team, optimal_points = self.find_optimal_team_for_gameweek(gw)
                
                # Generate all possible transfer combinations with limit
                transfer_combinations = self.generate_transfer_combinations(
                    current_team, optimal_team, available_transfers, gw,
                    max_combinations=max_combinations_per_gw
                )
                
                # Apply each transfer combination and evaluate
                for transfers in transfer_combinations:
                    new_team = current_team.copy()
                    for out_id, in_id in transfers:
                        new_team = [p if p != out_id else in_id for p in new_team]

                    if len(new_team) != len(set(new_team)):
                        duplicates = [pid for pid in set(new_team) if new_team.count(pid) > 1]
                        print(f"ERROR: Transfer created duplicates: {duplicates}")
                        print(f"ERROR: Original team: {current_team}")
                        print(f"ERROR: Transfers: {transfers}")
                        continue  # Skip invalid combinations
                    
                    # Find best starting 11 and calculate points
                    best_11, unweighted_points, formation, vice_captain_id = self.select_best_starting_11(new_team, gw)
                    
                    # Apply gameweek weight to the points
                    weighted_points = unweighted_points * weight
                    
                    # Calculate next available transfers
                    if transfers:
                        used_transfers = len(transfers)
                        next_available = max(1, available_transfers - used_transfers + 1)
                        if self.debug:
                            print(f"Made {used_transfers} transfers. Next GW available transfers: {next_available}")
                    else:
                        next_available = min(available_transfers + 1, self.max_transfers)  # Bank a transfer
                        if self.debug:
                            print(f"No transfers made. Banking 1 transfer. Next GW available transfers: {next_available}")
                    
                    # Create transfer details for history
                    transfer_details = []
                    for out_id, in_id in transfers:
                        out_name = self.player_lookup[out_id]["player_name"]
                        in_name = self.player_lookup[in_id]["player_name"]
                        out_points = self.player_lookup[out_id].get(f"{self.points_prefix}{gw}", 0)
                        in_points = self.player_lookup[in_id].get(f"{self.points_prefix}{gw}", 0)
                        if pd.isna(out_points):
                            out_points = 0.0
                        if pd.isna(in_points):
                            in_points = 0.0
                        points_gain = in_points - out_points
                        transfer_details.append({
                            "out": out_name,
                            "in": in_name,
                            "points_gain": points_gain,
                            "out_id": out_id,
                            "in_id": in_id
                        })
                    
                    # Extract captain ID (first player in best_11)
                    captain_id = best_11[0] if best_11 and best_11[0] is not None else None
                    captain_name = self.player_lookup[captain_id]["player_name"] if captain_id else "None"
                    
                    # Create new state
                    new_state = {
                        "team": new_team,
                        "transfers_available": next_available,
                        "total_points": state["total_points"] + weighted_points,
                        "history": state["history"] + [{
                            "gameweek": gw,
                            "transfers": transfer_details,
                            "captain": captain_name,
                            "captain_id": captain_id,
                            "vice_captain": self.player_lookup[vice_captain_id]["player_name"] if vice_captain_id else "None",
                            "vice_captain_id": vice_captain_id,
                            "best_11": best_11,
                            "points": unweighted_points,            # Unweighted points for reporting
                            "weighted_points": weighted_points,     # Weighted points separately
                            "optimal_points": optimal_points,
                            "formation": f"{formation[0]}-{formation[1]}-{formation[2]}",
                            "team": new_team,
                            "weight": weight,
                        }]
                    }
                    
                    next_beam.append(new_state)
            
            # Keep only the top beam_width states
            next_beam.sort(key=lambda x: x["total_points"], reverse=True)
            beam = next_beam[:self.beam_width]
            
            if self.debug and beam:
                print(f"DEBUG: Best state after GW{gw} has {beam[0]['total_points']:.2f} weighted points")

        best_solution = max(beam, key=lambda x: x["total_points"])
        
        # Calculate actual (unweighted) total points
        actual_total_points = sum(plan["points"] for plan in best_solution["history"])
        
        result = {
            "team": best_solution["team"],
            "team_names": [self.player_lookup[p]["player_name"] for p in best_solution["team"]],
            "total_points": actual_total_points,
            "weighted_total": best_solution["total_points"],
            "transfer_plan": best_solution["history"],
        }
        
        if self.debug:
            print(f"\nDEBUG: Optimization complete. Total predicted points: {result['total_points']:.2f}")
            print(f"DEBUG: Weighted total used for optimization: {result['weighted_total']:.2f}")

        self.print_results(result)
        
        return result
    
    def print_results(self, results: dict):
        """
        Print the optimization results
        
        Parameters:
            results: the results dictionary returned by optimize_transfers
        """
        
        # Print transfer plan for each gameweek
        print("\nTRANSFER PLAN:")
        print("-"*40)
        
        available_transfers = self.initial_transfers
        
        for plan in results["transfer_plan"]:
            gw = plan["gameweek"]
            captain = plan["captain"]
            transfers = plan["transfers"]
            formation = plan.get("formation", "Unknown")
            gw_points = plan["points"]
            optimal_points = plan["optimal_points"]
            best_11 = plan["best_11"]
            weight = plan.get("weight", 1.0)
            
            # Calculate available transfers for this gameweek
            if len(transfers) > 0:
                used_transfers = len(transfers)
                next_available = max(1, available_transfers - used_transfers + 1)
            else:
                used_transfers = 0
                next_available = min(available_transfers + 1, self.max_transfers)  # Bank a transfer
            
            print(f"\nGAMEWEEK {gw} (Weight: {weight:.2f}):")
            print(f"Available transfers: {available_transfers}")
            print(f"Formation: {formation}")
            print(f"Predicted points: {gw_points:.2f}")
            print(f"Optimal team points: {optimal_points:.2f}")
            print(f"POINTS DIFFERENCE: {optimal_points - gw_points:.2f} (Potential chip value)")
            
            if used_transfers > 0:
                print(f"\nTransfer actions: Making {used_transfers} transfer{'s' if used_transfers > 1 else ''}")
                for t in transfers:
                    points_impact = f"{t['points_gain']:+.1f}" if 'points_gain' in t else "Unknown"
                    print(f"  OUT: {t['out']} ({t['out_id']}) → IN: {t['in']} ({t['in_id']}) (Individual points impact: {points_impact})")
            else:
                print("Transfer actions: Saving transfer")
            
            print(f"Captain: {captain} ({plan['captain_id']})")
            print(f"Vice Captain: {plan.get('vice_captain', 'None')} ({plan.get('vice_captain_id', 'None')})")
            starters = [pid for pid in best_11 if pid is not None]
            bench = [pid for pid in plan["team"] if pid not in best_11 and pid is not None]
            
            starting_gks = []
            starting_defs = []
            starting_mids = []
            starting_fwds = []
            
            for pid in best_11:
                if pid:
                    player = self.player_lookup[pid]
                    pts = player.get(f"{self.points_prefix}{gw}", 0)
                    player_str = f"{player['player_name']} ({pid})"
                    if pid == plan["captain_id"]:
                        player_str += " (C)"
                    elif pid == plan.get("vice_captain_id"):
                        player_str += " (VC)"
                    
                    if player["position"] == "Goalkeeper":
                        starting_gks.append(player_str)
                    elif player["position"] == "Defender":
                        starting_defs.append(player_str)
                    elif player["position"] == "Midfielder":
                        starting_mids.append(player_str)
                    elif player["position"] == "Forward":
                        starting_fwds.append(player_str)
            
            bench_names = []
            for player_id in plan["team"]:
                if player_id not in best_11 and player_id:
                    bench_names.append(f"{self.player_lookup[player_id]['player_name']} ({player_id})")
            
            print("\nStarting XI:")
            print(f"  Goalkeeper: {', '.join(starting_gks)}")
            print(f"  Defender: {', '.join(starting_defs)}")
            print(f"  Midfielder: {', '.join(starting_mids)}")
            print(f"  Forward: {', '.join(starting_fwds)}")
            
            print("Bench:")
            print(f"  {', '.join(bench_names)}")
            
            available_transfers = next_available
        
        print("\n" + "-"*40)
        print(f"Total Predicted Points: {results['total_points']:.1f}")
        print(f"Total Weighted Points: {results['weighted_total']:.1f}")

        print("-"*40)
        print("\nFinal Team:")
        gks = [p for p in results["team"] if self.player_lookup[p]["position"] == "Goalkeeper"]
        defs = [p for p in results["team"] if self.player_lookup[p]["position"] == "Defender"]
        mids = [p for p in results["team"] if self.player_lookup[p]["position"] == "Midfielder"]
        fwds = [p for p in results["team"] if self.player_lookup[p]["position"] == "Forward"]
        
        print(f"Goalkeeper: {', '.join([self.player_lookup[p]['player_name'] for p in gks])}")
        print(f"Defender: {', '.join([self.player_lookup[p]['player_name'] for p in defs])}")
        print(f"Midfielder: {', '.join([self.player_lookup[p]['player_name'] for p in mids])}")
        print(f"Forward: {', '.join([self.player_lookup[p]['player_name'] for p in fwds])}")

        print("\nInitial Team:")
        gks = [p for p in self.current_team if self.player_lookup[p]["position"] == "Goalkeeper"]
        defs = [p for p in self.current_team if self.player_lookup[p]["position"] == "Defender"]
        mids = [p for p in self.current_team if self.player_lookup[p]["position"] == "Midfielder"]
        fwds = [p for p in self.current_team if self.player_lookup[p]["position"] == "Forward"]
        
        print(f"Goalkeeper: {', '.join([self.player_lookup[p]['player_name'] for p in gks])}")
        print(f"Defender: {', '.join([self.player_lookup[p]['player_name'] for p in defs])}")
        print(f"Midfielder: {', '.join([self.player_lookup[p]['player_name'] for p in mids])}")
        print(f"Forward: {', '.join([self.player_lookup[p]['player_name'] for p in fwds])}")


class GreedyFPLTransferOptimizer:
    def __init__(
            self,
            players_df: pd.DataFrame,
            current_team: list,
            points_prefix: str = "Points_GW", 
            team_size: int = 15,
            max_transfers: int = 5,
            max_per_team: int = 3,
            budget: int = 1000, 
            initial_transfers: int = 1,
            max_threshold: float = 0.3,
            min_threshold: float = 0.05,
            threshold_step: float = 0.05,
            top_transfers_to_track: int = 5,
            must_have_threshold: float = 0.15,
            future_min_threshold: float = 0.02,
            future_weights: dict | None = None,
            valid_status: list[str] = ["a", "d"],
            debug: bool = False,
    ):
        """       
        Parameters:
            players_df: Wide-format dataframe with player information and predicted points            
            current_team: List of player_ids in the current team            
            points_prefix: Prefix for the predicted points columns in the dataframe            
            team_size: Number of players in a team (default: 15)            
            max_transfers: Maximum number of transfers allowed at once (default: 5)            
            max_per_team: Maximum players allowed from a single team (default: 3)            
            budget: Total budget constraint (default: 1000, which is £100.0m in FPL)            
            initial_transfers: Number of transfers available at the start            
            max_threshold: Starting threshold for improvement (default: 0.3 or 30%)            
            min_threshold: Minimum threshold to consider (default: 0.1 or 10%)            
            threshold_step: Amount to reduce threshold by if no transfers found (default: 0.05 or 5%)            
            top_transfers_to_track: Number of top transfers to track for potential combinations            
            must_have_threshold: Threshold above which transfers are considered "must-have" (default: 0.15 or 15%)            
            future_min_threshold: Minimum threshold for considering future impact (default: 0.02 or 2%)            
            future_weights: Weights for future gameweeks (default: linearly decreasing by 0.2)            
            debug: Whether to print debug information            
        """
        self.players_df = players_df
        self.points_prefix = points_prefix
        self.team_size = team_size
        self.max_transfers = max_transfers
        self.max_per_team = max_per_team
        self.budget = budget
        self.initial_transfers = initial_transfers
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.threshold_step = threshold_step
        self.top_transfers_to_track = top_transfers_to_track
        self.must_have_threshold = must_have_threshold
        self.future_min_threshold = future_min_threshold
        self.valid_status = valid_status
        self.debug = debug
        
        # Extract all gameweeks in the prediction horizon
        self.gameweeks = sorted([int(col.replace(points_prefix, '')) 
                                for col in players_df.columns 
                                if col.startswith(points_prefix)])
        
        # Set future weights if not provided
        if future_weights is None:
            max_future_gws = len(self.gameweeks) - 1
            self.future_weights = {i: max(0, 1.0 - (0.2 * i)) for i in range(max_future_gws + 1)}
        else:
            self.future_weights = future_weights
        
        self.player_lookup = {}
        for player_id, row in players_df.set_index("player_id").iterrows():
            player_dict = row.to_dict()
            player_dict["player_id"] = player_id
            self.player_lookup[player_id] = player_dict

        self.current_team = current_team.copy()

        if len(self.current_team) != 15:
            raise ValueError(f"Current team must have exactly 15 players, got {len(self.current_team)}")

        # Validate all current team players exist in the data
        missing_players = [int(p) for p in self.current_team if p not in self.player_lookup]
        if missing_players:
            self.players_df, found_all = backfill_missing_players(
                df=self.players_df,
                missing_players=missing_players,
                points_prefix=self.points_prefix
            )

            if not found_all:
                raise ValueError(f"Players not found in data: {missing_players}")

            # Add backfilled players to the lookup
            new_rows = self.players_df[self.players_df["player_id"].isin(missing_players)]
            for player_id, row in new_rows.set_index("player_id").iterrows():
                player_dict = row.to_dict()
                player_dict["player_id"] = player_id
                self.player_lookup[player_id] = player_dict

        # When finding valid players for transfers, filter by status
        self.valid_transfer_players = {
            player_id: player for player_id, player in self.player_lookup.items()
            if player.get("status", "u") in valid_status
        }

        # Pre-filter by position
        self.players_by_position = {
            "Goalkeeper": [],
            "Defender": [],
            "Midfielder": [],
            "Forward": [],
        }
        for player_id, player in self.valid_transfer_players.items():
            self.players_by_position[player["position"]].append((player_id, player))
        
        # Detect the minimum available gameweek
        self.current_gw = min(self.gameweeks)

    def select_best_starting_11(self, team: list, gameweek : int) -> tuple[list, float, tuple, int]:
        """
        Selects the best starting 11 players and captain based on valid FPL formations
        
        Parameters:
            team: list of player_ids in the current 15-man squad            
            gameweek: the gameweek for which we are selecting the best team
        
        Returns:
            - List of the best 11 player_ids (with first player as captain)
            - Total predicted points for this best 11 (including doubled captain points)
            - Formation used (Defender, Midfielder, Forward)
            - Vice-captain player_id
        """
        valid_formations = [
            (5, 4, 1), (5, 3, 2),  # Five defenders formations
            (4, 4, 2), (4, 3, 3),  # Four defenders formations
            (3, 5, 2), (3, 4, 3)   # Three defenders formations
        ]
        
        points_column = f"{self.points_prefix}{gameweek}"
        # Extract player data and filter out players with NaN points for this gameweek
        all_players = []
        players_with_nan = []

        for p in team:
            player_data = self.player_lookup[p].copy()
            points_value = player_data.get(points_column)
            
            if pd.isna(points_value) or points_value is None:
                player_data[points_column] = 0.0
                players_with_nan.append(player_data["player_name"])
                if self.debug:
                    print(f"DEBUG: Player {player_data['player_name']} has NaN points, treating as 0")
            
            all_players.append(player_data)
        
        if self.debug and players_with_nan:
            print(f"DEBUG: {len(players_with_nan)} players have NaN points for GW{gameweek}: {players_with_nan}")

        # Separate players by position
        gks = sorted([p for p in all_players if p["position"] == "Goalkeeper"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        defs = sorted([p for p in all_players if p["position"] == "Defender"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        mids = sorted([p for p in all_players if p["position"] == "Midfielder"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        fwds = sorted([p for p in all_players if p["position"] == "Forward"], 
                    key=lambda p: p.get(points_column, 0), reverse=True)
        
        if self.debug:
            print(f"DEBUG: Position breakdown - GK: {len(gks)}, DEF: {len(defs)}, MID: {len(mids)}, FWD: {len(fwds)}")

        if len(gks) < 1 or len(defs) < 3 or len(mids) < 2 or len(fwds) < 1:
            print(f"WARNING: Cannot form standard FPL formation")
            print(f"WARNING: Have GK: {len(gks)}, DEF: {len(defs)}, MID: {len(mids)}, FWD: {len(fwds)}")

            selected_players = []

            if gks:
                selected_players.extend(gks[:1])
            
            available_def_slots = min(len(defs), 5)
            if defs:
                selected_players.extend(defs[:available_def_slots])

            available_mid_slots = min(len(mids), 5)
            if mids:
                selected_players.extend(mids[:available_mid_slots])

            available_fwd_slots = min(len(fwds), 3)
            if fwds:
                selected_players.extend(fwds[:available_fwd_slots])

            # If we still don't have 11 players, pad with remaining best players by position
            while len(selected_players) < 11:
                # Try to add more players respecting position limits
                current_gks = len([p for p in selected_players if p["position"] == "Goalkeeper"])
                current_defs = len([p for p in selected_players if p["position"] == "Defender"])
                current_mids = len([p for p in selected_players if p["position"] == "Midfielder"])
                current_fwds = len([p for p in selected_players if p["position"] == "Forward"])
                
                added_player = False
                
                # Try to add defender if we have room and players
                if current_defs < 5 and len(defs) > current_defs:
                    selected_players.append(defs[current_defs])
                    added_player = True
                # Try to add midfielder if we have room and players  
                elif current_mids < 5 and len(mids) > current_mids:
                    selected_players.append(mids[current_mids])
                    added_player = True
                # Try to add forward if we have room and players
                elif current_fwds < 3 and len(fwds) > current_fwds:
                    selected_players.append(fwds[current_fwds])
                    added_player = True
                
                if not added_player:
                    break  # Can't add any more players

            total_points = 0
            captain = max(selected_players, key=lambda p: p.get(points_column, 0)) if selected_players else None
            remaining_players = [p for p in selected_players if p != captain] if captain else selected_players
            vice_captain = max(remaining_players, key=lambda p: p.get(points_column, 0)) if remaining_players else None

            for player in selected_players:
                points = player.get(points_column, 0)
                if player == captain:
                    points *= 2
                total_points += points

            if captain:
                captain_id = captain["player_id"]
                team_ids = [p["player_id"] for p in selected_players if p["player_id"] != captain_id]
                if vice_captain:
                    vice_captain_id = vice_captain["player_id"]
                    if vice_captain_id in team_ids:
                        team_ids.remove(vice_captain_id)
                    best_team = [captain_id, vice_captain_id] + team_ids
                else:
                    best_team = [captain_id] + team_ids
            else:
                best_team = [p["player_id"] for p in selected_players]

            # Pad with None if we couldn't get 11 players
            while len(best_team) < 11:
                best_team.append(None)

            vice_captain_id = vice_captain["player_id"] if vice_captain else None
            return best_team, total_points, (0, 0, 0), vice_captain_id
        
        formation_points = {}
        best_team = []
        best_points = float("-inf")
        best_formation = None
        vice_captain_id = None

        for formation in valid_formations:
            num_def, num_mid, num_fwd = formation

            # Ensure we have enough players for the formation
            if len(gks) < 1 or len(defs) < num_def or len(mids) < num_mid or len(fwds) < num_fwd:
                continue  # Skip invalid formations

            # Select the best players for this formation
            selected_gk = [gks[0]]
            selected_defs = defs[:num_def]
            selected_mids = mids[:num_mid]
            selected_fwds = fwds[:num_fwd]
            
            # Combine all selected players
            selected_players = selected_gk + selected_defs + selected_mids + selected_fwds
            
            # Sort players by points to get captain and vice captain
            sorted_by_points = sorted(selected_players, 
                                    key=lambda p: p.get(f"{self.points_prefix}{gameweek}", 0), 
                                    reverse=True)

            # Get captain (highest points) and vice captain (second highest)
            captain = sorted_by_points[0]
            vice_captain = sorted_by_points[1] if len(sorted_by_points) > 1 else None
            
            # Calculate total predicted points (with captain doubled)
            total_points = 0
            for player in selected_players:
                points = player.get(f"{self.points_prefix}{gameweek}", 0)
                if player == captain:
                    points *= 2
                total_points += points
                
            # Store formation points for debugging
            formation_str = f"{num_def}-{num_mid}-{num_fwd}"
            formation_points[formation_str] = total_points
            
            selected_team = [p["player_id"] for p in selected_players]
            
            # Update the best team if this formation gives more points
            if total_points > best_points:
                best_points = total_points
                captain_id = captain["player_id"]
                vice_captain_id = vice_captain["player_id"] if vice_captain else None

                selected_team = [p["player_id"] for p in selected_players]
                selected_team.remove(captain_id)
                if vice_captain_id and vice_captain_id in selected_team:
                    selected_team.remove(vice_captain_id)
                    best_team = [captain_id, vice_captain_id] + selected_team
                else:
                    best_team = [captain_id] + selected_team
                best_formation = formation
        
        # Debug output for formation comparison
        if self.debug and formation_points:
            print("DEBUG: Points by formation (including captain bonus):")
            for form, points in sorted(formation_points.items(), key=lambda x: x[1], reverse=True):
                star = " *" if form == f"{best_formation[0]}-{best_formation[1]}-{best_formation[2]}" else ""
                print(f"DEBUG:   {form}: {points:.2f} pts{star}")

        # Final check - if we still couldn't form a valid team
        if not best_team or best_points <= 0:
            print(f"ERROR: Could not form any valid team for gameweek {gameweek}")
            best_team = [None] * 11
            best_points = 0.0
            best_formation = (0, 0, 0)

        return best_team, best_points, best_formation, vice_captain_id
    
    def find_optimal_team_for_gameweek(self, gameweek):
        """
        Use PuLP to find the optimal team for a gameweek from all available players
            
        Returns:
            List of player_ids in the optimal team
        """
        model = LpProblem(name="Optimal_FPL_Team", sense=LpMaximize)
        
        # Filter players with valid points for this gameweek
        valid_players = []
        for _, player in self.valid_transfer_players.items():
            if pd.notna(player.get(f"{self.points_prefix}{gameweek}", 0)):
                valid_players.append(player)
        
        player_vars = {}
        for player in valid_players:
            player_vars[player["player_id"]] = LpVariable(f"player_{player['player_id']}", cat="Binary")
        
        # Objective: Maximize total expected points
        model += lpSum([player_vars[player["player_id"]] * player.get(f"{self.points_prefix}{gameweek}", 0) 
                         for player in valid_players])
        
        # Budget constraint
        model += lpSum([player_vars[player["player_id"]] * player["now_cost"] for player in valid_players]) <= self.budget
        
        # Team size constraint (only 11 for the optimal team as we only care about starters)
        model += lpSum([player_vars[player["player_id"]] for player in valid_players]) == 11
        
        # Position constraints
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Goalkeeper"]) == 1
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Defender"]) >= 3
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Defender"]) <= 5
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Midfielder"]) >= 2
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Midfielder"]) <= 5
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Forward"]) >= 1
        model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                        if player["position"] == "Forward"]) <= 3
        
        # Team constraint (max 3 players from same team)
        for team_id in set(p["team_id"] for p in valid_players):
            model += lpSum([player_vars[player["player_id"]] for player in valid_players 
                            if player["team_id"] == team_id]) <= self.max_per_team
            
        model.solve(PULP_CBC_CMD(msg=False))

        if model.status != LpStatusOptimal:
            if self.debug:
                print(f"DEBUG: PuLP could not find optimal solution for GW{gameweek}, status: {model.status}")
            return [], 0  # Return empty team and 0 points
        
        optimal_team = [player["player_id"] for player in valid_players 
                        if player_vars[player["player_id"]].value() == 1]
        
        if self.debug:
            print(f"\nOptimal team found by PuLP for GW {gameweek}:")
            for pid in optimal_team:
                player = self.player_lookup[pid]
                print(f"  {player['player_name']} ({player['position']}) - {player.get(f'{self.points_prefix}{gameweek}', 0):.2f} pts")
        
        return optimal_team

    def evaluate_transfer_impact(
            self,
            current_team: list,
            out_player: int,
            in_player: int,
            gameweek: int,
            eval_future: bool = False,
    ) -> tuple:
        """
        Evaluate the impact of a single transfer on the team's total points, optionally including future gameweeks
        
        Parameters:
            current_team: current 15-player team
            out_player: player_id to transfer out
            in_player: player_id to transfer in
            gameweek: current gameweek
            eval_future: whether to evaluate future gameweeks impact
            
        Returns:
            If eval_future=False: (new team, old points, new points, current_improvement)
            If eval_future=True: (new team, old points, new points, current_improvement, weighted_total_improvement)
        """
        # Calculate points for current team
        _, old_points, _, _ = self.select_best_starting_11(current_team, gameweek)
        
        # Create new team with the transfer
        new_team = [p if p != out_player else in_player for p in current_team]
        
        # Get player_names for debugging
        out_name = self.player_lookup[out_player]["player_name"]
        in_name = self.player_lookup[in_player]["player_name"]
        
        # Check if the new team is valid
        if not self.is_valid_team(new_team):
            total_cost = sum(self.player_lookup[p]["now_cost"] for p in new_team)
            
            if self.debug:
                if total_cost > self.budget:
                    print(f"DEBUG: Transfer {out_name} → {in_name} invalid - budget exceeded: {total_cost} > {self.budget}")
            
            # Check position constraints
            positions = {"Goalkeeper": 0, "Defender": 0, "Midfielder": 0, "Forward": 0}
            for player_id in new_team:
                pos = self.player_lookup[player_id]["position"]
                positions[pos] = positions.get(pos, 0) + 1
            
            expected = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
            for pos, count in positions.items():
                if count != expected[pos]:
                    if self.debug:
                        print(f"DEBUG: Transfer {out_name} → {in_name} invalid - position constraint: {pos} has {count} players (expected {expected[pos]})")
            
            # Check team composition
            team_composition = {}
            for player_id in new_team:
                team_id = self.player_lookup[player_id]["team_id"]
                team_composition[team_id] = team_composition.get(team_id, 0) + 1
                if team_composition[team_id] > self.max_per_team:
                    if self.debug:
                        print(f"DEBUG: Transfer {out_name} → {in_name} invalid - too many players from team {team_id}: {team_composition[team_id]}")
            
            if eval_future:
                return new_team, old_points, old_points, 0, 0
            else:
                return new_team, old_points, old_points, 0
        
        # Calculate points for new team
        _, new_points, _, _ = self.select_best_starting_11(new_team, gameweek)
        
        # Calculate percentage improvement
        pct_improvement = ((new_points / old_points) - 1) if old_points > 0 else float("inf")
        
        # Debug output for significant improvements
        if pct_improvement > 0.05:  # Only log transfers with >5% improvement to reduce noise
            if self.debug:
                print(f"DEBUG: Transfer {out_name} → {in_name} yields {pct_improvement*100:.1f}% improvement")
                print(f"DEBUG:   Old points: {old_points:.2f}, New points: {new_points:.2f}")
        
        # If we're not evaluating future gameweeks, return current impact only
        if not eval_future:
            return new_team, old_points, new_points, pct_improvement
            
        # Calculate weighted future impact
        future_impact = self.evaluate_future_impact(current_team, out_player, in_player, gameweek)
        
        # Calculate total weighted impact (current + future)
        total_weighted_impact = pct_improvement + future_impact
        
        if future_impact > 0.05:  # Only log significant future impacts
            if self.debug:
                print(f"DEBUG:   Future weighted impact: {future_impact:.3f}")
                print(f"DEBUG:   Total weighted impact: {total_weighted_impact:.3f}")
            
        return new_team, old_points, new_points, pct_improvement, total_weighted_impact
        
    def evaluate_future_impact(self, current_team: list, out_player: int, in_player: int, current_gw: int) -> float:
        """
        Evaluate the impact of a transfer on future gameweeks
        
        Parameters:
            current_team: current 15-player team
            out_player: player_id to transfer out
            in_player: player_id to transfer in
            current_gw: current gameweek being evaluated
            
        Returns:
            Weighted future impact score
        """
        # Create new team with the transfer
        new_team = [p if p != out_player else in_player for p in current_team]
        
        # Get player_names for debugging
        out_name = self.player_lookup[out_player]["player_name"]
        in_name = self.player_lookup[in_player]["player_name"]
        
        weighted_impact = 0
        future_impacts = {}
        
        # Evaluate each future gameweek
        current_idx = self.gameweeks.index(current_gw)
        
        for i in range(current_idx + 1, len(self.gameweeks)):
            future_gw = self.gameweeks[i]
            future_offset = i - current_idx
            
            weight = self.future_weights.get(future_offset, 0)
            
            # Calculate impact on this future gameweek
            try:
                _, old_points, _, _ = self.select_best_starting_11(current_team, future_gw)
                _, new_points, _, _ = self.select_best_starting_11(new_team, future_gw)
                
                # Calculate percentage improvement for future gameweek
                if old_points > 0:
                    pct_improvement = (new_points / old_points) - 1
                else:
                    pct_improvement = float("inf") if new_points > 0 else 0
                    
                # Store impact for this gameweek
                future_impacts[future_gw] = pct_improvement
                    
                # Add weighted contribution
                weighted_impact += weight * pct_improvement
            except Exception as e:
                print(f"DEBUG: Error evaluating future impact for GW{future_gw}: {str(e)}")
                continue
                
        return weighted_impact

    def is_valid_team(self, team: list) -> bool:
        """
        Check if a team is valid according to FPL rules
        
        Parameters:
            team: list of player_ids
            
        Returns:
            True if team is valid, False otherwise
        """
        if len(team) != self.team_size:
            return False
        
        # Check budget constraint
        total_cost = sum(self.player_lookup[p]["now_cost"] for p in team)
        if total_cost > self.budget:
            return False
        
        # Check position constraints
        positions = {"Goalkeeper": 0, "Defender": 0, "Midfielder": 0, "Forward": 0}
        for player_id in team:
            pos = self.player_lookup[player_id]["position"]
            positions[pos] = positions.get(pos, 0) + 1
        
        if positions.get("Goalkeeper", 0) != 2 or positions.get("Defender", 0) != 5 or \
           positions.get("Midfielder", 0) != 5 or positions.get("Forward", 0) != 3:
            return False
        
        # Check team composition by team_id
        team_composition = {}
        for player_id in team:
            team_id = self.player_lookup[player_id]["team_id"]
            team_composition[team_id] = team_composition.get(team_id, 0) + 1
            if team_composition[team_id] > self.max_per_team:
                return False
        
        return True
    
    def find_best_transfers(self, current_team: list, optimal_team: list, available_transfers: int, gameweek: int):
        """
        Find the best transfers to make based on team-level improvement.
        Uses a dynamic threshold approach with future impact evaluation.
        
        Parameters:
            current_team: current 15-player team
            optimal_team: optimal 11-player team for the gameweek
            available_transfers: number of transfers available
            gameweek: current gameweek
            
        Returns:
            (list of (out, in) transfers, new team after transfers)
        """
        if available_transfers == 0:
            print("DEBUG: No transfers available for this gameweek")
            return [], current_team
        
        # Get current best 11 and their points
        current_best_11, current_points, _, _ = self.select_best_starting_11(current_team, gameweek)
        
        if self.debug:
            print(f"DEBUG: Current team's best 11 points: {current_points:.2f}")
            print(f"DEBUG: Team value: {sum(self.player_lookup[p]['now_cost'] for p in current_team)}")
            print(f"DEBUG: Budget: {self.budget}")
        
        # Identify players in common (these we might not prioritize transferring)
        common_players = set(current_best_11).intersection(set(optimal_team))
        
        if self.debug:
            print(f"DEBUG: Players in common with optimal team: {len(common_players)}")
            print(f"DEBUG: Common players: {[self.player_lookup[p]['player_name'] for p in common_players]}")
        
            print(f"DEBUG: Evaluating all possible transfers for gameweek {gameweek}...")
        
        # Dictionary to store all evaluated transfers
        high_impact_transfers = {}      # For transfers above must_have_threshold
        moderate_impact_transfers = {}  # For transfers that need future evaluation
        
        # Track evaluation counts
        evaluated_count = 0
        
        # Try all possible transfers for all players in the team
        for out_player in current_team:
            out_position = self.player_lookup[out_player]["position"]
            out_name = self.player_lookup[out_player]["player_name"]
            
            # Consider all players in the same position
            for in_player_id, in_player in self.players_by_position[out_position]:
                # Skip if player is already in team
                if in_player_id in current_team:
                    continue
                    
                # Skip if NaN points for this gameweek
                in_points = in_player.get(f"{self.points_prefix}{gameweek}", 0)
                if pd.isna(in_points):
                    in_points = 0.0
                
                evaluated_count += 1
                
                # First evaluate current gameweek impact
                new_team, _, new_points, pct_improvement = self.evaluate_transfer_impact(
                    current_team, out_player, in_player_id, gameweek, eval_future=False
                )
                
                # Check if this is a high-impact transfer (must-have)
                if pct_improvement >= self.must_have_threshold:
                    high_impact_transfers[(out_player, in_player_id)] = (new_points, pct_improvement)
                
                # If not high-impact but above future min threshold, evaluate future impact
                elif pct_improvement >= self.future_min_threshold:
                    # Get the full evaluation including future gameweeks
                    _, _, _, _, total_weighted_impact = self.evaluate_transfer_impact(
                        current_team, out_player, in_player_id, gameweek, eval_future=True
                    )
                    
                    # Store with total weighted impact
                    moderate_impact_transfers[(out_player, in_player_id)] = (new_points, pct_improvement, total_weighted_impact)
        
        if self.debug:
            print(f"DEBUG: Evaluated {evaluated_count} potential transfers")
        
        # No transfers found at all
        if not high_impact_transfers and not moderate_impact_transfers:
            print("DEBUG: No potential transfers found.")
            return [], current_team
        
        # Process high-impact transfers first
        if high_impact_transfers:
            if self.debug:
                print(f"DEBUG: Found {len(high_impact_transfers)} high-impact transfers (>= {self.must_have_threshold*100:.1f}%)")
            
            # Sort by percentage improvement
            sorted_transfers = sorted(
                high_impact_transfers.items(),
                key=lambda x: x[1][1],
                reverse=True
            )
            
            for i, ((out_id, in_id), (_, pct)) in enumerate(sorted_transfers[:5]):
                out_name = self.player_lookup[out_id]["player_name"]
                in_name = self.player_lookup[in_id]["player_name"]
                if self.debug:
                    print(f"DEBUG:   {i+1}. {out_name} → {in_name} (Improvement: {pct*100:.1f}%)")
            
            best_transfer_combo = self.find_best_transfer_combination(
                current_team, [t[0] for t in sorted_transfers[:self.top_transfers_to_track]], 
                available_transfers, gameweek, use_future_impact=False
            )
            
            if best_transfer_combo:
                if self.debug:
                    print(f"DEBUG: Selected high-impact transfers for gameweek {gameweek}:")
                for out_id, in_id in best_transfer_combo:
                    out_name = self.player_lookup[out_id]["player_name"]
                    in_name = self.player_lookup[in_id]["player_name"]
                    if self.debug:
                        print(f"DEBUG:   {out_name} → {in_name}")
                
                # Apply the transfers
                new_team = current_team.copy()
                for out_id, in_id in best_transfer_combo:
                    new_team = [p if p != out_id else in_id for p in new_team]
                    
                return best_transfer_combo, new_team
        
        # If no high-impact transfers or no valid combination found, check moderate-impact with future value
        if self.debug:
            print(f"DEBUG: Evaluating {len(moderate_impact_transfers)} moderate-impact transfers with future impact")
        
        # Sort by total weighted impact (current + future)
        sorted_transfers = sorted(
            moderate_impact_transfers.items(),
            key=lambda x: x[1][2],
            reverse=True
        )

        for i, ((out_id, in_id), (_, current_pct, total_impact)) in enumerate(sorted_transfers[:5]):
            out_name = self.player_lookup[out_id]["player_name"]
            in_name = self.player_lookup[in_id]["player_name"]
            if self.debug:
                print(f"DEBUG:   {i+1}. {out_name} → {in_name} (Current: {current_pct*100:.1f}%, Total weighted: {total_impact:.3f})")
        
        best_transfer_combo = self.find_best_transfer_combination(
            current_team, [t[0] for t in sorted_transfers[:self.top_transfers_to_track]], 
            available_transfers, gameweek, use_future_impact=True
        )
        
        if best_transfer_combo:
            if self.debug:
                print(f"DEBUG: Selected transfers with future impact for gameweek {gameweek}:")
            for out_id, in_id in best_transfer_combo:
                out_name = self.player_lookup[out_id]["player_name"]
                in_name = self.player_lookup[in_id]["player_name"]
                if self.debug:
                    print(f"DEBUG:   {out_name} → {in_name}")
            
            # Apply the transfers
            new_team = current_team.copy()
            for out_id, in_id in best_transfer_combo:
                new_team = [p if p != out_id else in_id for p in new_team]
                
            return best_transfer_combo, new_team
        
        print(f"DEBUG: No beneficial transfers found for gameweek {gameweek}")
        return [], current_team

    def find_best_transfer_combination(
            self,
            current_team: list,
            candidate_transfers: list,
            max_transfers: int,
            gameweek: int,
            use_future_impact: bool = False,
    ) -> list:
        """
        Find the best combination of transfers from a list of candidates.
        
        Parameters:
            current_team: current 15-player team
            candidate_transfers: list of (out, in) transfer candidates
            max_transfers: maximum number of transfers allowed
            gameweek: current gameweek
            use_future_impact: whether to include future gameweek impact in evaluation
            
        Returns:
            Best combination of (out, in) transfers
        """
        if not candidate_transfers:
            print("DEBUG: No candidate transfers to evaluate combinations")
            return []
            
        # Get current points as baseline
        _, baseline_points, _, _ = self.select_best_starting_11(current_team, gameweek)
        
        if self.debug:
            print(f"DEBUG: Exploring combinations of transfers for gameweek {gameweek}")
            print(f"DEBUG: Baseline points: {baseline_points:.2f}")
            print(f"DEBUG: Number of candidate transfers: {len(candidate_transfers)}")
            print(f"DEBUG: Maximum transfers allowed: {max_transfers}")
            print(f"DEBUG: Using future impact: {use_future_impact}")
        
        best_transfers = []
        best_improvement = 0
        total_combinations_tried = 0
        
        # Try different numbers of transfers
        for num_transfers in range(1, min(len(candidate_transfers), max_transfers) + 1):
            if self.debug:
                print(f"DEBUG: Trying combinations of {num_transfers} transfers")
                     
            combo_count = 0
            valid_combo_count = 0
            
            for combo in combinations(candidate_transfers, num_transfers):
                combo_count += 1
                total_combinations_tried += 1
                
                # Check if the combination has overlapping players
                out_players = [out for out, _ in combo]
                in_players = [in_id for _, in_id in combo]
                
                # Skip if any player is involved in multiple transfers
                if len(set(out_players)) != len(out_players) or len(set(in_players)) != len(in_players):
                    continue
                
                # Apply the transfers
                temp_team = current_team.copy()
                for out_id, in_id in combo:
                    if out_id in temp_team:
                        index = temp_team.index(out_id)
                        temp_team[index] = in_id
                    else:
                        continue  # Skip invalid transfer

                # Verify no duplicates
                if len(temp_team) != len(set(temp_team)):
                    duplicates = [pid for pid in set(temp_team) if temp_team.count(pid) > 1]
                    print(f"ERROR: Greedy transfer created duplicates!")
                    print(f"ERROR: Original team: {current_team}")
                    print(f"ERROR: Transfers applied: {combo}")
                    print(f"ERROR: Resulting team: {temp_team}")
                    print(f"ERROR: Duplicates: {duplicates}")
                    continue
                
                # Check if the team is valid
                if not self.is_valid_team(temp_team):
                    continue
                
                valid_combo_count += 1
                
                # Calculate points for the new team
                _, new_points, _, _ = self.select_best_starting_11(temp_team, gameweek)
                
                # Calculate improvement
                improvement = new_points - baseline_points
                
                # Include future impact if requested
                if use_future_impact:
                    future_impact = 0
                    for out_id, in_id in combo:
                        # Get individual future impact of this transfer
                        individual_future_impact = self.evaluate_future_impact(current_team, out_id, in_id, gameweek)
                        future_impact += individual_future_impact
                    
                    # Add future impact
                    improvement += future_impact * baseline_points
                
                # Update best transfers if this combo is better
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_transfers = list(combo)
                    
                    # Debug output for new best combination
                    combo_desc = ", ".join([f"{self.player_lookup[out_id]['player_name']} → {self.player_lookup[in_id]['player_name']}" 
                                            for out_id, in_id in combo])
                    if self.debug:
                        print(f"DEBUG: New best combination found: {combo_desc}")
                    improvement_pct = (improvement/baseline_points)*100 if baseline_points > 0 else 0
                    # Get current points as baseline
                    _, baseline_points, _, _ = self.select_best_starting_11(current_team, gameweek)
                
                    if baseline_points == 0:
                        print(f"DEBUG: WARNING - baseline_points is 0 for gameweek {gameweek}")
                        print(f"DEBUG: Current team: {[self.player_lookup[p]['player_name'] for p in current_team]}")
                        print(f"DEBUG: Team has {len(current_team)} players")
                        players_with_points = [p for p in current_team if pd.notna(self.player_lookup[p].get(f"{self.points_prefix}{gameweek}", 0))]
                        print(f"DEBUG: Players with valid points for GW{gameweek}: {len(players_with_points)}")

                    if self.debug:
                        print(f"DEBUG: Improvement: {improvement:.2f} points ({improvement_pct:.1f}%)")
                    if use_future_impact and self.debug:
                        print(f"DEBUG: Includes future impact contribution")
            
            if self.debug:
                print(f"DEBUG: Evaluated {combo_count} combinations of {num_transfers} transfers")
                print(f"DEBUG: Found {valid_combo_count} valid combinations")
        
        if self.debug:
            print(f"DEBUG: Total combinations tried: {total_combinations_tried}")

        if best_transfers:
            if self.debug:
                print(f"DEBUG: Best transfer combination selected: {len(best_transfers)} transfers")
            for out_id, in_id in best_transfers:
                out_name = self.player_lookup[out_id]["player_name"]
                in_name = self.player_lookup[in_id]["player_name"]
                if self.debug:
                    print(f"DEBUG:   {out_name} → {in_name}")
            if self.debug:
                print(f"DEBUG: Total improvement: {best_improvement:.2f} points ({(best_improvement/baseline_points)*100:.1f}%)")
        else:
            if self.debug:
                print("DEBUG: No valid transfer combination found")
        
        return best_transfers

    def optimize_transfers(self):
        """
        Optimize transfers using a modified greedy approach with team-level evaluation and future impact consideration
        
        Returns:
            Optimization results with transfer plan
        """
        current_team = self.current_team.copy()
        available_transfers = self.initial_transfers
        total_points = 0
        transfer_history = []
        
        # Calculate initial team value
        initial_team_value = sum(self.player_lookup[p]["now_cost"] for p in current_team)
        if self.debug:
            print(f"DEBUG: Initial team value: {initial_team_value}")
            print(f"DEBUG: Budget available: {self.budget}")
        
        if initial_team_value > self.budget:
            print(f"DEBUG: Warning - Initial team value ({initial_team_value}) exceeds budget ({self.budget})")
        
        # Process each gameweek
        for gw in self.gameweeks:
            print(f"\nProcessing gameweek {gw}...")
            print(f"Available transfers: {available_transfers}")
            
            # Find optimal team for this gameweek
            optimal_team = self.find_optimal_team_for_gameweek(gw)
            
            # Find best transfers to make
            transfers, new_team = self.find_best_transfers(current_team, optimal_team, available_transfers, gw)
            
            # Select best 11 from the team after transfers
            best_11, gw_points, formation, vice_captain_id = self.select_best_starting_11(new_team, gw)
            captain_id = best_11[0] if best_11 and best_11[0] is not None else None
            captain_name = self.player_lookup[captain_id]["player_name"] if captain_id else "None"
            if captain_id:
                captain_points = self.player_lookup[captain_id].get(f"{self.points_prefix}{gw}", 0)
                captain_points = 0.0 if pd.isna(captain_points) else captain_points
            else:
                captain_points = 0.0

            if vice_captain_id:
                vice_captain_points = self.player_lookup[vice_captain_id].get(f"{self.points_prefix}{gw}", 0)
                vice_captain_points = 0.0 if pd.isna(vice_captain_points) else vice_captain_points
            else:
                vice_captain_points = 0.0
            vice_captain_name = self.player_lookup[vice_captain_id]["player_name"] if vice_captain_id else "None"

            if self.debug:
                print(f"Captain: {captain_name} ({captain_id}) - {captain_points:.1f} pts")
                print(f"Vice Captain: {vice_captain_name} ({vice_captain_id}) - {vice_captain_points:.1f} pts")
                
                print(f"Best formation for GW{gw}: {formation[0]}-{formation[1]}-{formation[2]}")
                print(f"Expected points: {gw_points:.2f}")
            
            # Update transfer information
            transfer_details = []
            for out_id, in_id in transfers:
                out_name = self.player_lookup[out_id]["player_name"]
                in_name = self.player_lookup[in_id]["player_name"]
                
                # Get individual player points
                out_points = self.player_lookup[out_id].get(f"{self.points_prefix}{gw}", 0)
                in_points = self.player_lookup[in_id].get(f"{self.points_prefix}{gw}", 0)
                
                # Calculate the direct points impact of this transfer
                points_gain = in_points - out_points
                
                if self.debug:
                    print(f"Transfer details - {out_name} ({out_id}) ({out_points:.2f} pts) → {in_name} ({in_id}) ({in_points:.2f} pts)")
                    print(f"Direct points gain: {points_gain:.2f}")
                
                # Store transfer details
                transfer_details.append({
                    "out": out_name,
                    "in": in_name,
                    "points_gain": points_gain,
                    "out_id": out_id,
                    "in_id": in_id
                })
            
            # Update available transfers for next gameweek
            if transfers:
                used_transfers = len(transfers)
                next_available_transfers = max(1, available_transfers - used_transfers + 1)
                if self.debug:
                    print(f"Made {used_transfers} transfers. Next GW available transfers: {next_available_transfers}")
            else:
                next_available_transfers = min(available_transfers + 1, self.max_transfers)  # Bank a transfer
                if self.debug:
                    print(f"No transfers made. Banking 1 transfer. Next GW available transfers: {next_available_transfers}")
            
            # Add to history
            transfer_history.append({
                "gameweek": gw,
                "captain": captain_name,
                "captain_id": captain_id,
                "vice_captain": self.player_lookup[vice_captain_id]["player_name"] if vice_captain_id else "None",
                "vice_captain_id": vice_captain_id,
                "num_transfers": len(transfers),
                "transfers": transfer_details,
                "formation": f"{formation[0]}-{formation[1]}-{formation[2]}",
                "points": gw_points,
                "team": new_team,
                "best_11": best_11
            })
            
            # Update running variables
            current_team = new_team
            available_transfers = next_available_transfers
            total_points += gw_points
            
            new_team_value = sum(self.player_lookup[p]["now_cost"] for p in current_team)
            if self.debug:
                print(f"Team value after GW{gw}: {new_team_value}")

        result = {
            "team": current_team,
            "team_names": [self.player_lookup[p]["player_name"] for p in current_team],
            "total_points": total_points,
            "transfer_plan": transfer_history,
        }
        
        print(f"\nOptimization complete. Total predicted points: {total_points:.2f}")
        
        self.print_results(result)
        
        return result
    
    def print_results(self, results):
        """
        Print the optimization results
        
        Parameters:
            results: the results dictionary returned by optimize_transfers
        """        
        # Print transfer plan for each gameweek
        print("\nTRANSFER PLAN:")
        print("-"*40)
        
        available_transfers = self.initial_transfers
        
        for plan in results["transfer_plan"]:
            gw = plan["gameweek"]
            captain = plan["captain"]
            transfers = plan["transfers"]
            formation = plan.get("formation", "Unknown")
            gw_points = plan["points"]
            best_11 = plan["best_11"]
            
            # Calculate available transfers for this gameweek
            if len(transfers) > 0:
                used_transfers = len(transfers)
                next_available = max(1, available_transfers - used_transfers + 1)
            else:
                used_transfers = 0
                next_available = min(available_transfers + 1, self.max_transfers)  # Bank a transfer
            
            print(f"GAMEWEEK {gw}:")
            print(f"Available transfers: {available_transfers}")
            print(f"Formation: {formation}")
            print(f"Predicted points: {gw_points:.2f}")
            
            if used_transfers > 0:
                print(f"Transfer actions: Making {used_transfers} transfer{'s' if used_transfers > 1 else ''}")
                for t in transfers:
                    points_impact = f"{t['points_gain']:+.1f}" if "points_gain" in t else "Unknown"
                    print(f"  OUT: {t['out']} ({t['out_id']}) → IN: {t['in']} ({t['in_id']}) (Individual points impact: {points_impact})")
            else:
                print("Transfer actions: Saving transfer")
            
            print(f"Captain: {captain} ({plan['captain_id']})")
            print(f"Vice Captain: {plan.get('vice_captain', 'None')} ({plan.get('vice_captain_id', 'None')})")
            starters = [pid for pid in best_11 if pid is not None]
            bench = [pid for pid in plan["team"] if pid not in best_11 and pid is not None]

            print(f"Starters: {starters}")
            print(f"Bench: {bench}")

            starting_gks = []
            starting_defs = []
            starting_mids = []
            starting_fwds = []
            
            for pid in best_11:
                if pid:
                    player = self.player_lookup[pid]
                    pts = player.get(f"{self.points_prefix}{gw}", 0)
                    player_str = f"{player['player_name']} ({pid})"
                    if pid == plan["captain_id"]:
                        player_str += " (C)"
                    elif pid == plan.get("vice_captain_id"):
                        player_str += " (VC)"
                    
                    if player["position"] == "Goalkeeper":
                        starting_gks.append(player_str)
                    elif player["position"] == "Defender":
                        starting_defs.append(player_str)
                    elif player["position"] == "Midfielder":
                        starting_mids.append(player_str)
                    elif player["position"] == "Forward":
                        starting_fwds.append(player_str)
            
            bench_names = []
            for player_id in plan["team"]:
                if player_id not in best_11 and player_id:
                    bench_names.append(f"{self.player_lookup[player_id]['player_name']} ({player_id})")
            
            print("\nStarting XI:")
            print(f"  Goalkeeper: {', '.join(starting_gks)}")
            print(f"  Defender: {', '.join(starting_defs)}")
            print(f"  Midfielder: {', '.join(starting_mids)}")
            print(f"  Forward: {', '.join(starting_fwds)}")
            
            print("Bench:")
            print(f"  {', '.join(bench_names)}")
            
            available_transfers = next_available

        print("\n" + "-"*40)
        print(f"Total predicted points: {results['total_points']:.1f}")

        # Print final team (after all transfers)
        print("-"*40)
        print("\nFinal Team:")
        gks = [p for p in results["team"] if self.player_lookup[p]["position"] == "Goalkeeper"]
        defs = [p for p in results["team"] if self.player_lookup[p]["position"] == "Defender"]
        mids = [p for p in results["team"] if self.player_lookup[p]["position"] == "Midfielder"]
        fwds = [p for p in results["team"] if self.player_lookup[p]["position"] == "Forward"]
        
        print(f"Goalkeeper: {', '.join([self.player_lookup[p]['player_name'] for p in gks])}")
        print(f"Defender: {', '.join([self.player_lookup[p]['player_name'] for p in defs])}")
        print(f"Midfielder: {', '.join([self.player_lookup[p]['player_name'] for p in mids])}")
        print(f"Forward: {', '.join([self.player_lookup[p]['player_name'] for p in fwds])}")
        
        # Print initial team (15 players)
        print("\nInitial Team:")
        gks = [p for p in self.current_team if self.player_lookup[p]["position"] == "Goalkeeper"]
        defs = [p for p in self.current_team if self.player_lookup[p]["position"] == "Defender"]
        mids = [p for p in self.current_team if self.player_lookup[p]["position"] == "Midfielder"]
        fwds = [p for p in self.current_team if self.player_lookup[p]["position"] == "Forward"]
        
        print(f"Goalkeeper: {', '.join([self.player_lookup[p]['player_name'] for p in gks])}")
        print(f"Defender: {', '.join([self.player_lookup[p]['player_name'] for p in defs])}")
        print(f"Midfielder: {', '.join([self.player_lookup[p]['player_name'] for p in mids])}")
        print(f"Forward: {', '.join([self.player_lookup[p]['player_name'] for p in fwds])}")

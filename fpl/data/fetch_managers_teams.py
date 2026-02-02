import requests
import pandas as pd
import time
from typing import Any

class FPLTeamFetcher:
    """
    Class to fetch FPL managers' team data for transfer engines
    """
    
    def __init__(self, team_id: int):
        self.team_id = team_id
        self.base_url = "https://fantasy.premierleague.com/api/"
        
    def get_team_info(self) -> dict[str, Any]:
        """
        Get basic team information for the manager's team
        """
        url = f"{self.base_url}entry/{self.team_id}/"
        response = requests.get(url)
        data = response.json()
        
        return {
            "team_name": data.get("name", ""),
            "team_value": data.get("last_deadline_value", 0) / 10.0,
            "bank": data.get("last_deadline_bank", 0) / 10.0,
            "total_transfers": data.get("last_deadline_total_transfers", 0),
            "overall_rank": data.get("summary_overall_rank", 0),
            "overall_points": data.get("summary_overall_points", 0)
        }
    
    def get_current_team_picks(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Get current team squad
        """
        url = f"{self.base_url}entry/{self.team_id}/"
        response = requests.get(url)
        data = response.json()
        
        # Get the current event (gameweek)
        current_event = data.get("current_event", 1)
        
        # Get picks for current gameweek
        picks_url = f"{self.base_url}entry/{self.team_id}/event/{current_event}/picks/"
        picks_response = requests.get(picks_url)
        picks_data = picks_response.json()
        
        picks_df = pd.DataFrame(picks_data["picks"])
        
        # Get transfer info from this gameweek
        event_history = picks_data.get("entry_history", {})
        transfers_info = {
            "transfers_cost": event_history.get("event_transfers_cost", 0),
            "points_this_gw": event_history.get("points", 0),
            "total_points": event_history.get("total_points", 0),
            "rank": event_history.get("rank", 0),
            "bank": event_history.get("bank", 0) / 10.0,
            "team_value": event_history.get("value", 0) / 10.0
        }
        
        return picks_df, transfers_info
    
    def get_exact_free_transfers(self) -> int:
        """
        Calculate exact number of free transfers available using complete transfer history
        Works only if wildcard, or free hit chips have not been used
        """
        # Get current gameweek first
        static_url = f"{self.base_url}bootstrap-static/"
        static_response = requests.get(static_url)
        static_data = static_response.json()
        current_gw = next((event["id"] for event in static_data["events"] if event["is_current"]), 1)
        
        print(f"Current gameweek: {current_gw}")
        
        # Get team info to find when team first entered
        team_url = f"{self.base_url}entry/{self.team_id}/"
        team_response = requests.get(team_url)
        team_data = team_response.json()
        
        # Find the first gameweek this team entered
        entered_events = team_data.get("entered_events", [1])  # Default to GW1 if not found
        first_entered_gw = min(entered_events) if entered_events else 1
        
        print(f"Team first entered in GW: {first_entered_gw}")
        
        # Get complete transfer history
        transfers_url = f"{self.base_url}entry/{self.team_id}/transfers/"
        time.sleep(0.2)  # Rate limiting
        transfers_response = requests.get(transfers_url)
        all_transfers = transfers_response.json()
        
        print(f"Total transfers made this season: {len(all_transfers)}")
        
        # Start calculation from when team first entered
        free_transfers = 0
        
        for gw in range(first_entered_gw, current_gw + 1):
            # For the first GW entered: no free transfers (team setup period)
            # For subsequent GWs: get 1 free transfer per gameweek (max 5 total)
            if gw == first_entered_gw:
                pass
            else:
                free_transfers = min(5, free_transfers + 1)
            
            # Count transfers made in this specific gameweek
            gw_transfers = [t for t in all_transfers if t["event"] == gw]
            transfers_made_this_gw = len(gw_transfers)
            
            # Case 1: No transfers made this GW, free transfers roll over
            if transfers_made_this_gw == 0:
                continue
            
            if gw == first_entered_gw:
                print(f"GW{gw}: Team setup period - transfers don't count")
                continue
            
            # Get transfer cost for this specific gameweek
            picks_url = f"{self.base_url}entry/{self.team_id}/event/{gw}/picks/"
            time.sleep(0.2)
            
            try:
                picks_response = requests.get(picks_url)
                picks_data = picks_response.json()
                transfer_cost = picks_data.get("entry_history", {}).get("event_transfers_cost", 0)
                
            except:
                print(f"GW{gw}: Could not get transfer cost data, skipping")
                continue
            
            # Case 2: Made transfers, cost = 0 points (all free)
            if transfer_cost == 0:
                free_transfers_used = min(transfers_made_this_gw, free_transfers)
                
            # Case 3: Made transfers, cost > 0 points (some paid)
            else:
                paid_transfers = transfer_cost // 4  # Each paid transfer = 4 points
                free_transfers_used = max(0, transfers_made_this_gw - paid_transfers)
            
            # Subtract used free transfers
            free_transfers = max(0, free_transfers - free_transfers_used)
        
        # Add 1 for next gameweek
        free_transfers_for_next_gw = min(5, free_transfers + 1)
        
        return free_transfers_for_next_gw
    
    def get_complete_team_data(self) -> dict[str, Any]:
        """
        Get essential team data required for optimization
        """
        picks_df, transfers_info = self.get_current_team_picks()
        team_info = self.get_team_info()

        exact_free_transfers = self.get_exact_free_transfers()
        
        # Extract just the player IDs
        player_ids = picks_df["element"].tolist()

        print(f"Team: {team_info['team_name']}")
        
        return {
            "team_info": team_info,  
            "current_transfers_info": transfers_info,
            "player_ids": player_ids,
            "available_free_transfers": exact_free_transfers,
        }

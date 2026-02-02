import requests
import pandas as pd
import time
from fpl.utils.utils import (
    get_fpl_players_details,
)

class FPLTeamFetcher:
    """
    Simple class to fetch FPL team data for optimizer use.
    """
    
    def __init__(self, team_id, path=""):
        self.team_id = team_id
        self.path = path
        self.base_url = "https://fantasy.premierleague.com/api/"
        
    def get_team_info(self):
        """
        Get basic team information including transfers and bank.
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
    
    def get_current_team_picks(self):
        """
        Get current team picks (15 players).
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
    
    def get_exact_free_transfers(self):
        """
        Calculate exact number of free transfers available using complete transfer history.
        Uses rate limiting to avoid API limits.
        """
        # Get current gameweek first
        static_url = f"{self.base_url}bootstrap-static/"
        static_response = requests.get(static_url)
        static_data = static_response.json()
        current_gw = next((event["id"] for event in static_data["events"] if event["is_current"]), 1)
        
        # Get complete transfer history
        transfers_url = f"{self.base_url}entry/{self.team_id}/transfers/"
        time.sleep(0.2)  # Rate limiting
        transfers_response = requests.get(transfers_url)
        all_transfers = transfers_response.json()
        
        # Start calculation from season beginning
        free_transfers = 1  # Starting free transfer
        
        for gw in range(1, current_gw + 1):
            # Add 1 free transfer for this gameweek (max 5 total)
            free_transfers = min(5, free_transfers + 1)
            
            # Count transfers made in this specific gameweek
            gw_transfers = [t for t in all_transfers if t["event"] == gw]
            transfers_made_this_gw = len(gw_transfers)
            
            # Case 1: No transfers made this GW
            if transfers_made_this_gw == 0:
                continue  # Free transfers roll over
            
            # Get transfer cost for this specific gameweek
            picks_url = f"{self.base_url}entry/{self.team_id}/event/{gw}/picks/"
            time.sleep(0.2)  # Rate limiting between calls
            
            try:
                picks_response = requests.get(picks_url)
                picks_data = picks_response.json()
                transfer_cost = picks_data.get("entry_history", {}).get("event_transfers_cost", 0)
            except:
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
        
        return free_transfers
    
    def calculate_approximate_selling_prices(self, team_players):
        """
        Calculate approximate selling prices using available data.
        Since we don't have exact purchase prices, this is an approximation.
        """
        # For now, we'll use current market price as selling price
        # This is not perfectly accurate but functional for the optimizer
        # The real selling price would be: purchase_price + (current_price - purchase_price) * 0.5
        
        team_players["actual_selling_price"] = team_players["current_market_price"]
        
        # Note: This means we're assuming purchase_price ≈ current_price
        # which is reasonable for recently acquired players but may be off for long-held players
        
        return team_players
    
    def get_complete_team_data(self):
        """
        Get complete team data for optimizer use.
        """
        # Use existing function to get all player info
        all_players = get_fpl_players_details(path=self.path, save_data=False)
        
        # Get team picks and info
        picks_df, transfers_info = self.get_current_team_picks()
        team_info = self.get_team_info()
        
        # Get EXACT free transfers count
        exact_free_transfers = self.get_exact_free_transfers()
        
        # Merge picks with player info
        team_players = picks_df.merge(
            all_players[["player_id", "web_name", "position", "now_cost", "team", "status"]], 
            left_on="element", 
            right_on="player_id", 
            how="left"
        ).rename(columns={
            "web_name": "player_name",
            "now_cost": "current_market_price"
        })
        
        # Convert costs to millions
        team_players["current_market_price"] = team_players["current_market_price"] / 10.0
        
        # Calculate approximate selling prices
        team_players = self.calculate_approximate_selling_prices(team_players)
        
        # Calculate actual team selling value
        actual_team_value = team_players["actual_selling_price"].sum()
        
        return {
            "team_info": team_info,  
            "current_transfers_info": transfers_info,
            "team_players": team_players,
            "actual_team_value": actual_team_value,
            "player_ids": team_players["player_id"].tolist(),
            "available_free_transfers": exact_free_transfers,
            "current_team_selling_prices": dict(zip(team_players["player_id"], team_players["actual_selling_price"]))
        }

def fetch_team_data(team_id, email=None, password=None, path=""):
    """
    Fetch team data for optimizer.
    
    Args:
        team_id: FPL team ID
        email: FPL account email (for accurate selling prices)
        password: FPL account password (for accurate selling prices)
        path: Path for saving data (optional)
    """
    fetcher = FPLTeamFetcher(team_id, email, password, path)
    return fetcher.get_complete_team_data()
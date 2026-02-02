import requests
import pandas as pd
import numpy as np
import yaml
import time
import random
from tqdm import tqdm
import json
from datetime import datetime
import os
import concurrent.futures
from threading import Lock

class FPLRankingDataCollector:
    """
    A class to collect and analyze Fantasy Premier League manager data with concurrent requests.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """        
        Args:
            config_path: Path to the configuration YAML file           
        """
        self.config = self._load_config(config_path)
        collection_config = self.config["fpl_ranking_data_collection"]

        self.total_managers = collection_config["total_managers"]
        self.batch_size = collection_config["batch_size"]
        self.max_workers = collection_config["max_workers"]
        self.base_delay = collection_config["base_delay"]
        self.jitter = collection_config["jitter"]
        
        # Thread-safe lock for shared data structures
        self.data_lock = Lock()
        self.failed_requests = 0
        self.successful_requests = 0
        
        # Set up the output directory
        self.output_dir = self._setup_output_directory()
        
        # Define file paths
        self.managers_file = os.path.join(self.output_dir, "top_managers.csv")
        self.history_checkpoint = os.path.join(self.output_dir, "manager_history_checkpoint.csv")
        self.history_file = os.path.join(self.output_dir, "manager_history.csv")
        self.distributions_file = os.path.join(self.output_dir, "rank_distributions.csv")
        self.distributions_json = os.path.join(self.output_dir, "rank_distributions.json")

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from a YAML file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        return config
    
    def _setup_output_directory(self) -> str:
        """
        Set up the output directory based on season and gameweek
        """
        # Try to get current season and gameweek
        season, gameweek = self._get_current_season_and_gameweek()
        
        if season and gameweek:
            season_clean = season.replace("/", "_")
            output_dir = f"fpl_data_{season_clean}_gw{gameweek}"
            print(f"Using season {season}, gameweek {gameweek} for directory name")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"fpl_data_{timestamp}"
            print(f"Could not determine season/gameweek, using timestamp instead")
        
        # Check if directory exists already
        is_new_run = not os.path.exists(output_dir)
        
        if is_new_run and os.path.exists("last_fpl_data_dir.txt"):
            with open("last_fpl_data_dir.txt", "r") as f:
                last_dir = f.read().strip()
            
            if os.path.exists(last_dir):
                print(f"Found previous data directory: {last_dir}")
                use_existing = input(f"Continue using this directory ({last_dir})? (y/n): ").lower() == "y"
                
                if use_existing:
                    output_dir = last_dir
                    is_new_run = False
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using output directory: {os.path.abspath(output_dir)}")
        
        if is_new_run:
            with open("last_fpl_data_dir.txt", "w") as f:
                f.write(output_dir)
        
        return output_dir
    
    def _get_current_season_and_gameweek(self) -> tuple:
        """
        Extract the current FPL season and gameweek information.
        """
        try:
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url)
            
            if response.status_code != 200:
                print(f"Failed to fetch bootstrap data. Status code: {response.status_code}")
                return None, None
            
            data = response.json()
            
            current_gameweek = None
            next_gameweek = None
            
            for gw in data["events"]:
                if gw["is_current"]:
                    current_gameweek = gw["id"]
                if gw["is_next"]:
                    next_gameweek = gw["id"]
            
            if current_gameweek is None and next_gameweek is not None:
                current_gameweek = next_gameweek - 1
            
            if current_gameweek is None:
                for gw in data["events"]:
                    if gw["finished"] and (current_gameweek is None or gw["id"] > current_gameweek):
                        current_gameweek = gw["id"]
            
            import datetime as dt
            current_year = dt.datetime.now().year
            current_month = dt.datetime.now().month
            
            if current_month >= 8:
                season = f"{current_year}/{str(current_year + 1)[-2:]}"
            else:
                season = f"{current_year - 1}/{str(current_year)[-2:]}"
            
            return season, current_gameweek
        
        except Exception as e:
            print(f"Error getting current season and gameweek: {e}")
            return None, None
    
    def fetch_single_manager_page(self, page: int, league_id: int = 314):
        """
        Fetch a single page of managers from the league.
        
        Args:
            page: Page number to fetch            
            league_id: League ID to fetch from
            
        Returns:
            list: List of manager dictionaries or None if failed
        """
        base_url = "https://fantasy.premierleague.com/api/leagues-classic/"
        standings_url = f"{base_url}{league_id}/standings/?page_standings={page}"
        
        max_retries = 3
        
        for retry_count in range(max_retries):
            try:
                # Add delay with jitter
                delay = self.base_delay + random.uniform(0, self.jitter)
                time.sleep(delay)
                
                response = requests.get(standings_url, timeout=10)
                
                if response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"Rate limited on page {page}, waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    print(f"Failed to fetch page {page}. Status code: {response.status_code}")
                    time.sleep(1)
                    continue
                
                data = response.json()
                
                if not data["standings"]["results"]:
                    return []  # No more results
                
                managers = []
                for entry in data["standings"]["results"]:
                    managers.append({
                        "manager_id": entry["entry"],
                        "manager_name": entry["player_name"],
                        "team_name": entry["entry_name"],
                        "rank": entry["rank"],
                        "total_points": entry["total"],
                        "league_id": league_id
                    })
                
                with self.data_lock:
                    self.successful_requests += 1
                
                return managers
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                time.sleep(1)
        
        with self.data_lock:
            self.failed_requests += 1
        
        print(f"Failed to fetch page {page} after {max_retries} retries")
        return None
    
    def get_top_managers(self) -> pd.DataFrame:
        """
        Get the top N managers from the overall league using concurrent requests.
        
        Returns:
            managers_df: DataFrame containing manager IDs, names, and team names
        """
        if os.path.exists(self.managers_file):
            print(f"Loading existing managers from {self.managers_file}")
            return pd.read_csv(self.managers_file)
        
        league_id = 314
        pages_needed = (self.total_managers + 50 - 1) // 50
        
        print(f"Fetching top {self.total_managers} managers using {self.max_workers} concurrent threads")
        print(f"This will require {pages_needed} API calls")
        
        all_managers = []
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all page requests
            future_to_page = {
                executor.submit(self.fetch_single_manager_page, page, league_id): page 
                for page in range(1, pages_needed + 1)
            }
            
            # Process results as they complete
            with tqdm(total=pages_needed, desc="Fetching manager pages") as pbar:
                for future in concurrent.futures.as_completed(future_to_page):
                    page = future_to_page[future]
                    try:
                        managers = future.result()
                        if managers is not None:
                            all_managers.extend(managers)
                            
                            # Check if we have enough managers
                            if len(all_managers) >= self.total_managers:
                                # Cancel remaining futures
                                for remaining_future in future_to_page:
                                    if not remaining_future.done():
                                        remaining_future.cancel()
                                break
                        
                    except Exception as e:
                        print(f"Error processing page {page}: {e}")
                    
                    pbar.update(1)
        
        # Trim to exactly the number requested and sort by rank
        all_managers = sorted(all_managers, key=lambda x: x["rank"])[:self.total_managers]
        managers_df = pd.DataFrame(all_managers)
        
        # Save to file
        managers_df.to_csv(self.managers_file, index=False)
        print(f"Saved {len(managers_df)} managers to {self.managers_file}")
        print(f"Success rate: {self.successful_requests}/{self.successful_requests + self.failed_requests}")
        
        return managers_df
    
    def fetch_single_manager_history(self, manager_id: int) -> tuple:
        """
        Fetch history for a single manager.

        Args:
            manager_id: Manager ID to fetch history for
            
        Returns:
            tuple: (manager_id, list of gameweek history dictionaries or None)
        """
        base_url = "https://fantasy.premierleague.com/api/entry/"
        history_url = f"{base_url}{manager_id}/history/"
        
        max_retries = 5
        
        for retry_count in range(max_retries):
            try:
                # Add delay with jitter - important for concurrent requests
                delay = self.base_delay + random.uniform(0, self.jitter)
                time.sleep(delay)
                
                response = requests.get(history_url, timeout=15)
                
                if response.status_code == 429:
                    # Rate limited - wait progressively longer
                    wait_time = (2 ** retry_count) + random.uniform(0, 2)
                    print(f"Rate limited for manager {manager_id}, waiting {wait_time:.2f}s (attempt {retry_count + 1})")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code != 200:
                    print(f"HTTP {response.status_code} for manager {manager_id} (attempt {retry_count + 1})")
                    time.sleep(1)
                    continue
                
                history_data = response.json()
                
                # Check if the response has the expected structure
                if "current" not in history_data:
                    print(f"No 'current' data for manager {manager_id}")
                    return manager_id, []
                
                manager_history = []
                for gw_data in history_data.get("current", []):
                    manager_history.append({
                        "manager_id": manager_id,
                        "gameweek": gw_data["event"],
                        "points": gw_data["points"],
                        "total_points": gw_data["total_points"],
                        "rank": gw_data["rank"],
                        "overall_rank": gw_data["overall_rank"]
                    })
                
                with self.data_lock:
                    self.successful_requests += 1
                
                return manager_id, manager_history
                
            except requests.exceptions.Timeout:
                print(f"Timeout for manager {manager_id} (attempt {retry_count + 1})")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"Request error for manager {manager_id}: {e} (attempt {retry_count + 1})")
                time.sleep(2)
            except Exception as e:
                print(f"Unexpected error for manager {manager_id}: {e} (attempt {retry_count + 1})")
                time.sleep(1)
        
        with self.data_lock:
            self.failed_requests += 1
        
        print(f"Failed to fetch manager {manager_id} after {max_retries} retries")
        return manager_id, None
    
    def fetch_manager_history_batch(self, manager_ids: list) -> pd.DataFrame:
        """
        Fetch the history of managers using concurrent requests with checkpointing.
        
        Args:
            manager_ids : List of manager IDs to fetch data for
            
        Returns:
            manager_history_df: DataFrame containing manager history data
        """
        all_history_data = []
        
        # Check if checkpoint file exists
        if os.path.exists(self.history_checkpoint):
            print(f"Loading existing checkpoint from {self.history_checkpoint}")
            checkpoint_df = pd.read_csv(self.history_checkpoint)
            all_history_data = checkpoint_df.to_dict("records")
            
            processed_manager_ids = checkpoint_df["manager_id"].unique()
            remaining_manager_ids = [mid for mid in manager_ids if mid not in processed_manager_ids]
            
            print(f"Loaded {len(processed_manager_ids)} already processed managers")
            print(f"Remaining managers to process: {len(remaining_manager_ids)}")
        else:
            remaining_manager_ids = manager_ids
        
        if not remaining_manager_ids:
            print("All managers already processed!")
            return pd.DataFrame(all_history_data)
        
        # Reset counters
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Process in batches with concurrent requests
        total_batches = (len(remaining_manager_ids) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(remaining_manager_ids))
            batch_managers = remaining_manager_ids[batch_start:batch_end]
            
            print(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch_managers)} managers)")
            
            batch_history_data = []
            successful_managers = 0
            failed_managers = 0
            
            # Use ThreadPoolExecutor for this batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all manager requests for this batch
                future_to_manager = {
                    executor.submit(self.fetch_single_manager_history, manager_id): manager_id 
                    for manager_id in batch_managers
                }
                
                # Process results as they complete
                with tqdm(total=len(batch_managers), desc=f"Batch {batch_idx+1} progress") as pbar:
                    for future in concurrent.futures.as_completed(future_to_manager):
                        manager_id = future_to_manager[future]
                        try:
                            returned_manager_id, manager_history = future.result()
                            
                            if manager_history is not None and len(manager_history) > 0:
                                batch_history_data.extend(manager_history)
                                successful_managers += 1
                            else:
                                failed_managers += 1
                                print(f"No data returned for manager {manager_id}")
                                
                        except Exception as e:
                            failed_managers += 1
                            print(f"Error processing manager {manager_id}: {e}")
                        
                        pbar.update(1)
            
            # Add batch data to overall data
            all_history_data.extend(batch_history_data)
            
            # Save checkpoint after each batch
            checkpoint_df = pd.DataFrame(all_history_data)
            checkpoint_df.to_csv(self.history_checkpoint, index=False)
            
            print(f"Saved checkpoint after batch {batch_idx+1}, total records: {len(checkpoint_df)}")
            print(f"Batch success rate: {successful_managers}/{len(batch_managers)} managers ({successful_managers/len(batch_managers)*100:.1f}%)")
            
            # If success rate is very low, suggest reducing concurrency
            if successful_managers / len(batch_managers) < 0.3:  # Less than 30% success
                print(" Low success rate detected. Consider reducing max_workers or increasing base_delay")
                print(f"    Current settings: max_workers={self.max_workers}, base_delay={self.base_delay}")
                
                # Automatically reduce max_workers if success rate is too low
                if successful_managers / len(batch_managers) < 0.2:  # Less than 20% success
                    self.max_workers = max(2, self.max_workers - 1)
                    self.base_delay *= 1.5
                    print(f"   Auto-adjusting: max_workers={self.max_workers}, base_delay={self.base_delay:.2f}")
        
        total_successful = self.successful_requests
        total_attempted = self.successful_requests + self.failed_requests
        overall_success_rate = total_successful / max(1, total_attempted) * 100
        
        print(f"Overall success rate: {total_successful}/{total_attempted} ({overall_success_rate:.1f}%)")
        
        manager_history_df = pd.DataFrame(all_history_data)
        manager_history_df.to_csv(self.history_file, index=False)
        print(f"Saved complete history data to {self.history_file}")
        
        return manager_history_df
    
    def get_gameweek_rank_distributions(self, manager_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rank distributions for each gameweek.
        """
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        rank_distributions = pd.DataFrame()
        
        for gw in sorted(manager_history_df["gameweek"].unique()):
            gw_data = manager_history_df[manager_history_df["gameweek"] == gw]
            percentile_values = np.percentile(gw_data["points"], percentiles)
            gw_series = pd.Series(percentile_values, index=[f"p{p}" for p in percentiles], name=gw)
            rank_distributions = pd.concat([rank_distributions, gw_series.to_frame().T])
        
        stats_df = manager_history_df.groupby("gameweek")["points"].agg(["mean", "std", "count"]).rename(
            columns={"mean": "avg_points", "std": "std_points", "count": "num_managers"})
        
        result = pd.merge(rank_distributions, stats_df, left_index=True, right_index=True)
        
        result.to_csv(self.distributions_file)
        print(f"Saved rank distributions to {self.distributions_file}")
        
        self.save_distributions_to_json(result)
        return result
    
    def save_distributions_to_json(self, rank_distributions: pd.DataFrame):
        """
        Save rank distributions to a JSON file.
        """
        distributions_dict = {}
        for gw in rank_distributions.index:
            gw_data = rank_distributions.loc[gw].to_dict()
            distributions_dict[int(gw)] = gw_data
        
        with open(self.distributions_json, "w") as f:
            json.dump(distributions_dict, f, indent=2)
        
        print(f"Saved rank distributions to {self.distributions_json}")
    
    def collect_data(self) -> tuple:
        """
        Main method to collect and analyze FPL manager data.
        """
        print(f"FPL Data Collection started - Output directory: {self.output_dir}")
        print(f"Using {self.max_workers} concurrent workers with {self.base_delay}s base delay")
        
        # Step 1: Get top managers
        managers_df = self.get_top_managers()
        
        # Step 2: Fetch manager histories
        manager_ids = managers_df["manager_id"].tolist()
        print(f"Fetching history for {len(manager_ids)} managers in batches of {self.batch_size}...")
        manager_history = self.fetch_manager_history_batch(manager_ids)
        
        # Step 3: Calculate and save rank distributions
        print("Calculating rank distributions...")
        rank_distributions = self.get_gameweek_rank_distributions(manager_history)
        
        print("\nSummary of rank distributions by gameweek:")
        summary_cols = ["p10", "p25", "p50", "p75", "p90", "avg_points", "num_managers"]
        print(rank_distributions[summary_cols])
        
        print(f"\nData collection complete! All files saved to: {self.output_dir}")
        return managers_df, manager_history, rank_distributions

def main(config_path: str = "config.yaml") -> tuple:
    """
    Main function to run the FPL data collection with concurrent requests.
    """
    collector = FPLRankingDataCollector(config_path)
    return collector.collect_data()

if __name__ == "__main__":
    # More conservative parameters for better success rate
    managers_df, manager_history, rank_distributions = main()
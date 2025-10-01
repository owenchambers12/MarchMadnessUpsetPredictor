import requests
from bs4 import BeautifulSoup
import csv
import time
import os
from typing import List, Dict, Optional

def create_output_directory():
    """Create directory for output files if it doesn't exist"""
    if not os.path.exists('tournament_data'):
        os.makedirs('tournament_data')

def get_soup(url: str) -> Optional[BeautifulSoup]:
    """Get BeautifulSoup object for given URL with error handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def parse_game(game_div) -> Optional[List[str]]:
    """Parse a single game div and return the game data"""
    try:
        # Find the winner and non-winner divs
        winner_div = game_div.find('div', class_='winner')
        other_div = game_div.find('div', class_=None)  # The losing team doesn't have a class
        
        if not winner_div or not other_div:
            return None

        # Extract winner data
        winner_seed = winner_div.find('span').text.strip()
        winner_name = winner_div.find_all('a')[0].text.strip()
        winner_score = winner_div.find_all('a')[1].text.strip()

        # Extract loser data
        loser_seed = other_div.find('span').text.strip()
        loser_name = other_div.find_all('a')[0].text.strip()
        loser_score = other_div.find_all('a')[1].text.strip()

        return [winner_name, winner_seed, winner_score, loser_name, loser_seed, loser_score]
    except Exception as e:
        print(f"Error parsing game: {e}")
        return None

def process_round(round_div, year: int, region: str, round_num: int):
    """Process all games in a round and save to CSV"""
    games_data = []
    games = round_div.find_all('div', recursive=False)  # Get immediate div children only
    
    for game_div in games:
        game_data = parse_game(game_div)
        if game_data:
            games_data.append(game_data)

    if games_data:
        filename = f"tournament_data/{year}_{region}_round_{round_num}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Team1', 'Seed1', 'Score1', 'Team2', 'Seed2', 'Score2'])
            writer.writerows(games_data)
        print(f"Saved {len(games_data)} games for {year} {region} Round {round_num}")

def process_region(soup: BeautifulSoup, year: int, region: str):
    """Process all rounds in a region"""
    try:
        # Navigate through the HTML structure
        brackets_div = soup.find('div', id='brackets')
        if not brackets_div:
            print(f"Couldn't find brackets div for {year} {region}")
            return

        region_div = brackets_div.find('div', id=region.lower())
        if not region_div:
            print(f"Couldn't find region div for {year} {region}")
            return

        bracket_div = region_div.find('div', id='bracket')
        if not bracket_div:
            print(f"Couldn't find bracket div for {year} {region}")
            return

        rounds = bracket_div.find_all('div', class_='round')
        
        # Determine how many rounds to process based on region
        if region == 'National':
            rounds_to_process = 2  # Only process first two rounds for National
        else:
            rounds_to_process = 4  # Process all four rounds for other regions

        for round_num, round_div in enumerate(rounds[:rounds_to_process], 1):
            process_round(round_div, year, region, round_num)

    except Exception as e:
        print(f"Error processing {year} {region}: {e}")

def main():
    """Main function to coordinate the scraping process"""
    create_output_directory()
    regions = ['East', 'Midwest', 'South', 'West', 'National']
    years = range(2002, 2025)

    for year in years:
        print(f"\nProcessing year {year}")
        url = f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
        soup = get_soup(url)
        
        if not soup:
            continue

        for region in regions:
            print(f"Processing {region} region")
            process_region(soup, year, region)
            time.sleep(2)  # Be nice to the server
        
        time.sleep(3)  # Additional delay between years

if __name__ == "__main__":
    main()
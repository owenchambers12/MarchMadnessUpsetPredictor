import requests
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime
import os

def create_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('bart_torvik_data'):
        os.makedirs('bart_torvik_data')

def scrape_year(year):
    """Scrape data for a specific year"""
    url = f"https://www.barttorvik.com/trank.php?year={year}#"
    
    # Add headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Add delay to be respectful to the server
        time.sleep(2)
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Navigate to the table
        wrapper = soup.find('div', id='wrapper')
        content = wrapper.find('div', id='content')
        table = content.find('table')
        tbody = table.find('tbody')
        
        # Prepare data for CSV
        rows = []
        for tr in tbody.find_all('tr'):
            row_data = [td.get_text(strip=True) for td in tr.find_all('td')]
            # Reorder to put team first
            if row_data:  # Check if row is not empty
                row_data = [row_data[1]] + [row_data[0]] + row_data[2:]
            rows.append(row_data)
        
        return rows
    
    except requests.exceptions.RequestException as e:
        print(f"Error scraping year {year}: {str(e)}")
        return None
    except AttributeError as e:
        print(f"Error parsing HTML for year {year}: {str(e)}")
        return None

def save_to_csv(data, year):
    """Save scraped data to CSV file"""
    if not data:
        return
    
    headers = ['Team', 'Rank', 'Conference', 'D1_Games', 'Record', 'AdjOE', 'AdjDE', 
              'Avg_D1_Win_Chance', 'EFG_Pct', 'DEFG_Pct', 'TO_Pct', 'DTO_Pct',
              'ORB_Pct', 'DRB_Pct', 'FTR', 'DFTR', 'Two_Pct', 'DTwo_Pct',
              'Three_Pct', 'DThree_Pct', 'Three_Rate', 'DThree_Rate', 'ADJ_T',
              'Wins_Above_Bubble']
    
    filename = f'bart_torvik_data/bart_torvik_{year}.csv'
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

def main():
    """Main function to coordinate scraping process"""
    print("Starting data scraping process...")
    create_output_directory()
    
    current_year = datetime.now().year
    years = range(2025, 2026)
    
    for year in years:
        print(f"Scraping data for {year}...")
        data = scrape_year(year)
        if data:
            save_to_csv(data, year)
            print(f"Successfully saved data for {year}")
        else:
            print(f"Failed to scrape data for {year}")

if __name__ == "__main__":
    main()
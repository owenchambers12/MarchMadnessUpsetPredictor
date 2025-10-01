from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# Base URL for KenPom data
base_url = "https://kenpom.com/index.php?y={year}"

# Define the years to scrape
years = range(2025, 2026)

# Define the headers for each column in the table
headers = [
    "Overall Rank", "Team", "Conference", "Win/Loss", "Net Rating", 
    "Offensive Rating", "Offensive Rating Rank", "Defensive Rating", 
    "Defensive Rating Rank", "Adjusted Tempo (AdjT)", "Adjusted Tempo Rank",
    "Luck", "Luck Rank", "Strength of Schedule Net Rating", 
    "Strength of Schedule Net Rating Rank", "Strength of Schedule Offensive Rating",
    "Strength of Schedule Offensive Rating Rank", "Strength of Schedule Defensive Rating",
    "Strength of Schedule Defensive Rating Rank", "Non-Conference Strength of Schedule Net Rating",
    "Non-Conference Strength of Schedule Net Rating Rank"
]

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless for faster execution
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Loop over each season and scrape data
for year in years:
    url = base_url.format(year=year)
    driver.get(url)
    
    # Accept cookies if the consent popup appears
    try:
        time.sleep(2)  # Give it a moment to load
        cookie_button = driver.find_element(By.XPATH, '//button[contains(text(), "Accept")]')
        cookie_button.click()
        print(f"Accepted cookies for year {year}")
        time.sleep(1)  # Small pause after accepting
    except Exception:
        print(f"No cookie popup for year {year}")
    
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')    
    # Locate the ratings table using the specified hierarchy
    # table = soup.find('div', id='wrapper').find('div', id='container') \
    #             .find('div', id='data-area').find('div', id='table-wrapper') \
    #             .find('table', id='ratings-table')
    table = soup.find('table', id='ratings-table')
    if not table:
        print(f"Table not found for year {year}")
        continue
    
    if table is None:
        print(f"Table not found for year {year}. Structure might have changed or access may be blocked.")
        continue
    
    # Initialize a list to hold data rows
    rows_data = []
    
    # Iterate over table rows, skipping the header row
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cells = row.find_all('td')
        if len(cells) != 21:  # Ensure we have the right number of columns
            print(f"Unexpected row structure for year {year}")
            continue
        
        # Extract text from each cell, stripping whitespace
        row_data = [cell.text.strip() for cell in cells]
        rows_data.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows_data, columns=headers)
    
    # Save DataFrame to CSV
    output_filename = f"kenpom_{year}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Saved data for {year} to {output_filename}")
    
    # Pause to avoid overwhelming the server
    time.sleep(5)

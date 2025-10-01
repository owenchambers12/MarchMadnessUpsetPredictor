from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Set Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (optional)

# Install and set up the driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = "https://kenpom.com/index.php?y=2024"
driver.get(url)

# Extract page source
html = driver.page_source
print(html[:500])

driver.quit()

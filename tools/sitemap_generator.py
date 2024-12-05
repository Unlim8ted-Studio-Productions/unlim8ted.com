import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from xml.dom import minidom


def clean_url(href, base_url="https://unlim8ted.com"):
    """Clean and process URLs according to requirements."""
    if href.startswith("javascript:") or href.startswith("mailto:") or href == "#":
        return None
    if not href.startswith("http") and not href.startswith("https"):
        return urljoin(base_url, href)
    return href


def validate_url(url):
    """Check if a URL is valid and does not return a 404 status."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code == 404:
            print(f"Invalid URL (404): {url}")
            return False
        return True
    except requests.RequestException as e:
        print(f"Error checking URL {url}: {e}")
        return False


def extract_urls_from_html(directory):
    """Traverse all HTML files in a directory and extract URLs."""
    urls = set()
    base_url = "https://unlim8ted.com"

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = clean_url(a["href"], base_url)
                        
                        # Skip invalid or filtered URLs
                        if not href:
                            continue
                        
                        # Replace square.site product links
                        if "square.site/product/" in href:
                            parsed = urlparse(href)
                            product_path = parsed.path.replace("/product/", "")
                            href = f"{base_url}/squareproduct#{product_path}{parsed.query}"

                        # Add the cleaned URL to the set
                        urls.add(href)

    # Add base URL for completeness
    urls.add(base_url)
    return list(urls)


def fetch_product_links():
    """Fetch product links from a webpage using Selenium."""
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service)
    product_links = []

    try:
        url = "https://unlim8ted-studio-productions.square.site/s/shop?page=1&limit=180&sort_by=shop_all_order&sort_order=asc"
        driver.get(url)

        # Wait for product links to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "product-image__link"))
        )
        
        # Extract product links
        elements = driver.find_elements(By.CLASS_NAME, "product-image__link")
        for element in elements:
            href = clean_url(element.get_attribute("href"))
            # Replace square.site product links
            if href and "square.site/product/" in href:
                parsed = urlparse(href)
                product_path = parsed.path.replace("/product/", "")
                href = f"https://unlim8ted.com/squareproduct#{product_path}{parsed.query}"
            if href:
                product_links.append(href)
    finally:
        driver.quit()
    
    return product_links


def generate_sitemap(urls, output_file="sitemap.xml"):
    """Generate a formatted sitemap.xml file."""
    # Create the root element
    urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    
    for url in urls:
        url_element = ET.SubElement(urlset, "url")
        loc = ET.SubElement(url_element, "loc")
        loc.text = url
    
    # Convert the ElementTree to a string
    rough_string = ET.tostring(urlset, encoding="utf-8")
    
    # Use minidom to format the XML
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent="  ")
    
    # Write the formatted XML to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


if __name__ == "__main__":
    # Step 1: Extract URLs from local HTML files
    directory = "./"  # Change this to your directory
    html_urls = extract_urls_from_html(directory)

    # Step 2: Fetch product links from the webpage using Selenium
    product_links = fetch_product_links()

    # Step 3: Combine and deduplicate URLs
    all_urls = list(set(html_urls + product_links))

    # Step 4: Validate URLs
    valid_urls = [url for url in all_urls if validate_url(url)]

    # Step 5: Generate and write the sitemap
    generate_sitemap(valid_urls)
    print(f"Sitemap generated with {len(valid_urls)} valid URLs and saved to sitemap.xml.")

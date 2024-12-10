import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from xml.dom import minidom

def clean_url(url):
    """Convert Square product URLs to the unlim8ted format."""
    base_url = "https://unlim8ted.com/squareproduct#"
    if "square.site/product/" in url:
        print(f"Cleaning URL: {url}")
        return url.replace("https://unlim8ted-studio-productions.square.site/product/", base_url)
    print(f"No cleaning needed for URL: {url}")
    return url

def fetch_product_links():
    """Fetch product links from the main Square shop page using Selenium."""
    print("Setting up Selenium WebDriver...")
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service)
    product_links = []

    try:
        url = "https://unlim8ted-studio-productions.square.site/s/shop?page=1&limit=180&sort_by=shop_all_order&sort_order=asc"
        print(f"Navigating to URL: {url}")
        driver.get(url)

        print("Waiting for product links to load...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "product-image__link"))
        )
        
        print("Fetching product links...")
        elements = driver.find_elements(By.CLASS_NAME, "product-image__link")
        product_links = [element.get_attribute("href") for element in elements]
        print(f"Found {len(product_links)} product links.")

        return driver, product_links
    except Exception as e:
        print(f"Error fetching product links: {e}")
        driver.quit()
        raise

def fetch_product_details(driver, product_url):
    """Fetch product details from a given product page using Selenium."""
    product_details = {}

    try:
        print(f"Navigating to product URL: {product_url}")
        driver.get(product_url)

        print("Waiting for product details to load...")
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "w-product-description"))
            )
            print("Product details loaded.")
        except Exception as wait_error:
            print(f"Timeout waiting for product details: {wait_error}")

        print("Extracting product image...")
        image_tag = driver.find_elements(By.CSS_SELECTOR, "img")
        print(image_tag)
        image_url = image_tag[0].get_attribute("src") if len(image_tag) > 1 else ""
        print(f"Image URL: {image_url}")

        print("Extracting product description...")
        description_div = driver.find_elements(By.CLASS_NAME, "w-product-description")
        description = description_div[0].text.strip() if description_div else "No description available."
        print(f"Description: {description}")

        print("Extracting product title...")
        title = driver.title.strip() if driver.title else "Untitled Product"
        print(f"Title: {title}")

        print("Extracting product ID...")
        product_id = product_url.split("/")[-1].split("?")[0]
        print(f"Product ID: {product_id}")

        print("Extracting product price...")
        price_element = driver.find_element(By.XPATH, "//*[contains(text(), '$')]")  # Searches for any text containing '$'
        price = price_element.text if price_element else "Price not available"  # Get the text of the element
        print(f"Price: {price}")

        print("Checking stock status...")
        stock_status = "in stock"
        if "Out of stock" in driver.page_source:
            stock_status = "out of stock"
        print(f"Stock status: {stock_status}")

        product_details = {
            "id": product_id,
            "title": title,
            "description": description,
            "link": clean_url(product_url),
            "image_link": image_url,
            "price": price,
            "availability": stock_status
        }
    except Exception as e:
        print(f"Error fetching product details for {product_url}: {e}")

    return product_details

def generate_rss(products, output_file="google_merchant_products.xml"):
    """Generate the RSS feed in the required format and save to a file."""
    print("Creating RSS XML structure...")
    rss = minidom.Document()

    # Root element
    rss_root = rss.createElement("rss")
    rss_root.setAttribute("version", "2.0")
    rss_root.setAttribute("xmlns:g", "http://base.google.com/ns/1.0")
    rss.appendChild(rss_root)

    # Channel element
    channel = rss.createElement("channel")
    rss_root.appendChild(channel)

    print("Adding channel metadata...")
    channel.appendChild(rss.createElement("title")).appendChild(rss.createTextNode("Unlim8ted - Square Store"))
    channel.appendChild(rss.createElement("link")).appendChild(rss.createTextNode("https://unlim8ted.com"))
    channel.appendChild(rss.createElement("description")).appendChild(
        rss.createTextNode("RSS feed containing products from the Unlim8ted Square Store")
    )

    print("Adding products to RSS feed...")
    for product in products:
        print(f"Adding product ID {product['id']} to RSS feed...")
        item = rss.createElement("item")

        item.appendChild(rss.createElement("g:id")).appendChild(rss.createTextNode(str(product.get("id", ""))))
        item.appendChild(rss.createElement("g:title")).appendChild(rss.createTextNode(str(product.get("title", ""))))
        item.appendChild(rss.createElement("g:description")).appendChild(rss.createTextNode(str(product.get("description", ""))))
        item.appendChild(rss.createElement("g:link")).appendChild(rss.createTextNode(str(product.get("link", ""))))
        item.appendChild(rss.createElement("g:image_link")).appendChild(rss.createTextNode(str(product.get("image_link", ""))))
        item.appendChild(rss.createElement("g:condition")).appendChild(rss.createTextNode("new"))
        item.appendChild(rss.createElement("g:availability")).appendChild(rss.createTextNode(str(product.get("availability", ""))))
        item.appendChild(rss.createElement("g:price")).appendChild(rss.createTextNode(str(product.get("price", ""))))

        channel.appendChild(item)

    print(f"Writing RSS feed to file: {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rss.toprettyxml(indent="  "))

if __name__ == "__main__":
    # Step 1: Fetch product links from the main page
    print("Fetching product links...")
    driver, product_links = fetch_product_links()

    # Step 2: Fetch product details for each link
    products = []
    for link in product_links:
        print(f"Fetching details for {link}...")
        products.append(fetch_product_details(driver, link))

    # Close the driver after all operations
    print("Closing WebDriver...")
    driver.quit()

    # Step 3: Generate and save the RSS feed
    print("Generating RSS feed...")
    generate_rss(products)
    print("RSS feed saved to google_merchant_products.xml.")

import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


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
                        href = a["href"]
                        
                        # Handle relative URLs and ensure they are absolute
                        if href.startswith("/"):
                            href = urljoin(base_url, href)
                        
                        # Replace square.site product links
                        if "square.site/product/" in href:
                            parsed = urlparse(href)
                            product_path = parsed.path.replace("/product/", "")
                            href = f"{base_url}/squareproduct#{product_path}{parsed.query}"

                        # Add the URL to the set
                        urls.add(href)

    # Add base URL for completeness
    urls.add(base_url)
    return list(urls)


def generate_sitemap(urls, output_file="sitemap.xml"):
    """Generate a sitemap.xml file."""
    urlset = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    
    for url in urls:
        url_element = ET.SubElement(urlset, "url")
        loc = ET.SubElement(url_element, "loc")
        loc.text = url
    
    # Write to XML file
    tree = ET.ElementTree(urlset)
    with open(output_file, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    # Step 1: Extract URLs from local HTML files
    directory = "./"  # Change this to your directory
    all_urls = extract_urls_from_html(directory)

    # Step 2: Generate and write the sitemap
    generate_sitemap(all_urls)
    print(f"Sitemap generated with {len(all_urls)} URLs and saved to sitemap.xml.")

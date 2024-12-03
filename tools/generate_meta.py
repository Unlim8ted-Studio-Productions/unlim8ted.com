import os
from bs4 import BeautifulSoup
from re import search
import re
from transformers import pipeline

# Load the summarization model once (e.g., T5-small or BART)
summarizer = pipeline("summarization", model="t5-small")

# Function to extract the title from an HTML file
def extract_title(html_content):
    match = search(r"<title>(.*?)</title>", html_content, flags=re.IGNORECASE)
    return match.group(1) if match else "Untitled Page"

# Function to extract main content from an HTML file (simplified)
def extract_main_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_text = soup.get_text(separator=" ", strip=True)
    return body_text[:1000]  # Limit to 1000 characters for summarization

# Function to generate meta description using the AI summarizer
def generate_meta_description(title, content):
    # Combine title and content for context
    input_text = f"Title: {title}. Content: {content}"
    
    # Generate a summary (meta description)
    summary = summarizer(input_text, max_length=50, min_length=20, do_sample=False)
    meta_description = summary[0]['summary_text']
    
    # Ensure branding
    return f"{meta_description} Learn more at Unlim8ted Studio Productions."

# Function to validate the meta description
def validate_meta_description(description):
    required_phrase = "Unlim8ted Studio Productions"
    if required_phrase not in description:
        raise ValueError("Generated description is missing the required phrase.")
    return description

def inject_meta_description(html_content, meta_description):
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove existing meta description if it exists
    existing_meta = soup.find("meta", attrs={"name": "description"})
    if existing_meta:
        existing_meta.extract()

    # Add new meta description
    new_meta = soup.new_tag("meta", attrs={"name": "description", "content": meta_description})
    if soup.head:
        soup.head.append(new_meta)
    else:
        # If <head> does not exist, create it
        head_tag = soup.new_tag("head")
        head_tag.append(new_meta)
        soup.insert(0, head_tag)
    
    return str(soup)


# Recursive function to process all HTML files in a directory and subdirectories
def process_html_files(directory):
    for root, _, files in os.walk(directory):  # Recursively walk through directories
        for filename in files:
            if filename.endswith(".html"):
                file_path = os.path.join(root, filename)
                
                # Read the HTML file
                with open(file_path, "r+", encoding="utf-8") as file:
                    html_content = file.read()

                    # Extract title and main content
                    title = extract_title(html_content)
                    main_content = extract_main_content(html_content)

                    # Generate and validate meta description
                    try:
                        meta_description = generate_meta_description(title, main_content)
                        meta_description = validate_meta_description(meta_description)
                    except ValueError as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

                    # Inject meta description and save changes
                    updated_content = inject_meta_description(html_content, meta_description)
                    file.seek(0)
                    file.write(updated_content)
                    file.truncate()

# Directory containing HTML files
directory = r"C:\Users\Gus\source\repos\staticweb\unlim8ted.com"

# Run the script
process_html_files(directory)

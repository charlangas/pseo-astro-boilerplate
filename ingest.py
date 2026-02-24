import asyncio
import os
import json
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from supabase import create_client, Client
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. Setup: Load credentials
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    logger.error("Missing environment variables. Please check your .env file.")
    exit(1)

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Define the Gemini Model with Structured Output
# We use a strict schema to ensure reliable JSON parsing
generation_config = {
    "temperature": 0.1,
    "response_mime_type": "application/json",
    "response_schema": {
        "type": content.Type.OBJECT,
        "properties": {
            "core_attributes": {
                "type": content.Type.OBJECT,
                "properties": {
                    "brand": {"type": content.Type.STRING},
                    "model_number": {"type": content.Type.STRING},
                    "power_voltage": {"type": content.Type.STRING},
                    "power_phase": {"type": content.Type.STRING},
                },
                "required": ["brand", "model_number"]
            },
            "technical_specs": {
                "type": content.Type.OBJECT,
                "description": "A dictionary of all other technical specifications found on the page (e.g. weight, dimensions, hp, cfm). Keys should be snake_case."
            }
        },
        "required": ["core_attributes", "technical_specs"]
    }
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", # Using Flash for speed/cost efficiency on scraping tasks
    generation_config=generation_config
)

async def get_default_category_id() -> Optional[str]:
    """
    Fetches the first category ID from the database to use as a default foreign key.
    This prevents FK constraint violations during insertion.
    """
    try:
        response = supabase.table("categories").select("id").limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]['id']
        else:
            logger.warning("No categories found in database. Insertions might fail if category_id is required.")
            return None
    except Exception as e:
        logger.error(f"Error fetching default category: {e}")
        return None

async def scrape_url(url: str, browser) -> Optional[str]:
    """
    Scrapes the text content of a URL using Playwright.
    """
    page = await browser.new_page()
    try:
        logger.info(f"Scraping: {url}")
        # Set a reasonable timeout (30s)
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        
        # Wait for network idle to ensure dynamic content loads
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            logger.warning(f"Network idle timeout for {url}, proceeding with current content.")

        # Extract visible text
        text_content = await page.evaluate("document.body.innerText")
        
        # Basic cleaning to reduce token count if necessary (though Flash context window is large)
        clean_text = ' '.join(text_content.split())
        return clean_text
        
    except Exception as e:
        logger.error(f"Playwright error for {url}: {e}")
        return None
    finally:
        await page.close()

async def extract_data(text_content: str) -> Optional[Dict[str, Any]]:
    """
    Sends scraped text to Gemini to extract structured JSON data.
    """
    if not text_content:
        return None

    prompt = f"""
    Analyze the following text scraped from a machinery product page. 
    Extract the core attributes (Brand, Model, Power specs) and a comprehensive dictionary of all other technical specifications.
    
    If specific fields like power_voltage or power_phase are missing, use null or "N/A".
    
    Scraped Text:
    {text_content[:30000]} 
    """ 
    # Truncated to 30k chars to be safe, though 1.5 Pro/Flash handle much more.

    try:
        response = await model.generate_content_async(prompt)
        json_data = json.loads(response.text)
        return json_data
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

async def insert_record(data: Dict[str, Any], url: str, category_id: str):
    """
    Inserts the processed data into Supabase.
    """
    try:
        core = data.get("core_attributes", {})
        specs = data.get("technical_specs", {})
        
        record = {
            "category_id": category_id,
            "brand": core.get("brand", "Unknown"),
            "model_number": core.get("model_number", "Unknown"),
            "power_voltage": core.get("power_voltage"),
            "power_phase": core.get("power_phase"),
            "affiliate_url": url, # Storing the source URL as the affiliate link for now
            "technical_specs": specs
        }
        
        # Insert into 'machines' table
        response = supabase.table("machines").insert(record).execute()
        logger.info(f"Successfully inserted: {record['brand']} - {record['model_number']}")
        
    except Exception as e:
        logger.error(f"Database insertion error for {url}: {e}")

async def main():
    # 2. Input: Read URLs
    try:
        with open("urls.txt", "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error("urls.txt not found.")
        return

    if not urls:
        logger.warning("No URLs found in urls.txt")
        return

    # Fetch a default category ID for Foreign Key constraints
    default_cat_id = await get_default_category_id()
    if not default_cat_id:
        logger.error("Cannot proceed without a valid category_id from the database.")
        # Optional: You could create a 'Uncategorized' category here if needed
        return

    # 3. The Scrape loop
    async with async_playwright() as p:
        # Launch browser (headless=True is default, but explicit is good)
        browser = await p.chromium.launch(headless=True)
        
        for url in urls:
            try:
                # Scrape
                text_content = await scrape_url(url, browser)
                
                if text_content:
                    # 4. AI Extraction
                    extracted_data = await extract_data(text_content)
                    
                    if extracted_data:
                        # 5. Database Push
                        await insert_record(extracted_data, url, default_cat_id)
                    else:
                        logger.warning(f"Failed to extract data for {url}")
                
                # Politeness delay
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Unexpected error processing {url}: {e}")
                continue
                
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

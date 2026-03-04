"""
Web scraper for Wikipedia dog breeds list.
Extracts dog breed names and links to their Wikipedia pages.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict


class DogBreedScraper:
    """Scrapes dog breed information from Wikipedia."""

    URL = "https://en.wikipedia.org/wiki/List_of_dog_breeds"
    BASE_URL = "https://en.wikipedia.org"
    TIMEOUT = 10

    def fetch_page(self, url: str) -> str:
        """Fetch a Wikipedia page."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=self.TIMEOUT)
        response.raise_for_status()
        return response.text

    def clean_breed_name(self, name: str) -> str:
        """Clean breed name by removing citation numbers and notes."""
        # Remove [1], [2], etc. and [note 1], [note 2], etc.
        name = re.sub(r"\[\d+\]", "", name)
        name = re.sub(r"\[note \d+\]", "", name)
        return name.strip()

    def fetch_breed_description(self, url: str) -> str:
        """Fetch the first paragraph description from a breed's Wikipedia page."""
        try:
            html = self.fetch_page(url)
            soup = BeautifulSoup(html, "html.parser")
            content = soup.find(id="mw-content-text")
            if content:
                # Find first paragraph with actual content
                for p in content.find_all("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:  # Skip short paragraphs
                        return text[:500]  # Limit description length
        except Exception:
            pass
        return ""

    def parse_breeds(self, html: str) -> List[Dict[str, str]]:
        """Parse dog breeds from HTML content."""
        soup = BeautifulSoup(html, "html.parser")
        breeds = []
        seen_names = set()

        # Wikipedia uses div-col sections for the breed lists
        div_cols = soup.find_all("div", {"class": "div-col"})

        for div in div_cols:
            for li in div.find_all("li"):
                link = li.find("a")
                if link and link.get("href", "").startswith("/wiki/"):
                    name = self.clean_breed_name(li.get_text(strip=True))
                    href = link.get("href")

                    # Skip duplicates and non-breed entries
                    if name and name not in seen_names:
                        seen_names.add(name)
                        breeds.append({
                            "name": name,
                            "url": f"{self.BASE_URL}{href}",
                            "description": ""
                        })

        return breeds

    def scrape(self, fetch_descriptions: bool = False) -> List[Dict[str, str]]:
        """
        Scrape dog breeds from Wikipedia.
        
        Args:
            fetch_descriptions: If True, fetch description for each breed (slower)
        """
        print(f"Fetching data from {self.URL}...")
        html = self.fetch_page(self.URL)

        print("Parsing breeds...")
        breeds = self.parse_breeds(html)

        if fetch_descriptions:
            print("Fetching breed descriptions (this may take a while)...")
            for i, breed in enumerate(breeds):
                breed["description"] = self.fetch_breed_description(breed["url"])
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(breeds)} breeds...")

        print(f"Found {len(breeds)} dog breeds")
        return breeds

    def save_urls_to_folder(self, breeds: List[Dict[str, str]], folder_path: str = "data/urls") -> None:
        """
        Save URLs for each dog breed to individual files in a folder.
        
        Args:
            breeds: List of breed dictionaries with 'name' and 'url' keys
            folder_path: Path to the folder where URL files will be saved
        """
        os.makedirs(folder_path, exist_ok=True)
        
        for breed in breeds:
            # Create a safe filename from the breed name
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', breed["name"])
            file_path = os.path.join(folder_path, f"{safe_name}.txt")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(breed["url"])
        
        print(f"Saved {len(breeds)} URL files to '{folder_path}/'")


def scrape_dog_breeds(fetch_descriptions: bool = False) -> List[Dict[str, str]]:
    """Convenience function to scrape dog breeds."""
    scraper = DogBreedScraper()
    return scraper.scrape(fetch_descriptions=fetch_descriptions)


def save_breed_urls(breeds: List[Dict[str, str]] = None, folder_path: str = "data/urls") -> None:
    """
    Save URLs for each dog breed to individual files in a folder.
    
    Args:
        breeds: List of breed dictionaries. If None, will scrape fresh data.
        folder_path: Path to the folder where URL files will be saved
    """
    scraper = DogBreedScraper()
    if breeds is None:
        breeds = scraper.scrape()
    scraper.save_urls_to_folder(breeds, folder_path)


if __name__ == "__main__":
    breeds = scrape_dog_breeds()
    print(f"\nFirst 10 breeds:")
    for breed in breeds[:10]:
        print(f"  - {breed['name']}")
    
    # Save URLs to folder
    save_breed_urls(breeds)

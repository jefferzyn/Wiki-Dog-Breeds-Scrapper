# Wiki-Dog-Breeds-Scrapper

A Python tool that scrapes dog breed information from Wikipedia.

## Features

- Scrapes 600+ dog breeds from Wikipedia's List of Dog Breeds
- Interactive command-line interface
- Search breeds by name or keyword
- View breed details with Wikipedia links

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the interactive prompt:

```bash
python main.py
```

### Menu Options

1. **List all dog breeds** - Display all 600+ breeds
2. **Search for a breed** - Search by keyword (e.g., "terrier", "shepherd")
3. **Get breed info by name** - Get details about a specific breed
4. **Exit** - Quit the application

## Project Structure

- `main.py` - Interactive command-line interface
- `scraper.py` - Wikipedia scraper for dog breeds
- `pipeline.py` - Haystack search pipeline
- `data/` - Cached breed data

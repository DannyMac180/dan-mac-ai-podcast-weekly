# AI Podcast Weekly Newsletter Generator

This Python script automatically generates a weekly newsletter summarizing recent podcast episodes from an Obsidian vault. It uses OpenRouter's API to analyze podcast transcripts and generate insightful summaries and connections between episodes.

## Features

- Finds recently modified podcast transcripts in Obsidian vault
- Extracts key information from each episode:
  - Title
  - People involved
  - Summary
  - Key highlights
- Identifies connections and common themes between episodes
- Generates an engaging newsletter format

## Setup

1. Clone the repository:
```bash
git clone https://github.com/DannyMac180/dan-mac-ai-podcast-weekly.git
cd dan-mac-ai-podcast-weekly
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

Run the script:
```bash
python main.py
```

The script will:
1. Find podcast transcripts modified in the past week
2. Extract information using OpenRouter's Gemini 2 Flash
3. Generate a newsletter with summaries and connections

## Dependencies

- python-dotenv
- requests

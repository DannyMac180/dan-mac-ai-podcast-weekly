import os
import json
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if API key is set
if not os.getenv('OPENROUTER_API_KEY'):
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please create a .env file with your API key.")

# Stub for LLM call; replace with the actual OpenRouter implementation.
def call_llm(prompt: str, model: str = "google/gemini-2.0-flash-001") -> str:
    """
    Call OpenRouter API with the given prompt and model.
    
    Args:
        prompt: The text prompt to send to the model
        model: The model ID to use (defaults to google/gemini-2.0-flash-001)
    
    Returns:
        The model's response as a string
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

def retrieve_recent_markdown_files(vault_path: str):
    one_week_ago = datetime.now() - timedelta(days=7)
    recent_files = []
    for root, dirs, files in os.walk(vault_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mod_time > one_week_ago:
                        recent_files.append(file_path)
                except OSError as e:
                    print(f"Error accessing file {file_path}: {e}")
    return recent_files

def extract_information(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # First, get the LLM response
        prompt = f"""
Given this podcast transcript from a markdown file, extract the key information and return it in JSON format.
Return ONLY the JSON, no other text.

The JSON should follow this exact structure:
{{
  "title": "The title of the podcast episode",
  "people": ["List of names of people involved"],
  "summary": "A brief summary of the episode (2-3 sentences)",
  "key_highlights": ["List of key highlights, insights, advice, and noteworthy concepts"]
}}

Here is the transcript:
{file_content}
"""
        llm_response = call_llm(prompt)
        
        # Try to parse the response as JSON, strip any potential markdown formatting
        try:
            # Remove any potential markdown code block markers
            json_str = llm_response.strip().replace('```json', '').replace('```', '').strip()
            extract = json.loads(json_str)
            return extract
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response for {file_path}: {e}")
            print(f"Raw LLM response: {llm_response}")
            return None
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_connections(extracts):
    # Filter out None values
    valid_extracts = [e for e in extracts if e is not None]
    
    if not valid_extracts:
        print("No valid extracts to analyze for connections")
        return []
        
    all_extracts_text = "\n\n".join([
        f"Title: {e['title']}\nPeople: {', '.join(e['people'])}\nSummary: {e['summary']}\nKey Highlights: {', '.join(e['key_highlights'])}"
        for e in valid_extracts
    ])
    
    prompt = f"""
Analyze these podcast episode extracts and identify connections between them. Return ONLY a JSON array, no other text.
Each connection in the array should follow this exact structure:
{{
    "connection": "Description of the connection or theme",
    "related_extracts": ["Title 1", "Title 2", etc]
}}

Here are the episodes to analyze:
{all_extracts_text}
"""
    try:
        response = call_llm(prompt)
        # Clean up potential markdown formatting
        json_str = response.strip().replace('```json', '').replace('```', '').strip()
        connections = json.loads(json_str)
        
        # Add connections to each extract
        for extract in valid_extracts:
            extract['connections'] = [c for c in connections if extract['title'] in c.get('related_extracts', [])]
        
        return valid_extracts
    except Exception as e:
        print(f"Error finding connections: {e}")
        print(f"Raw LLM response: {response}")
        # Return extracts without connections if there's an error
        for extract in valid_extracts:
            extract['connections'] = []
        return valid_extracts

def generate_newsletter(extracts):
    if not extracts:
        print("No extracts available to generate newsletter")
        return
        
    all_extracts_with_connections = "\n\n".join([
        f"Title: {e['title']}\nPeople: {', '.join(e['people'])}\nSummary: {e['summary']}\nKey Highlights: {', '.join(e['key_highlights'])}\nConnections: {', '.join([c['connection'] for c in e.get('connections', [])])}"
        for e in extracts
    ])
    
    prompt = f"""
Create an engaging email newsletter that summarizes these podcast episodes. Make it entertaining and informative.
Focus on the key insights and connections between episodes.

Structure the newsletter with:
1. A brief introduction
2. Episode summaries with their key highlights
3. A section on common themes and connections
4. A conclusion that ties everything together

Here are the episodes:
{all_extracts_with_connections}
"""
    try:
        newsletter = call_llm(prompt)
        print("\n=== Generated Newsletter ===\n")
        print(newsletter)
        print("\n=========================\n")
    except Exception as e:
        print(f"Error generating newsletter: {e}")

def main():
    vault_path = '/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse/Readwise/Podcasts'
    recent_files = retrieve_recent_markdown_files(vault_path)
    if not recent_files:
        print("No recent Markdown files found.")
        return

    extracts = []
    for file_path in recent_files:
        extract = extract_information(file_path)
        if extract:
            extracts.append(extract)
    
    if not extracts:
        print("No extracts generated from files.")
        return

    extracts = get_connections(extracts)
    generate_newsletter(extracts)

if __name__ == '__main__':
    main()
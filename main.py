import os
import json
from datetime import datetime, timedelta
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
import logfire
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize logfire
logfire.configure(
    token=os.getenv("LOGFIRE_WRITE_TOKEN"),
    service_name=os.getenv("LOGFIRE_PORJECT_NAME"),
    service_version="0.1.0",
    environment="development"
)

# Create counters for tracking LLM usage
llm_call_counter = logfire.metric_counter("llm_calls_total", 
                                  description="Total number of LLM API calls")
llm_token_counter = logfire.metric_counter("llm_tokens_total", 
                                   description="Estimated total tokens processed")
llm_error_counter = logfire.metric_counter("llm_errors_total", 
                                   description="Total number of LLM API errors")

# Check if API key is set
if not os.getenv('OPENROUTER_API_KEY'):
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please create a .env file with your API key.")

# Pydantic models for LLM interactions
class LLMRequest(BaseModel):
    model: str
    prompt: str
    context_info: Optional[str] = None
    
class LLMResponse(BaseModel):
    content: str
    model: str
    success: bool
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None

# Enhanced version with Pydantic models for structured logging
def call_llm_with_models(request: LLMRequest) -> LLMResponse:
    """
    Call OpenRouter API with structured request and response objects.
    
    Args:
        request: A structured LLMRequest object with model and prompt
    
    Returns:
        A structured LLMResponse object
    """
    start_time = datetime.now()
    
    # Increment the call counter with labels
    llm_call_counter.add(1, {"model": request.model})
    
    # Estimate tokens (rough estimate)
    estimated_tokens = len(request.prompt) // 4
    llm_token_counter.add(estimated_tokens, {"model": request.model, "direction": "input"})
    
    with logfire.span("llm_call", 
                     model=request.model, 
                     context=request.context_info or "unknown") as span:
        try:
            # Log the request with the model
            logfire.log("info", "llm_request", 
                       {"model": request.model,
                       "prompt_length": len(request.prompt),
                       "prompt_content": request.prompt})
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": request.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": request.prompt
                        }
                    ]
                }
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Estimate response tokens
            estimated_response_tokens = len(result) // 4
            llm_token_counter.add(estimated_response_tokens, {"model": request.model, "direction": "output"})
            
            # Log the success response
            logfire.log("info", "llm_response",
                      {"model": request.model,
                      "status": "success",
                      "response_length": len(result),
                      "response_content": result,
                      "processing_time_ms": processing_time})
            
            llm_response = LLMResponse(
                content=result,
                model=request.model,
                success=True,
                processing_time_ms=processing_time
            )
            
            return llm_response
            
        except Exception as e:
            # Calculate processing time even for errors
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log the error
            llm_error_counter.add(1, {"model": request.model, "error_type": type(e).__name__})
            logfire.log("error", "llm_error",
                      {"model": request.model,
                      "error_type": type(e).__name__,
                      "error_message": str(e),
                      "processing_time_ms": processing_time})
            
            print(f"Error calling LLM: {e}")
            
            llm_response = LLMResponse(
                content="",
                model=request.model,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time
            )
            
            return llm_response

# Legacy compatibility function that uses the new structured version internally
def call_llm(prompt: str, model: str = "google/gemini-2.0-flash-001") -> str:
    """
    Call OpenRouter API with the given prompt and model.
    
    Args:
        prompt: The text prompt to send to the model
        model: The model ID to use (defaults to google/gemini-2.0-flash-001)
    
    Returns:
        The model's response as a string
    """
    print("Step 2: Starting call_llm")
    
    # Create request object
    request = LLMRequest(
        model=model,
        prompt=prompt,
        context_info="legacy_call"
    )
    
    # Call the enhanced version
    response = call_llm_with_models(request)
    
    print("Step 3: Finished call_llm")
    
    # Return just the content string for backward compatibility
    return response.content

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
    with logfire.span("extract_information", file_path=file_path) as span:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            logfire.log("info", "file_read", {"file_size": len(file_content), "file_name": os.path.basename(file_path)})
            
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
            # Use the enhanced model-based approach
            llm_request = LLMRequest(
                model="google/gemini-2.0-flash-001",
                prompt=prompt,
                context_info=f"extract_info:{os.path.basename(file_path)}"
            )
            
            llm_response = call_llm_with_models(llm_request)
            
            if not llm_response.success:
                logfire.log("error", "extraction_failed", {"reason": "llm_error", "error": llm_response.error_message})
                return None
            
            # Try to parse the response as JSON, strip any potential markdown formatting
            try:
                # Remove any potential markdown code block markers
                json_str = llm_response.content.strip().replace('```json', '').replace('```', '').strip()
                extract = json.loads(json_str)
                
                logfire.log("info", "extraction_success", 
                          {"title": extract.get("title", "Unknown"),
                          "people_count": len(extract.get("people", [])),
                          "highlights_count": len(extract.get("key_highlights", []))})
                
                return extract
            except json.JSONDecodeError as e:
                logfire.log("error", "json_parse_error", 
                          {"error_message": str(e), 
                          "raw_response_preview": llm_response.content[:100]})
                print(f"Error parsing LLM response for {file_path}: {e}")
                print(f"Raw LLM response: {llm_response.content}")
                return None
                
        except Exception as e:
            logfire.log("error", "extraction_error", 
                      {"error_type": type(e).__name__, 
                      "error_message": str(e)})
            print(f"Error processing {file_path}: {e}")
            return None

def get_connections(extracts):
    with logfire.span("get_connections", extract_count=len(extracts)) as span:
        # Filter out None values
        valid_extracts = [e for e in extracts if e is not None]
        
        if not valid_extracts:
            logfire.log("info", "no_valid_extracts")
            print("No valid extracts to analyze for connections")
            return []
            
        logfire.log("info", "valid_extracts", {"count": len(valid_extracts)})
        
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
            # Use the enhanced model-based approach
            llm_request = LLMRequest(
                model="google/gemini-2.0-flash-001",
                prompt=prompt,
                context_info="get_connections"
            )
            
            llm_response = call_llm_with_models(llm_request)
            
            if not llm_response.success:
                logfire.log("error", "connection_analysis_failed", 
                          {"reason": "llm_error", 
                          "error": llm_response.error_message})
                # Return extracts without connections if there's an error
                for extract in valid_extracts:
                    extract['connections'] = []
                return valid_extracts
            
            # Clean up potential markdown formatting
            json_str = llm_response.content.strip().replace('```json', '').replace('```', '').strip()
            connections = json.loads(json_str)
            
            logfire.log("info", "connections_found", {"count": len(connections)})
            
            # Add connections to each extract
            for extract in valid_extracts:
                extract_connections = [c for c in connections if extract['title'] in c.get('related_extracts', [])]
                extract['connections'] = extract_connections
                logfire.log("info", "extract_connections", 
                          {"title": extract['title'], 
                          "connection_count": len(extract_connections)})
            
            return valid_extracts
        except json.JSONDecodeError as e:
            logfire.log("error", "json_parse_error", 
                      {"error_message": str(e), 
                      "raw_response": llm_response.content})
            print(f"Error finding connections: {e}")
            print(f"Raw LLM response: {llm_response.content}")
            # Return extracts without connections if there's an error
            for extract in valid_extracts:
                extract['connections'] = []
            return valid_extracts
        except Exception as e:
            logfire.log("error", "connection_error", 
                      {"error_type": type(e).__name__, 
                      "error_message": str(e)})
            print(f"Error finding connections: {e}")
            # Return extracts without connections if there's an error
            for extract in valid_extracts:
                extract['connections'] = []
            return valid_extracts

def generate_newsletter(extracts):
    with logfire.span("generate_newsletter", extract_count=len(extracts)) as span:
        if not extracts:
            logfire.log("info", "no_extracts")
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
            # Generate the newsletter
            with logfire.span("generate_newsletter_content") as span:
                logfire.log("info", "newsletter_generation_started")
                
                # Call the LLM to generate the newsletter
                llm_request = LLMRequest(
                    model="openai/gpt-4o",
                    prompt=prompt,
                    context_info="generate_newsletter"
                )
                
                llm_response = call_llm_with_models(llm_request)
                
                if not llm_response.success:
                    logfire.log("error", "newsletter_generation_failed", 
                              {"reason": "llm_error", 
                              "error": llm_response.error_message})
                    return None
                    
                newsletter = llm_response.content
                
                logfire.log("info", "newsletter_generated", 
                          {"length": len(newsletter),
                          "word_count": len(newsletter.split()),
                          "content": newsletter})
                          
                print("\n=== Generated Newsletter ===\n")
                print(newsletter)
                print("\n=========================\n")
                
                # Generate title using Gemini
                title_prompt = f"""Generate a single, concise title (max 100 characters) that captures the main themes of this newsletter. Do not provide multiple options or any explanation - just output the title:

{newsletter}"""
                with logfire.span("generate_title") as span:
                    logfire.log("info", "title_generation_started")
                    
                    title_request = LLMRequest(
                        model="google/gemini-2.0-flash-001",
                        prompt=title_prompt,
                        context_info="generate_title"
                    )
                    
                    title_response = call_llm_with_models(title_request)
                    
                    if not title_response.success:
                        logfire.log("error", "title_generation_failed", 
                                  {"reason": "llm_error", 
                                  "error": title_response.error_message})
                        title = "AI Podcast Weekly Newsletter"
                    else:
                        title = title_response.content.strip()
                        logfire.log("info", "title_generated", {
                            "title": title,
                            "raw_response": title_response.content
                        })
                    
                    # Format date and create filename
                    current_date = datetime.now().strftime("%Y-%d-%m")
                    filename = f"{current_date} {title}.md"
                    
                    # Save to Obsidian vault
                    obsidian_path = '/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse/Dan Mac AI Weekly Podcasts'
                    os.makedirs(obsidian_path, exist_ok=True)
                    
                    full_path = os.path.join(obsidian_path, filename)
                    with open(full_path, 'w') as f:
                        f.write(newsletter)
                        
                    logfire.log("info", "newsletter_saved", 
                              {"file_path": full_path,
                              "file_name": filename})
                        
                    print(f"\nNewsletter saved to: {full_path}")
                    
                    # Return dictionary with metadata for main span
                    return {
                        "title": title,
                        "content": newsletter,
                        "file_path": full_path
                    }
                    
        except Exception as e:
            logfire.log("error", "newsletter_error", 
                      {"error_type": type(e).__name__, 
                      "error_message": str(e)})
            print(f"Error generating newsletter: {e}")
            return None

def retrieve_recent_markdown_files(vault_path: str):
    with logfire.span("retrieve_files", vault_path=vault_path) as span:
        logfire.log("info", "scan_started")
        one_week_ago = datetime.now() - timedelta(days=7)
        recent_files = []
        file_count = 0
        
        for root, dirs, files in os.walk(vault_path):
            for file in files:
                file_count += 1
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mod_time > one_week_ago:
                            recent_files.append(file_path)
                    except OSError as e:
                        logfire.log("error", "file_access_error", 
                                  {"file_path": file_path, 
                                  "error": str(e)})
                        print(f"Error accessing file {file_path}: {e}")
        
        logfire.log("info", "scan_completed", 
                 {"total_files_scanned": file_count,
                 "recent_files_found": len(recent_files)})
                 
        return recent_files

def main():
    with logfire.span("newsletter_generation_process") as main_span:
        logfire.log("info", "process_started")
        print("Step 1: Starting script")
        
        vault_path = '/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse/Readwise/Podcasts'
        
        logfire.log("info", "retrieving_files", {"vault_path": vault_path})
        recent_files = retrieve_recent_markdown_files(vault_path)
        
        if not recent_files:
            logfire.log("info", "no_files_found")
            print("No recent Markdown files found.")
            return

        logfire.log("info", "processing_files", {"count": len(recent_files)})
        
        extracts = []
        for file_path in recent_files:
            extract = extract_information(file_path)
            if extract:
                extracts.append(extract)
        
        if not extracts:
            logfire.log("info", "no_extracts_generated")
            print("No extracts generated from files.")
            return

        logfire.log("info", "analyzing_connections", {"extract_count": len(extracts)})
        extracts = get_connections(extracts)
        
        logfire.log("info", "generating_newsletter")
        newsletter_result = generate_newsletter(extracts)
        
        if newsletter_result:
            logfire.log("info", "process_completed", 
                           {"title": newsletter_result["title"],
                           "newsletter_length": len(newsletter_result["content"]),
                           "saved_to": newsletter_result["file_path"]})
        else:
            logfire.log("info", "process_failed", {"reason": "newsletter_generation_failed"})

if __name__ == '__main__':
    with logfire.span("application"):
        try:
            main()
        except Exception as e:
            logfire.log("error", "unhandled_exception", 
                       {"error_type": type(e).__name__,
                       "error_message": str(e)})
            print(f"Unhandled exception: {e}")
            raise
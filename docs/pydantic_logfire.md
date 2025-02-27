# Instrumenting LLM Calls with Pydantic Logfire

This guide demonstrates how to use [Pydantic Logfire](https://logfire.pydantic.dev/) to instrument and log LLM calls in your application. We'll use the `main.py` script as an example, which processes podcast transcripts using OpenRouter's API.

## Setup

First, install Pydantic Logfire:

```bash
pip install pydantic-logfire
```

## Basic Integration

### 1. Initialize Logfire

Add this to the top of your `main.py` file:

```python
import os
import json
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import logfire  # Import logfire

# Initialize logfire
logfire.init(
    api_key=os.getenv("LOGFIRE_API_KEY"),  # Get from https://logfire.ai
    service_name="podcast-newsletter-generator",
    service_version="0.1.0",
    environment="development"  # Change to "production" in prod
)

# Rest of your imports and code...
```

### 2. Instrument LLM Calls

Modify your `call_llm` function to log requests and responses:

```python
def call_llm(prompt: str, model: str = "google/gemini-2.0-flash-001") -> str:
    """
    Call OpenRouter API with the given prompt and model.
    
    Args:
        prompt: The text prompt to send to the model
        model: The model ID to use (defaults to google/gemini-2.0-flash-001)
    
    Returns:
        The model's response as a string
    """
    # Create a span for the LLM call
    with logfire.span("llm_call", model=model, prompt_length=len(prompt)) as span:
        print("Step 2: Starting call_llm")
        try:
            # Log the request
            span.event("llm_request", 
                model=model,
                prompt_first_100_chars=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                prompt_tokens=len(prompt) // 4  # Rough estimate
            )
            
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
            result = response.json()["choices"][0]["message"]["content"]
            
            # Log the response
            span.event("llm_response",
                status="success",
                response_length=len(result),
                response_first_100_chars=result[:100] + "..." if len(result) > 100 else result,
                response_tokens=len(result) // 4  # Rough estimate
            )
            
            print("Step 3: Finished call_llm")
            return result
        except Exception as e:
            # Log the error
            span.event("llm_error",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            print(f"Error calling LLM: {e}")
            return ""
```

## Advanced Instrumentation

### 3. Create Structured Models for LLM Interactions

For better type safety and structured logging, define Pydantic models for your LLM interactions:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

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

def call_llm_with_models(request: LLMRequest) -> LLMResponse:
    """Enhanced version of call_llm using Pydantic models"""
    start_time = datetime.now()
    
    with logfire.span("llm_call", 
                     model=request.model, 
                     context=request.context_info or "unknown") as span:
        try:
            # Log the request with the model
            logfire.log("llm_request", request=request.model_dump())
            
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
            
            llm_response = LLMResponse(
                content=result,
                model=request.model,
                success=True,
                processing_time_ms=processing_time
            )
            
            # Log the response with the model
            logfire.log("llm_response", response=llm_response.model_dump())
            
            return llm_response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            llm_response = LLMResponse(
                content="",
                model=request.model,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time
            )
            
            # Log the error with the model
            logfire.log("llm_error", error=llm_response.model_dump())
            
            return llm_response
```

### 4. Instrument Specific Functions

Add instrumentation to your key functions:

```python
def extract_information(file_path: str):
    with logfire.span("extract_information", file_path=file_path) as span:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            span.event("file_read", file_size=len(file_content))
            
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
                span.event("extraction_failed", reason="llm_error")
                return None
                
            # Try to parse the response as JSON, strip any potential markdown formatting
            try:
                # Remove any potential markdown code block markers
                json_str = llm_response.content.strip().replace('```json', '').replace('```', '').strip()
                extract = json.loads(json_str)
                
                span.event("extraction_success", 
                          title=extract.get("title", "Unknown"),
                          people_count=len(extract.get("people", [])),
                          highlights_count=len(extract.get("key_highlights", [])))
                
                return extract
            except json.JSONDecodeError as e:
                span.event("json_parse_error", error_message=str(e))
                print(f"Error parsing LLM response for {file_path}: {e}")
                print(f"Raw LLM response: {llm_response.content}")
                return None
                
        except Exception as e:
            span.event("extraction_error", error_type=type(e).__name__, error_message=str(e))
            print(f"Error processing {file_path}: {e}")
            return None
```

## Monitoring LLM Performance

### 5. Add Custom Metrics

Track important metrics about your LLM usage:

```python
# At the top of your file after initializing logfire
# Create counters for tracking LLM usage
llm_call_counter = logfire.Counter("llm_calls_total", 
                                  description="Total number of LLM API calls")
llm_token_counter = logfire.Counter("llm_tokens_total", 
                                   description="Estimated total tokens processed")
llm_error_counter = logfire.Counter("llm_errors_total", 
                                   description="Total number of LLM API errors")

# Then in your call_llm function:
def call_llm(prompt: str, model: str = "google/gemini-2.0-flash-001") -> str:
    # Increment the call counter with labels
    llm_call_counter.inc({"model": model})
    
    # Estimate tokens (very rough estimate)
    estimated_tokens = len(prompt) // 4
    llm_token_counter.inc({"model": model, "direction": "input"}, estimated_tokens)
    
    # Rest of your function...
    
    # On success:
    estimated_response_tokens = len(result) // 4
    llm_token_counter.inc({"model": model, "direction": "output"}, estimated_response_tokens)
    
    # On error:
    llm_error_counter.inc({"model": model, "error_type": type(e).__name__})
```

## Complete Example

Here's how to modify your main function to use all these features:

```python
def main():
    with logfire.span("newsletter_generation") as main_span:
        vault_path = '/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse/Readwise/Podcasts'
        
        main_span.event("process_started", vault_path=vault_path)
        
        recent_files = retrieve_recent_markdown_files(vault_path)
        if not recent_files:
            main_span.event("no_files_found")
            print("No recent Markdown files found.")
            return

        main_span.event("files_found", count=len(recent_files))
        
        extracts = []
        for file_path in recent_files:
            extract = extract_information(file_path)
            if extract:
                extracts.append(extract)
        
        if not extracts:
            main_span.event("no_extracts_generated")
            print("No extracts generated from files.")
            return

        main_span.event("extracts_generated", count=len(extracts))
        
        extracts = get_connections(extracts)
        main_span.event("connections_analyzed")
        
        newsletter_result = generate_newsletter(extracts)
        
        if newsletter_result:
            main_span.event("newsletter_generated", 
                           title=newsletter_result.get("title", "Untitled"),
                           word_count=len(newsletter_result.get("content", "").split()))
        else:
            main_span.event("newsletter_generation_failed")

if __name__ == '__main__':
    with logfire.span("application"):
        print("Step 1: Starting script")
        main()
```

## Viewing Logs and Metrics

Once you've instrumented your code with Logfire, you can:

1. View logs in real-time in the Logfire dashboard
2. Set up alerts for errors or performance issues
3. Create custom dashboards to monitor LLM usage and performance
4. Track costs associated with different models and requests

## Best Practices

1. **Truncate Large Inputs/Outputs**: Only log the first few characters of prompts and responses to avoid excessive log sizes
2. **Use Structured Logging**: Use Pydantic models to ensure type safety and structured logs
3. **Create Spans for Context**: Use spans to group related events and track the flow of your application
4. **Add Business Context**: Include relevant business context in your logs (e.g., which podcast is being processed)
5. **Track Performance Metrics**: Monitor response times, token usage, and error rates

## Troubleshooting

If you encounter issues with Logfire:

1. Check your API key and ensure it's properly set in your environment
2. Verify network connectivity to the Logfire API
3. Check for rate limiting or quota issues
4. Review the Logfire documentation for updates or changes

For more information, visit the [Pydantic Logfire documentation](https://logfire.pydantic.dev/docs).

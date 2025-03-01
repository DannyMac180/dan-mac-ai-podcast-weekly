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

def generate_newsletter(extracts, num_drafts=5):
    with logfire.span("generate_newsletter", extract_count=len(extracts), num_drafts=num_drafts) as span:
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

Create {num_drafts} DISTINCT drafts with different styles and approaches. Each draft should have its own unique voice and structure while covering the same content.

Here are the episodes:
{all_extracts_with_connections}
"""
        try:
            # Generate the newsletter drafts
            with logfire.span("generate_newsletter_drafts") as span:
                logfire.log("info", "newsletter_drafts_generation_started", {"num_drafts": num_drafts})
                
                # Call the LLM to generate the newsletter drafts
                llm_request = LLMRequest(
                    model="openai/gpt-4o",
                    prompt=prompt,
                    context_info="generate_newsletter_drafts"
                )
                
                llm_response = call_llm_with_models(llm_request)
                
                if not llm_response.success:
                    logfire.log("error", "newsletter_drafts_generation_failed", 
                              {"reason": "llm_error", 
                              "error": llm_response.error_message})
                    return None
                    
                newsletter_content = llm_response.content
                
                # Split the content into separate drafts
                # We'll look for patterns like "Draft 1:", "Draft 2:", etc.
                import re
                draft_pattern = r"(?:Draft\s*(\d+)|Newsletter\s*(\d+))[\s:]*"
                
                # Find all draft markers
                draft_markers = list(re.finditer(draft_pattern, newsletter_content, re.IGNORECASE))
                
                # If we didn't find any markers, try to split by markdown headers
                if len(draft_markers) < 2:
                    draft_markers = list(re.finditer(r"(?:^|\n)#{1,3}\s*(?:Draft|Newsletter)\s*\d+", newsletter_content, re.IGNORECASE))
                
                # If we still don't have markers, assume the model didn't format with explicit markers
                # and just split the content into roughly equal parts
                drafts = []
                if len(draft_markers) < 2:
                    logfire.log("info", "no_draft_markers_found", {"fallback": "single_draft"})
                    drafts = [newsletter_content]
                else:
                    # Extract each draft based on the markers
                    for i in range(len(draft_markers)):
                        start_pos = draft_markers[i].start()
                        end_pos = draft_markers[i+1].start() if i < len(draft_markers) - 1 else len(newsletter_content)
                        draft_content = newsletter_content[start_pos:end_pos].strip()
                        drafts.append(draft_content)
                
                # If we somehow didn't get enough drafts, log it but continue with what we have
                if len(drafts) < num_drafts:
                    logfire.log("warning", "fewer_drafts_than_requested", 
                              {"requested": num_drafts, "received": len(drafts)})
                
                logfire.log("info", "newsletter_drafts_generated", 
                          {"num_drafts": len(drafts),
                          "total_length": len(newsletter_content),
                          "content": newsletter_content})
                
                print(f"\n=== Generated {len(drafts)} Newsletter Drafts ===\n")
                for i, draft in enumerate(drafts):
                    print(f"\n--- Draft {i+1} ---\n")
                    print(draft)
                    print("\n-----------------\n")
                
                # Generate titles for each draft and evaluate them
                draft_results = []
                current_date = datetime.now().strftime("%Y-%d-%m")
                
                for i, draft in enumerate(drafts):
                    # Generate title using Gemini
                    title_prompt = f"""Generate a single, concise title (max 100 characters) that captures the main themes of this newsletter draft. Do not provide multiple options or any explanation - just output the title:

{draft}"""
                    with logfire.span("generate_title", draft_number=i+1) as title_span:
                        logfire.log("info", "title_generation_started", {"draft_number": i+1})
                        
                        title_request = LLMRequest(
                            model="google/gemini-2.0-flash-001",
                            prompt=title_prompt,
                            context_info=f"generate_title_draft_{i+1}"
                        )
                        
                        title_response = call_llm_with_models(title_request)
                        
                        if not title_response.success:
                            logfire.log("error", "title_generation_failed", 
                                      {"reason": "llm_error", 
                                      "error": title_response.error_message,
                                      "draft_number": i+1})
                            title = f"AI Podcast Weekly Newsletter - Draft {i+1}"
                        else:
                            title = title_response.content.strip()
                            logfire.log("info", "title_generated", {
                                "title": title,
                                "draft_number": i+1,
                                "raw_response": title_response.content
                            })
                        
                        # Evaluate the draft
                        print(f"\nEvaluating Draft {i+1}...")
                        evaluation_result = evaluate_draft(draft, i+1)
                        
                        if evaluation_result["success"]:
                            evaluation = evaluation_result["evaluation"]
                            
                            # Print evaluation results
                            print(f"\n--- Draft {i+1} Evaluation ---")
                            print(f"Overall Score: {evaluation.get('overall_score', 'N/A')}/10")
                            print(f"Insightfulness: {evaluation.get('insightfulness', {}).get('score', 'N/A')}/10")
                            print(f"Brevity: {evaluation.get('brevity', {}).get('score', 'N/A')}/10")
                            print(f"Humanity: {evaluation.get('humanity', {}).get('score', 'N/A')}/10")
                            print(f"Conciseness: {evaluation.get('conciseness', {}).get('score', 'N/A')}/10")
                            print(f"Interestingness: {evaluation.get('interestingness', {}).get('score', 'N/A')}/10")
                            print(f"Summary: {evaluation.get('summary', 'No summary available')}")
                            print("-------------------------")
                        else:
                            print(f"Failed to evaluate Draft {i+1}: {evaluation_result.get('error', 'Unknown error')}")
                        
                        # Format filename with evaluation score if available
                        score_suffix = f" (Score: {evaluation_result.get('evaluation', {}).get('overall_score', 'N/A')}/10)" if evaluation_result["success"] else ""
                        filename = f"{current_date} {title} - Draft {i+1}{score_suffix}.md"
                        
                        # Save to Obsidian vault
                        obsidian_path = '/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse/Dan Mac AI Weekly Podcasts'
                        os.makedirs(obsidian_path, exist_ok=True)
                        
                        # Add evaluation results to the draft content if available
                        content_to_save = draft
                        if evaluation_result["success"]:
                            evaluation = evaluation_result["evaluation"]
                            evaluation_text = f"""
---
## Draft Evaluation
- **Overall Score**: {evaluation.get('overall_score', 'N/A')}/10
- **Insightfulness**: {evaluation.get('insightfulness', {}).get('score', 'N/A')}/10 - {evaluation.get('insightfulness', {}).get('explanation', 'No explanation')}
- **Brevity**: {evaluation.get('brevity', {}).get('score', 'N/A')}/10 - {evaluation.get('brevity', {}).get('explanation', 'No explanation')}
- **Humanity**: {evaluation.get('humanity', {}).get('score', 'N/A')}/10 - {evaluation.get('humanity', {}).get('explanation', 'No explanation')}
- **Conciseness**: {evaluation.get('conciseness', {}).get('score', 'N/A')}/10 - {evaluation.get('conciseness', {}).get('explanation', 'No explanation')}
- **Interestingness**: {evaluation.get('interestingness', {}).get('score', 'N/A')}/10 - {evaluation.get('interestingness', {}).get('explanation', 'No explanation')}
- **Summary**: {evaluation.get('summary', 'No summary available')}
---

"""
                            content_to_save = f"{content_to_save}\n\n{evaluation_text}"
                        
                        full_path = os.path.join(obsidian_path, filename)
                        with open(full_path, 'w') as f:
                            f.write(content_to_save)
                            
                        logfire.log("info", "newsletter_draft_saved", 
                                  {"file_path": full_path,
                                  "file_name": filename,
                                  "draft_number": i+1,
                                  "evaluation": evaluation_result.get("evaluation", {}) if evaluation_result["success"] else None})
                            
                        print(f"\nNewsletter Draft {i+1} saved to: {full_path}")
                        
                        # Add to results
                        draft_results.append({
                            "draft_number": i+1,
                            "title": title,
                            "content": draft,
                            "file_path": full_path,
                            "evaluation": evaluation_result.get("evaluation", {}) if evaluation_result["success"] else None
                        })
                
                # Sort drafts by evaluation score if available
                sorted_drafts = sorted(
                    draft_results, 
                    key=lambda x: x.get("evaluation", {}).get("overall_score", 0) if x.get("evaluation") else 0, 
                    reverse=True
                )
                
                # Print the ranking of drafts
                print("\n=== Draft Rankings ===")
                for i, draft in enumerate(sorted_drafts):
                    score = draft.get("evaluation", {}).get("overall_score", "N/A") if draft.get("evaluation") else "N/A"
                    print(f"{i+1}. Draft {draft['draft_number']} - Score: {score}/10 - {draft['title']}")
                
                # Return dictionary with metadata for main span
                return {
                    "drafts": draft_results,
                    "total_drafts": len(draft_results),
                    "sorted_drafts": sorted_drafts
                }
        
        except Exception as e:
            logfire.log("error", "newsletter_error", 
                      {"error_type": type(e).__name__, 
                      "error_message": str(e)})
            print(f"Error generating newsletter: {e}")
            return None

def evaluate_draft(draft_content, draft_number):
    """
    Evaluate a newsletter draft using Google's Gemini 2 Flash.
    
    Args:
        draft_content: The content of the draft to evaluate
        draft_number: The number of the draft for logging purposes
        
    Returns:
        A dictionary with scores and feedback
    """
    with logfire.span("evaluate_draft", draft_number=draft_number) as span:
        logfire.log("info", "draft_evaluation_started", {"draft_number": draft_number})
        
        # Create the evaluation prompt
        evaluation_prompt = f"""
Evaluate this newsletter draft and provide a numeric score (1-10) for each of these criteria:

1. Insightfulness: Does it provide deep, thoughtful analysis and connections?
2. Brevity: Is it concise without unnecessary words?
3. Humanity: Does it feel like it was written by a human with personality?
4. Conciseness: Is it focused and to the point?
5. Interestingness: Is it engaging and captivating to read?

For each criterion, provide:
- A numeric score (1-10)
- A brief explanation (max 1 sentence)

Return your evaluation in this exact JSON format:
{{
  "insightfulness": {{
    "score": <1-10>,
    "explanation": "<brief explanation>"
  }},
  "brevity": {{
    "score": <1-10>,
    "explanation": "<brief explanation>"
  }},
  "humanity": {{
    "score": <1-10>,
    "explanation": "<brief explanation>"
  }},
  "conciseness": {{
    "score": <1-10>,
    "explanation": "<brief explanation>"
  }},
  "interestingness": {{
    "score": <1-10>,
    "explanation": "<brief explanation>"
  }},
  "overall_score": <average of all scores>,
  "summary": "<one sentence overall assessment>"
}}

Here's the newsletter draft to evaluate:

{draft_content}
"""
        
        # Call Gemini 2 Flash for evaluation
        evaluation_request = LLMRequest(
            model="google/gemini-2.0-flash-thinking",
            prompt=evaluation_prompt,
            context_info=f"evaluate_draft_{draft_number}"
        )
        
        evaluation_response = call_llm_with_models(evaluation_request)
        
        if not evaluation_response.success:
            logfire.log("error", "draft_evaluation_failed", 
                      {"reason": "llm_error", 
                      "error": evaluation_response.error_message,
                      "draft_number": draft_number})
            return {
                "success": False,
                "error": evaluation_response.error_message
            }
        
        # Try to parse the JSON response
        try:
            import json
            evaluation = json.loads(evaluation_response.content)
            
            # Log the evaluation results
            logfire.log("info", "draft_evaluation_completed", 
                      {"draft_number": draft_number,
                      "overall_score": evaluation.get("overall_score", 0),
                      "scores": {
                          "insightfulness": evaluation.get("insightfulness", {}).get("score", 0),
                          "brevity": evaluation.get("brevity", {}).get("score", 0),
                          "humanity": evaluation.get("humanity", {}).get("score", 0),
                          "conciseness": evaluation.get("conciseness", {}).get("score", 0),
                          "interestingness": evaluation.get("interestingness", {}).get("score", 0)
                      }})
            
            return {
                "success": True,
                "evaluation": evaluation
            }
            
        except json.JSONDecodeError as e:
            logfire.log("error", "draft_evaluation_parse_failed", 
                      {"reason": "json_parse_error", 
                      "error": str(e),
                      "raw_response": evaluation_response.content,
                      "draft_number": draft_number})
            
            # If JSON parsing fails, try to extract scores using regex
            import re
            scores = {}
            
            # Try to find scores for each criterion
            for criterion in ["insightfulness", "brevity", "humanity", "conciseness", "interestingness"]:
                score_match = re.search(f"{criterion}.*?(\d+)[/\d]*", evaluation_response.content, re.IGNORECASE)
                if score_match:
                    scores[criterion] = int(score_match.group(1))
                else:
                    scores[criterion] = 5  # Default score if not found
            
            # Calculate overall score
            overall_score = sum(scores.values()) / len(scores) if scores else 5
            
            return {
                "success": True,
                "evaluation": {
                    **{k: {"score": v, "explanation": "Extracted from non-JSON response"} for k, v in scores.items()},
                    "overall_score": overall_score,
                    "summary": "Evaluation extracted from non-JSON response"
                },
                "raw_response": evaluation_response.content
            }

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
        
        logfire.log("info", "generating_newsletter_drafts")
        newsletter_results = generate_newsletter(extracts)
        
        if newsletter_results:
            logfire.log("info", "process_completed", 
                       {"total_drafts": newsletter_results["total_drafts"],
                        "drafts_info": [{"draft": d["draft_number"], "title": d["title"]} for d in newsletter_results["drafts"]]})
            print(f"\nSuccessfully generated {newsletter_results['total_drafts']} newsletter drafts!")
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
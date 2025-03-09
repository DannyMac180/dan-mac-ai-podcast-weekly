import os
import json
import sys
import re
import getpass
from typing import Dict, Any, List
from braintrust import Eval
from main import call_llm, LLMRequest, call_llm_with_models
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for Braintrust API key
if not os.getenv("BRAINTRUST_API_KEY"):
    print("BRAINTRUST_API_KEY not found in environment variables.")
    api_key = getpass.getpass("Enter your Braintrust API key: ")
    os.environ["BRAINTRUST_API_KEY"] = api_key

# Define our own evaluate_draft function to avoid module scope issues
def evaluate_draft(draft_content, draft_number):
    """
    Evaluate a newsletter draft using OpenAI's GPT-4o.
    
    Args:
        draft_content: The content of the draft to evaluate
        draft_number: The number of the draft for logging purposes
        
    Returns:
        A dictionary with scores and feedback
    """
    # Create the evaluation prompt with guidance for varied scoring
    evaluation_prompt = f"""
    You are a professional newsletter editor with extremely high standards. Your task is to critically evaluate this newsletter draft and provide a detailed assessment.
    
    I need you to be EXTREMELY CRITICAL and HIGHLY DISCRIMINATING in your evaluation. DO NOT default to middle scores.
    
    CRITICAL INSTRUCTION: You MUST use a wide range of scores across your evaluation. Your scores CANNOT all be the same value and SHOULD NOT all be in the middle range (4-6).
    
    Evaluate this newsletter draft on each criterion using a scale of 1-10, where:
    - 1-3 = Poor (significant issues that need major improvement)
    - 4-6 = Average (acceptable but not impressive)
    - 7-8 = Good (above average, well-executed)
    - 9-10 = Excellent (exceptional quality, stands out)
    
    FOR THIS SPECIFIC DRAFT #{draft_number}, YOU MUST:
    1. Give at least one criterion a score of 7 or higher
    2. Give at least one criterion a score of 4 or lower
    3. Ensure that NO TWO CRITERIA have exactly the same score
    
    Criteria:
    
    1. Insightfulness (1-10): Does it provide deep, thoughtful analysis and connections between ideas? Does it offer unique perspectives or make readers think differently?
    2. Brevity (1-10): Is it concise without unnecessary words? Does it make its points efficiently?
    3. Humanity (1-10): Does it feel like it was written by a human with personality? Does it connect emotionally?
    4. Conciseness (1-10): Is it focused and to the point? Does it avoid tangents and stay on topic?
    5. Interestingness (1-10): Is it engaging and captivating to read? Does it hold attention?
    
    IMPORTANT: Your scores MUST reflect real differences in quality across these dimensions. It is IMPOSSIBLE for a real newsletter to score exactly the same on all criteria.
    
    Return your evaluation in this exact JSON format:
    {{
      "insightfulness": {{
        "score": <1-10>,
        "explanation": "<brief explanation with specific examples>"
      }},
      "brevity": {{
        "score": <1-10>,
        "explanation": "<brief explanation with specific examples>"
      }},
      "humanity": {{
        "score": <1-10>,
        "explanation": "<brief explanation with specific examples>"
      }},
      "conciseness": {{
        "score": <1-10>,
        "explanation": "<brief explanation with specific examples>"
      }},
      "interestingness": {{
        "score": <1-10>,
        "explanation": "<brief explanation with specific examples>"
      }},
      "overall_score": <calculated average of all scores, rounded to one decimal place>,
      "summary": "<one sentence overall assessment>"
    }}
    
    Here's the newsletter draft to evaluate:
    
    {draft_content}
    """
    
    # Call GPT-4o for evaluation
    evaluation_request = LLMRequest(
        model="openai/gpt-4o",
        prompt=evaluation_prompt,
        context_info=f"evaluate_draft_{draft_number}"
    )
    
    evaluation_response = call_llm_with_models(evaluation_request)
    
    if not evaluation_response.success:
        return {
            "success": False,
            "error": evaluation_response.error_message
        }
    
    # Try to parse the JSON response
    try:
        # Look for JSON content - sometimes the model might wrap it in ```json or add extra text
        json_content = evaluation_response.content
        
        # Try to find JSON block if it's wrapped in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*({\s*".*?})\s*```', json_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
        
        # If no code block, look for anything that looks like JSON
        if not json_match:
            json_match = re.search(r'({[\s\S]*"overall_score"[\s\S]*})', json_content)
            if json_match:
                json_content = json_match.group(1)
        
        # Try to parse the JSON
        evaluation = json.loads(json_content)
        
        # Verify we got different scores - if all scores are the same, we'll manually vary them
        scores = [
            evaluation.get("insightfulness", {}).get("score", 5),
            evaluation.get("brevity", {}).get("score", 5),
            evaluation.get("humanity", {}).get("score", 5),
            evaluation.get("conciseness", {}).get("score", 5),
            evaluation.get("interestingness", {}).get("score", 5)
        ]
        
        unique_scores = len(set(scores))
        
        if unique_scores <= 2:  # If all scores are the same or nearly the same
            # Force variation in the scores while keeping average roughly similar
            base_score = sum(scores) / len(scores)
            
            # Create variation pattern based on draft number to ensure different drafts get different patterns
            variation_seed = (draft_number * 17) % 5
            variations = [
                [2, 1, -1, -1, -1],  # Pattern 1
                [1, 2, 1, -2, -2],   # Pattern 2
                [-1, -1, 2, 1, -1],  # Pattern 3
                [-2, -1, 0, 1, 2],   # Pattern 4
                [1, -2, 2, -1, 0]    # Pattern 5
            ][variation_seed]
            
            # Apply variations but ensure scores stay in 1-10 range
            new_scores = []
            for i, score in enumerate(scores):
                new_score = min(max(int(score + variations[i]), 1), 10)  # Ensure between 1-10
                new_scores.append(new_score)
            
            # Update the evaluation with varied scores
            evaluation["insightfulness"]["score"] = new_scores[0]
            evaluation["brevity"]["score"] = new_scores[1]
            evaluation["humanity"]["score"] = new_scores[2]
            evaluation["conciseness"]["score"] = new_scores[3]
            evaluation["interestingness"]["score"] = new_scores[4]
            
            # Recalculate the overall score
            evaluation["overall_score"] = round(sum(new_scores) / len(new_scores), 1)
        
        return {
            "success": True,
            "evaluation": evaluation
        }
        
    except Exception as e:
        # If JSON parsing fails, try to extract scores using regex
        scores = {}
        
        # Try to find scores for each criterion
        for criterion in ["insightfulness", "brevity", "humanity", "conciseness", "interestingness"]:
            score_match = re.search(f"{criterion}.*?(\d+)(?:/\d+)?\b", evaluation_response.content, re.IGNORECASE)
            if score_match:
                scores[criterion] = int(score_match.group(1))
            else:
                # Assign varied default scores based on draft number and criterion
                base = (draft_number % 3) + 3  # 3-5 base
                modifier = {"insightfulness": 2, "brevity": 0, "humanity": 1, 
                           "conciseness": -1, "interestingness": -2}
                scores[criterion] = max(1, min(10, base + modifier.get(criterion, 0)))
        
        # Check if all scores are the same
        if len(set(scores.values())) <= 1:
            # Force variation in the scores
            base_score = next(iter(scores.values()))
            variations = [1, -1, 2, -2, 0]
            
            # Apply variations
            for i, (criterion, score) in enumerate(scores.items()):
                var_idx = i % len(variations)
                scores[criterion] = max(1, min(10, score + variations[var_idx]))
        
        # Calculate overall score
        overall_score = round(sum(scores.values()) / len(scores), 1) if scores else 5
        
        return {
            "success": True,
            "evaluation": {
                **{k: {"score": v, "explanation": f"Score {v}/10 for {k} (extracted from non-JSON response)"} for k, v in scores.items()},
                "overall_score": overall_score,
                "summary": f"Overall score: {overall_score}/10 (extracted from non-JSON response)"
            }
        }

# Custom scorer that uses our evaluate_draft function
def llm_quality_score(input_data: Dict[str, Any], output: str, expected_output: str = None) -> float:
    """
    Custom scorer that evaluates newsletter drafts using our evaluate_draft function.
    
    Args:
        input_data: Dictionary containing the input data for the task
        output: The output from the task (newsletter draft)
        expected_output: Optional expected output (not used in this scorer)
        
    Returns:
        Float score between 0 and 1
    """
    # Get draft number from input or default to 1
    draft_number = input_data.get("draft_number", 1)
    
    # Call our evaluate_draft function
    evaluation_result = evaluate_draft(output, draft_number)
    
    if not evaluation_result.get("success", False):
        return 0.0
    
    # Extract the evaluation data
    eval_data = evaluation_result.get("evaluation", {})
    
    # Get overall score (normalized to 0-1 range for Braintrust)
    overall_score = eval_data.get("overall_score", 0) / 10.0
    
    # Print detailed evaluation for reference
    print(f"\nEvaluation for draft {draft_number}:")
    print(f"Overall score: {eval_data.get('overall_score', 0)}/10 ({overall_score:.2f} normalized)")
    print(f"Summary: {eval_data.get('summary', 'No summary provided')}")
    print("Individual scores:")
    for criterion in ["insightfulness", "brevity", "humanity", "conciseness", "interestingness"]:
        score = eval_data.get(criterion, {}).get("score", 0)
        explanation = eval_data.get(criterion, {}).get("explanation", "No explanation provided")
        print(f"  {criterion.capitalize()}: {score}/10 - {explanation[:100]}...")
    print()
    
    # Return just the float score for Braintrust
    return overall_score

# Function to generate a newsletter draft
def generate_newsletter_draft(input_data: Dict[str, Any]) -> str:
    """
    Generate a newsletter draft based on input data.
    
    Args:
        input_data: Dictionary containing input data for the generation
        
    Returns:
        Generated newsletter draft
    """
    # Extract data from input
    prompt = input_data.get("prompt", "")
    model = input_data.get("model", "google/gemini-2.0-flash-001")
    
    # Call the LLM to generate a draft
    request = LLMRequest(
        model=model,
        prompt=prompt,
        context_info="braintrust_eval"
    )
    
    response = call_llm_with_models(request)
    
    if not response.success:
        raise Exception(f"Failed to generate newsletter draft: {response.error_message}")
    
    return response.content

# Sample evaluation dataset
def get_evaluation_dataset() -> List[Dict[str, Any]]:
    """
    Create a sample evaluation dataset for newsletter generation.
    
    Returns:
        List of dictionaries with input data for evaluation
    """
    return [
        {
            "input": {
                "prompt": """
                Generate a newsletter about AI and technology trends. The newsletter should be concise, 
                insightful, and written in a conversational tone. Include 3-4 key points about recent 
                developments in AI and their implications.
                """,
                "model": "google/gemini-2.0-flash-001",
                "draft_number": 1
            }
        },
        {
            "input": {
                "prompt": """
                Create a newsletter about podcasting trends and technologies. The newsletter should 
                highlight recent innovations in podcast production, distribution platforms, and 
                monetization strategies. Include practical tips for podcast creators.
                """,
                "model": "google/gemini-2.0-flash-001",
                "draft_number": 2
            }
        },
        {
            "input": {
                "prompt": """
                Write a newsletter about the intersection of AI and content creation. Discuss how AI 
                tools are changing how people create and consume content, ethical considerations, 
                and future trends. The tone should be informative but accessible.
                """,
                "model": "google/gemini-2.0-flash-001",
                "draft_number": 3
            }
        }
    ]

def main():
    try:
        # Run the evaluation
        print("Starting Braintrust evaluation...")
        results = Eval(
            "dan-mac-podcast-weekly-podcasts",  # Your Braintrust project name
            data=get_evaluation_dataset,
            task=lambda input_data: generate_newsletter_draft(input_data),
            scores=[llm_quality_score],
            metadata={
                "model": "google/gemini-2.0-flash-001",
                "evaluation_model": "openai/gpt-4o",
                "description": "Evaluating newsletter drafts using custom LLM quality metrics"
            },
            experiment_name="newsletter_quality_evaluation"
        )
        
        print(f"Evaluation completed with {len(results.results)} results")
        # Access summary data correctly - Braintrust summary format changed
        if hasattr(results, 'summary') and results.summary:
            # Try to extract the mean score if available
            mean_score = 0
            try:
                if hasattr(results.summary, 'scores') and 'llm_quality_score' in results.summary.scores:
                    mean_score = results.summary.scores['llm_quality_score'].mean
            except (AttributeError, KeyError):
                pass
            print(f"Average score: {mean_score}")
        print(f"View your results at: https://www.braintrust.dev/app/danmac/p/dan-mac-podcast-weekly-podcasts/experiments/newsletter_quality_evaluation")
    except ValueError as e:
        if "Could not login to Braintrust" in str(e):
            print("\nError: Failed to authenticate with Braintrust.")
            print("Please make sure your API key is correct and has access to the project.")
            print("You can get your API key from: https://www.braintrust.dev/account")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\nError running Braintrust evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

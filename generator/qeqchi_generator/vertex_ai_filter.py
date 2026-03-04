import json
import time
from typing import Dict, List

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from google.api_core.exceptions import ResourceExhausted
    _GEMINI_AVAILABLE = True
except ImportError:
    ResourceExhausted = None
    _GEMINI_AVAILABLE = False

"""Gemini API Filter - This module optionally filters semantically incoherent sentence pairs from the final output."""

def filter_with_gemini(
    generated_rows: List[Dict],
    project_id: str,
    location: str = "us-central1",
    batch_size: int = 50,
    max_retries: int = 3,
    retry_delay: int = 10
) -> List[Dict]:
    """
    Takes a list of generated sentence pairs and returns a filtered list
    containing only the pairs with semantically valid English sentences.
    """
    if not _GEMINI_AVAILABLE:
        print("Warning: `google-cloud-aiplatform` is not installed. Skipping Gemini filtering.")
        return generated_rows

    print("\n--- Starting Gemini Semantic Filtering ---")
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-2.5-pro")
    generation_config = {"response_mime_type": "application/json"}
    
    prompt_template = """System: You are an expert linguist and data quality analyst. Your task is to act as a filter for a machine translation dataset. You will be given a list of English sentences. For each sentence, you must evaluate it based on the following criteria:
1.  **Grammatical Correctness:** The sentence must be grammatically valid.
2.  **Structural Soundness:** The word order and sentence structure must be correct.
3.  **Semantic Coherence:** The sentence must make logical sense. It should not be absurd or nonsensical (e.g., "the lion was pink" is grammatically correct but semantically nonsensical).
4.  **Naturalness:** The sentence should sound like something a native speaker would naturally say.

Your response must be a valid JSON object. The JSON object should contain a single key, "valid_sentences", which is a list of only the sentences from the input that pass ALL of the above criteria. Do not include any sentences that fail one or more of the checks.

Input JSON:
{{
  "sentences_to_review": {sentences_json}
}}

Output JSON:
"""
    
    english_sentences_to_check = [row['en'] for row in generated_rows]
    valid_english_set = set()
    
    # Calculate total number of batches for logging
    total_batches = (len(english_sentences_to_check) + batch_size - 1) // batch_size

    for i in range(0, len(english_sentences_to_check), batch_size):
        current_batch = (i // batch_size) + 1
        print(f"  > Processing batch {current_batch}/{total_batches}...", end="\r", flush=True)
        
        batch = english_sentences_to_check[i:i + batch_size]
        prompt = prompt_template.format(sentences_json=json.dumps(batch, indent=2))
        
        # Set counter variables
        retries = 0
        success = False
        response = None

        # Retry loop for different types of exceptions
        while retries <= max_retries and not success:
            try:
                response = model.generate_content(prompt, generation_config=generation_config)
                result_json = json.loads(response.text)
                valid_batch = result_json.get("valid_sentences", [])
                valid_english_set.update(valid_batch)
                success = True
            except ResourceExhausted as e:
                retries += 1
                print()
                if retries <= max_retries:
                    print(f"[Batch {current_batch}/{total_batches}] Rate limit hit (429). Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(retry_delay)
                    print(f"> Processing batch {current_batch}/{total_batches}...", end="\r", flush=True)
                else:
                    print(f"[Batch {current_batch}/{total_batches} FAILED] Max retries ({max_retries}) exceeded for rate limit error. Skipping batch.")
            except json.JSONDecodeError as e:
                print()
                print(f"[Batch {current_batch}/{total_batches} FAILED] Could not decode JSON from API.")
                print(f"> Error: {e}")
                if response:
                    print("  > --- Start of Failed API Response ---")
                    print(response.text)
                    print("  > --- End of Failed API Response ---")
                break
            except Exception as e:
                print()
                print(f"\n[Batch {current_batch}/{total_batches} FAILED] An unexpected error occurred.")
                print(f"  > Error: {e}")
                if response:
                    print("  > API Response (if available):", response.text)
                break

        time.sleep(1) # API rate limit

    print() 

    # Filter the original rows
    filtered_rows = [row for row in generated_rows if row['en'] in valid_english_set]
    
    print(f"Gemini filtering complete. Kept {len(filtered_rows)} out of {len(generated_rows)} sentences.")
    return filtered_rows
import os
import json
import random
from openai import AzureOpenAI
from tqdm import tqdm
# --- Helper Functions ---

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _make_api_call(client, model, messages, max_tokens):
    """Makes an API call to Azure OpenAI or DeepSeek API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        
        message = response.choices[0].message
        content = message.content
        
        # 特殊处理 DeepSeek-Reasoner 模型：当 content 为空时使用 reasoning_content
        if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
            content = message.reasoning_content
            print(f"DeepSeek reasoning content used: {len(content)} characters")
        
        if content is None:
            print("Warning: Both content and reasoning_content are None")
            return None
            
        return content.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# --- A2P Scaffolding Helper Functions ---

def construct_a2p_prompt(problem: str, ground_truth: str, chat_content: str, chat_history: list, index_agent: str, a2p: bool = False) -> str:
    """
    Constructs a prompt for failure attribution, optionally using A2P (Abduct-Act-Predict) scaffolding.
    
    Args:
        problem: The problem being solved
        ground_truth: The correct answer
        chat_content: Formatted conversation history
        chat_history: Raw chat history for detailed analysis
        index_agent: Key for agent identification ('role' or 'name')
        a2p: Whether to use A2P scaffolding framework
    
    Returns:
        Constructed prompt string
    """
    if not a2p:
        # Original prompt (baseline)
        return (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem}\n"
            f"The Answer for the problem is: {ground_truth}\n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction.\n\n"
            "CRITICAL: You MUST respond in EXACTLY the following format with NO deviation:\n"
            "Agent Name: [Agent name only]\n"
            "Step Number: [Number only]\n"
            "Reason for Mistake: [Your explanation here]\n\n"
            "Do NOT include any additional text, reasoning, or commentary outside of this format. "
            "Do NOT start with phrases like 'Looking at this conversation' or 'Based on my analysis'. "
            "Your response must begin immediately with 'Agent Name:' followed by the agent name, then 'Step Number:' with the number, then 'Reason for Mistake:' with your explanation. "
            "Keep your reason explanation concise and under 150 words."
        )
    
    # Enhanced A2P (Abduct-Act-Predict) scaffolding prompt
    # Build step-by-step context for better causal analysis
    step_context = []
    for idx, entry in enumerate(chat_history):
        agent_name = entry.get(index_agent, 'Unknown Agent')
        content = entry.get('content', '')
        step_context.append(f"Step {idx} - {agent_name}: {content}")
    
    structured_conversation = "\n".join(step_context)
    
    return (
        "You are an AI assistant performing failure analysis on a multi-agent conversation using the A2P (Abduct-Act-Predict) scaffolding framework. "
        "Multiple agents collaborated to solve a problem but produced an incorrect solution. "
        "Your task is to identify the root cause of failure using structured causal inference.\n\n"
        f"PROBLEM: {problem}\n"
        f"CORRECT ANSWER: {ground_truth}\n\n"
        "CONVERSATION HISTORY:\n"
        f"{structured_conversation}\n\n"
        "A2P SCAFFOLDING INSTRUCTIONS:\n"
        "Apply the A2P framework to identify the critical error through three sequential steps:\n\n"
        "STEP 1 - ABDUCTION (Infer Hidden Root Causes):\n"
        "For each agent's potentially problematic action, infer the hidden causal factors:\n"
        "- What knowledge gaps, misinterpretations, or flawed assumptions explain their behavior?\n"
        "- What latent variables (beliefs, misconceptions, missing information) led to their decision?\n"
        "- Identify the agent whose error represents the most plausible root cause of failure\n\n"
        "STEP 2 - ACTION (Define Minimal Corrective Intervention):\n"
        "For the identified critical error:\n"
        "- What specific action should the agent have taken instead?\n"
        "- Define the minimal, concrete intervention that addresses the root cause\n"
        "- How exactly would the correct action differ from what they actually did?\n\n"
        "STEP 3 - PREDICTION (Simulate Counterfactual Trajectory):\n"
        "Test the causal hypothesis by simulating the intervention:\n"
        "- Predict the next 3-5 conversational turns if the correct action had been taken\n"
        "- Would this counterfactual trajectory lead to the correct answer?\n"
        "- Trace the causal chain: intervention → intermediate effects → final success/failure\n\n"
        "FINAL ATTRIBUTION:\n"
        "Based on your A2P analysis, identify:\n"
        "1. The agent whose error is the decisive root cause of failure\n"
        "2. The step number where this critical error first occurred\n"
        "3. A clear explanation of the causal mechanism linking the error to failure\n\n"
        "CRITICAL OUTPUT FORMAT - You MUST respond EXACTLY as follows:\n"
        "Agent Name: [Agent name only]\n"
        "Step Number: [Number only]\n"
        "Reason for Mistake: [Your A2P-based explanation in under 150 words, focusing on the counterfactual reasoning]\n\n"
        "Remember: Use A2P scaffolding to perform rigorous counterfactual inference. The error you identify must be "
        "the decisive factor that, if corrected through your defined intervention, would change the outcome from failure to success."
    )

# --- All-at-Once Method ---

def all_at_once(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, a2p: bool = False):
    """
    Analyzes chat history by feeding the entire conversation at once to the model.
    
    Args:
        client: OpenAI client instance
        directory_path: Path to directory containing JSON files
        is_handcrafted: Whether dataset is hand-crafted
        model: Model identifier
        max_tokens: Maximum tokens for response
        a2p: Whether to use A2P (Abduct-Act-Predict) scaffolding framework
    """
    analysis_type = "All-at-Once with A2P Scaffolding" if a2p else "All-at-Once"
    print(f"\n--- Starting {analysis_type} Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for entry in chat_history
        ])

        # Use the A2P prompt construction function
        prompt = construct_a2p_prompt(
            problem=problem,
            ground_truth=ground_truth,
            chat_content=chat_content,
            chat_history=chat_history,
            index_agent=index_agent,
            a2p=a2p
        )

        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]

        result = _make_api_call(client, model, messages, max_tokens)

        print(f"Prediction for {json_file}:")
        if result:
            print(result)
        else:
            print("Failed to get prediction.")
        print("\n" + "="*50 + "\n")

# --- Step-by-Step Method ---

def step_by_step(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    """
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process.\n\n"
                "CRITICAL: You MUST respond in EXACTLY the following format with NO deviation:\n"
                "1. Yes\n"
                "2. Reason: [Your explanation here]\n\n"
                "OR\n\n"
                "1. No\n"
                "2. Reason: [Your explanation here]\n\n"
                "Do NOT include any additional text, reasoning, or commentary outside of this format. "
                "Do NOT start with phrases like 'Looking at this step' or 'We are evaluating'. "
                "Do NOT include long explanations before the numbered format. "
                "Your response must begin immediately with either '1. Yes' or '1. No' followed by '2. Reason:'. "
                "Keep your reason explanation concise and under 100 words."
            )

            messages=[
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                 print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "="*50 + "\n")


# --- Binary Search Method ---

def _construct_binary_search_prompt(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc):
    """Constructs the prompt for the binary search step."""
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n" # Included as per original code - remove if ground truth shouldn't be used
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that.\n\n"
        "CRITICAL: You MUST respond with EXACTLY one of these two phrases and NOTHING else:\n"
        "- upper half\n"
        "- lower half\n\n"
        "Do NOT include any additional text, reasoning, explanation, or commentary. "
        "Do NOT start with phrases like 'Based on my analysis' or 'Looking at this segment'. "
        "Do NOT include phrases like 'I believe' or 'It appears'. "
        "Your entire response must be exactly 'upper half' or 'lower half' (without quotes)."
    )

def _report_binary_search_error(chat_history, step, json_file, is_handcrafted):
    """Reports the identified error step from binary search."""
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {json_file}:")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_recursive(client: AzureOpenAI, model: str, max_tokens: int, chat_history: list, problem: str, answer: str, start: int, end: int, json_file: str, is_handcrafted: bool):
    """Recursive helper function for binary search analysis."""
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error(chat_history, end if end >= 0 else 0, json_file, is_handcrafted) # Report something reasonable
         return
    if start == end:
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if is_handcrafted else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Cannot proceed.")
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2 

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)

    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
        {"role": "user", "content": prompt}
    ]

    print(f"Analyzing step {start}-{end} for {json_file}...")
    result = _make_api_call(client, model, messages, max_tokens)

    if not result:
        print(f"API call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower() 

    if "upper half" in result_lower:
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted)
    else:
        print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}. Randomly choosing a half.")
        if random.randint(0, 1) == 0:
            print("Randomly chose upper half.")
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)
        else:
            print("Randomly chose lower half.")
            new_start = min(mid + 1, end)
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted)


def binary_search(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history using a binary search approach to find the error step.
    """
    print("\n--- Starting Binary Search Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, 0, len(chat_history) - 1, json_file, is_handcrafted)

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI, AsyncAzureOpenAI
from tqdm.asyncio import tqdm
import time
from .utils import construct_a2p_prompt

# --- Data Classes ---
@dataclass
class RequestTask:
    """Represents a single API request task"""
    index: int  # Original index for order preservation
    json_file: str
    prompt: str
    messages: List[Dict[str, str]]
    step: Optional[int] = None  # For step-by-step method
    agent_name: Optional[str] = None  # For step-by-step method
    
@dataclass
class ResponseResult:
    """Represents a response from the API"""
    index: int
    json_file: str
    result: Optional[str]
    step: Optional[int] = None
    agent_name: Optional[str] = None
    error: Optional[str] = None

@dataclass
class SampleTask:
    """Represents a sample-level task for step-by-step or binary search"""
    file_index: int
    json_file: str
    file_path: str
    data: Dict[str, Any]

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

async def _make_async_api_call(client, model: str, messages: List[Dict[str, str]], max_tokens: int) -> Optional[str]:
    """Makes an async API call to OpenAI-compatible API."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        
        message = response.choices[0].message
        content = message.content
        
        # Handle DeepSeek-Reasoner special case
        if content is None and hasattr(message, 'reasoning_content') and message.reasoning_content:
            content = message.reasoning_content
        
        if content is None:
            return None
            
        return content.strip()
    except Exception as e:
        print(f"Error during async API call: {e}")
        return None

class AsyncBatchProcessor:
    """Handles batch processing of API requests with concurrency control"""
    
    def __init__(self, client, model: str, max_tokens: int, batch_size: int = 48):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(batch_size)
        
    async def process_single_request(self, task: RequestTask) -> ResponseResult:
        """Process a single request with semaphore control"""
        async with self.semaphore:
            try:
                result = await _make_async_api_call(
                    self.client, 
                    self.model, 
                    task.messages, 
                    self.max_tokens
                )
                return ResponseResult(
                    index=task.index,
                    json_file=task.json_file,
                    result=result,
                    step=task.step,
                    agent_name=task.agent_name
                )
            except Exception as e:
                return ResponseResult(
                    index=task.index,
                    json_file=task.json_file,
                    result=None,
                    step=task.step,
                    agent_name=task.agent_name,
                    error=str(e)
                )
    
    async def process_batch(self, tasks: List[RequestTask]) -> List[ResponseResult]:
        """Process a batch of requests concurrently while maintaining order"""
        # Create tasks for concurrent execution
        coroutines = [self.process_single_request(task) for task in tasks]
        
        # Process with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Processing requests") as pbar:
            for coro in asyncio.as_completed(coroutines):
                result = await coro
                results.append(result)
                pbar.update(1)
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x.index)
        return results

# --- Async All-at-Once Method ---
async def all_at_once_async(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, batch_size: int = 48, a2p: bool = False):
    """
    Async version of all-at-once analysis with batch processing.
    
    Args:
        client: Async OpenAI client instance
        directory_path: Path to directory containing JSON files
        is_handcrafted: Whether dataset is hand-crafted
        model: Model identifier
        max_tokens: Maximum tokens for response
        batch_size: Number of concurrent requests
        a2p: Whether to use A2P (Abduct-Act-Predict) scaffolding framework
    """
    analysis_type = "Async All-at-Once with A2P Scaffolding" if a2p else "Async All-at-Once"
    print(f"\n--- Starting {analysis_type} Analysis ---\n")
    print(f"Batch size: {batch_size}")
    
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"
    
    # Prepare all tasks
    tasks = []
    task_index = 0
    
    for json_file in json_files:
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
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" 
            for entry in chat_history
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
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]
        
        tasks.append(RequestTask(
            index=task_index,
            json_file=json_file,
            prompt=prompt,
            messages=messages
        ))
        task_index += 1
    
    # Process all tasks in batches
    processor = AsyncBatchProcessor(client, model, max_tokens, batch_size)
    results = await processor.process_batch(tasks)
    
    # Output results in order
    for result in results:
        print(f"Prediction for {result.json_file}:")
        if result.result:
            print(result.result)
        else:
            print(f"Failed to get prediction. Error: {result.error}")
        print("\n" + "="*50 + "\n")

# --- Sample-level Async Step-by-Step Method ---
async def process_single_sample_step_by_step(client, sample_task: SampleTask, is_handcrafted: bool, model: str, max_tokens: int):
    """Process a single sample file step-by-step"""
    index_agent = "role" if is_handcrafted else "name"
    
    chat_history = sample_task.data.get("history", [])
    problem = sample_task.data.get("question", "")
    ground_truth = sample_task.data.get("ground_truth", "")
    
    if not chat_history:
        return {
            'file': sample_task.json_file,
            'error': 'No chat history found',
            'error_found': False
        }
    
    print(f"--- Analyzing File: {sample_task.json_file} ---")
    current_conversation_history = ""
    error_found = False
    result_info = {
        'file': sample_task.json_file,
        'error_found': False,
        'agent_name': None,
        'step': None,
        'reason': None
    }
    
    for idx, entry in enumerate(chat_history):
        agent_name = entry.get(index_agent, 'Unknown Agent')
        content = entry.get('content', '')
        current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"
        
        prompt = (
            f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
            f"The Answer for the problem is: {ground_truth}\n"
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
        
        messages = [
            {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
            {"role": "user", "content": prompt},
        ]
        
        print(f"Evaluating Step {idx} by {agent_name}...")
        answer = await _make_async_api_call(client, model, messages, max_tokens)
        
        if not answer:
            print("Failed to get evaluation for this step.")
            result_info['error'] = f"API failed at step {idx}"
            break
        
        print(f"LLM Evaluation: {answer}")
        
        if answer.lower().strip().startswith("1. yes"):
            print(f"\nPrediction for {sample_task.json_file}: Error found.")
            print(f"Agent Name: {agent_name}")
            print(f"Step Number: {idx}")
            reason = answer.split('Reason:', 1)[-1].strip() if 'Reason:' in answer else "N/A"
            print(f"Reason provided by LLM: {reason}")
            
            result_info['error_found'] = True
            result_info['agent_name'] = agent_name
            result_info['step'] = idx
            result_info['reason'] = reason
            break
        elif answer.lower().strip().startswith("1. no"):
            print("No significant error detected in this step.")
        else:
            print("Warning: Unexpected response format from LLM. Continuing evaluation.")
    
    if not result_info['error_found']:
        print(f"\nNo decisive errors found by step-by-step analysis in file {sample_task.json_file}")
    
    print("\n" + "="*50 + "\n")
    return result_info

async def step_by_step_async(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, batch_size: int = 48):
    """
    Sample-level async version of step-by-step analysis.
    Processes multiple samples concurrently, each sample's steps sequentially.
    """
    print("\n--- Starting Sample-Level Async Step-by-Step Analysis ---\n")
    print(f"Batch size (concurrent samples): {batch_size}")
    
    json_files = _get_sorted_json_files(directory_path)
    
    # Prepare sample tasks
    sample_tasks = []
    for idx, json_file in enumerate(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if data:
            sample_tasks.append(SampleTask(
                file_index=idx,
                json_file=json_file,
                file_path=file_path,
                data=data
            ))
    
    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(batch_size)
    
    async def process_with_semaphore(sample_task):
        async with semaphore:
            return await process_single_sample_step_by_step(
                client, sample_task, is_handcrafted, model, max_tokens
            )
    
    # Process all samples
    coroutines = [process_with_semaphore(task) for task in sample_tasks]
    results = []
    
    with tqdm(total=len(sample_tasks), desc="Processing samples") as pbar:
        for coro in asyncio.as_completed(coroutines):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    # Sort results by file order
    results.sort(key=lambda x: x['file'])
    
    # Summary
    errors_found = sum(1 for r in results if r['error_found'])
    print(f"\n--- Summary ---")
    print(f"Total samples processed: {len(results)}")
    print(f"Errors found: {errors_found}")

# --- Sample-level Async Binary Search Method ---
async def binary_search_recursive_async(client, model: str, max_tokens: int, chat_history: list, 
                                       problem: str, answer: str, start: int, end: int, 
                                       json_file: str, is_handcrafted: bool):
    """Async recursive helper for binary search"""
    if start > end:
        return {'step': end if end >= 0 else 0, 'agent': None}
    if start == end:
        index_agent = "role" if is_handcrafted else "name"
        entry = chat_history[start]
        agent_name = entry.get(index_agent, 'Unknown Agent')
        return {'step': start, 'agent': agent_name}
    
    index_agent = "role" if is_handcrafted else "name"
    segment_history = chat_history[start : end + 1]
    
    if not segment_history:
        return {'step': start, 'agent': chat_history[start].get(index_agent, 'Unknown Agent')}
    
    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])
    
    mid = start + (end - start) // 2
    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"
    
    prompt = (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. "
        "Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. "
        "Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user's query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        f"Review the following conversation segment {range_description}:\n\n{chat_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. "
        "Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. "
        "If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that.\n\n"
        "CRITICAL: You MUST respond with EXACTLY one of these two phrases and NOTHING else:\n"
        "- upper half\n"
        "- lower half\n\n"
        "Do NOT include any additional text, reasoning, explanation, or commentary. "
        "Do NOT start with phrases like 'Based on my analysis' or 'Looking at this segment'. "
        "Do NOT include phrases like 'I believe' or 'It appears'. "
        "Your entire response must be exactly 'upper half' or 'lower half' (without quotes)."
    )
    
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
        {"role": "user", "content": prompt}
    ]
    
    print(f"Analyzing step {start}-{end} for {json_file}...")
    result = await _make_async_api_call(client, model, messages, max_tokens)
    
    if not result:
        print(f"API call failed for segment {start}-{end}.")
        return {'step': start, 'agent': chat_history[start].get(index_agent, 'Unknown Agent')}
    
    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower()
    
    if "upper half" in result_lower:
        return await binary_search_recursive_async(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)
    elif "lower half" in result_lower:
        new_start = min(mid + 1, end)
        return await binary_search_recursive_async(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted)
    else:
        print(f"Warning: Ambiguous response '{result}'. Choosing upper half.")
        return await binary_search_recursive_async(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted)

async def process_single_sample_binary_search(client, sample_task: SampleTask, is_handcrafted: bool, model: str, max_tokens: int):
    """Process a single sample file using binary search"""
    chat_history = sample_task.data.get("history", [])
    problem = sample_task.data.get("question", "")
    answer = sample_task.data.get("ground_truth", "")
    
    if not chat_history:
        return {
            'file': sample_task.json_file,
            'error': 'No chat history found',
            'agent': None,
            'step': None
        }
    
    print(f"--- Analyzing File: {sample_task.json_file} ---")
    
    result = await binary_search_recursive_async(
        client, model, max_tokens, chat_history, problem, answer, 
        0, len(chat_history) - 1, sample_task.json_file, is_handcrafted
    )
    
    print(f"\nPrediction for {sample_task.json_file}:")
    print(f"Agent Name: {result['agent']}")
    print(f"Step Number: {result['step']}")
    print("\n" + "="*50 + "\n")
    
    return {
        'file': sample_task.json_file,
        'agent': result['agent'],
        'step': result['step']
    }

async def binary_search_async(client, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int, batch_size: int = 48):
    """
    Sample-level async version of binary search analysis.
    Processes multiple samples concurrently, each sample's binary search independently.
    """
    print("\n--- Starting Sample-Level Async Binary Search Analysis ---\n")
    print(f"Batch size (concurrent samples): {batch_size}")
    
    json_files = _get_sorted_json_files(directory_path)
    
    # Prepare sample tasks
    sample_tasks = []
    for idx, json_file in enumerate(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if data:
            sample_tasks.append(SampleTask(
                file_index=idx,
                json_file=json_file,
                file_path=file_path,
                data=data
            ))
    
    # Process samples with concurrency control
    semaphore = asyncio.Semaphore(batch_size)
    
    async def process_with_semaphore(sample_task):
        async with semaphore:
            return await process_single_sample_binary_search(
                client, sample_task, is_handcrafted, model, max_tokens
            )
    
    # Process all samples
    coroutines = [process_with_semaphore(task) for task in sample_tasks]
    results = []
    
    with tqdm(total=len(sample_tasks), desc="Processing samples") as pbar:
        for coro in asyncio.as_completed(coroutines):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    # Sort results by file order
    results.sort(key=lambda x: x['file'])
    
    # Summary
    print(f"\n--- Summary ---")
    print(f"Total samples processed: {len(results)}")

# --- Create Async Client ---
def create_async_client(model_type: str, **kwargs):
    """Create appropriate async client based on model type"""
    if model_type == 'gpt':
        return AsyncAzureOpenAI(
            api_key=kwargs.get('api_key'),
            api_version=kwargs.get('api_version'),
            azure_endpoint=kwargs.get('azure_endpoint')
        )
    elif model_type in ['gpt-oss', 'deepseek']:
        return AsyncOpenAI(
            api_key=kwargs.get('api_key'),
            base_url=kwargs.get('base_url')
        )
    else:
        raise ValueError(f"Unsupported model type for async: {model_type}")
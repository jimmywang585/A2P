"""
Independent evaluation of A2P inference outputs.
Derives metrics independently and compares against claimed results.
Inspired by RGCD evaluation methodology (who_and_when_metrics.py / eval_accuracy.py).
"""

import re
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================
# Paper's claimed results (Table from README / paper)
# ============================================================
CLAIMED_RESULTS = {
    "Algorithm-Generated": {
        "A2P (gpt-oss-120b)":       {"step": 47.46, "agent": None},
        "All-at-Once (baseline)":   {"step": 16.67, "agent": None},
        "Step-by-Step (baseline)":  {"step": 27.78, "agent": None},
        "Binary Search (baseline)": {"step": 28.57, "agent": None},
    },
    "Hand-Crafted": {
        "A2P (gpt-oss-120b)":       {"step": 29.31, "agent": None},
        "All-at-Once (baseline)":   {"step": 12.07, "agent": None},
        "Step-by-Step (baseline)":  {"step": 18.97, "agent": None},
        "Binary Search (baseline)": {"step": 13.79, "agent": None},
    },
}


def parse_predictions(eval_file):
    """Parse predictions from inference output file."""
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file not found at {eval_file}")
        return {}

    with open(eval_file, 'r', encoding='utf-8') as f:
        data = f.read()

    predictions = {}
    pattern = r"Prediction for ([^:]+\.json):(.*?)(?=Prediction for|\Z)"
    blocks = re.finditer(pattern, data, re.DOTALL)

    for block in blocks:
        idx = block.group(1).strip()
        content = block.group(2).strip()

        # Try multiple parsing patterns for robustness
        agent_name = None
        step_number = None

        # Pattern 1: "Agent Name: xxx"
        agent_match = re.search(r"Agent Name:\s*([\w_]+)", content, re.IGNORECASE)
        if agent_match:
            agent_name = agent_match.group(1)

        # Pattern 2: "**Agent Name:** xxx" (markdown bold)
        if not agent_name:
            agent_match = re.search(r"\*\*Agent Name:?\*\*\s*([\w_]+)", content, re.IGNORECASE)
            if agent_match:
                agent_name = agent_match.group(1)

        # Pattern 1: "Step Number: N"
        step_match = re.search(r"Step Number:\s*(\d+)", content, re.IGNORECASE)
        if step_match:
            step_number = int(step_match.group(1))

        # Pattern 2: "**Step Number:** N"
        if step_number is None:
            step_match = re.search(r"\*\*Step Number:?\*\*\s*(\d+)", content, re.IGNORECASE)
            if step_match:
                step_number = int(step_match.group(1))

        if agent_name is not None and step_number is not None:
            predictions[idx] = {
                'agent': agent_name,
                'step': step_number,
                'raw': content
            }
        else:
            # Check if "Failed to get prediction" 
            if "Failed to get prediction" in content:
                predictions[idx] = {'agent': None, 'step': None, 'raw': content, 'failed': True}
            else:
                predictions[idx] = {'agent': agent_name, 'step': step_number, 'raw': content, 'parse_error': True}

    return predictions


def load_ground_truth(data_path):
    """Load ground truth from JSON files."""
    ground_truth = {}
    json_files = sorted(
        [f for f in os.listdir(data_path) if f.endswith('.json')],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )

    for jf in json_files:
        fpath = os.path.join(data_path, jf)
        with open(fpath, 'r', encoding='utf-8') as f:
            d = json.load(f)

        history = d.get("history", [])
        ground_truth[jf] = {
            'agent': str(d.get('mistake_agent', '')),
            'step': int(d.get('mistake_step', -1)),
            'num_steps': len(history),
            'level': d.get('level', 'unknown'),
        }

    return ground_truth


def evaluate(predictions, ground_truth, dataset_name="Dataset"):
    """
    Comprehensive evaluation following RGCD-style metrics.
    Returns a dict of metrics.
    """
    n_total = len(ground_truth)
    n_parsed = sum(1 for p in predictions.values() if p.get('step') is not None)
    n_failed = sum(1 for p in predictions.values() if p.get('failed', False))
    n_parse_error = sum(1 for p in predictions.values() if p.get('parse_error', False))

    # Core metrics
    step_exact = 0
    step_within_1 = 0
    step_within_2 = 0
    step_within_5 = 0
    agent_exact = 0
    both_exact = 0

    # Distance distribution
    dist_counts = defaultdict(int)

    # Per-level breakdown
    level_metrics = defaultdict(lambda: {
        "total": 0, "step_exact": 0, "agent_exact": 0,
        "step_d1": 0, "step_d2": 0
    })

    # Step distance list for statistics
    step_distances = []

    for jf, gt in ground_truth.items():
        pred = predictions.get(jf)
        if pred is None or pred.get('step') is None:
            # Count as wrong prediction
            dist_counts["missing"] += 1
            level_metrics[gt['level']]["total"] += 1
            continue

        pred_step = pred['step']
        pred_agent = pred['agent']
        gt_step = gt['step']
        gt_agent = gt['agent']
        level = gt['level']

        level_metrics[level]["total"] += 1

        # Step distance
        step_dist = abs(pred_step - gt_step)
        step_distances.append(step_dist)

        if step_dist >= 10:
            dist_counts["10+"] += 1
        else:
            dist_counts[step_dist] += 1

        # Exact step match
        if pred_step == gt_step:
            step_exact += 1
            level_metrics[level]["step_exact"] += 1

        # Within-K step matches
        if step_dist <= 1:
            step_within_1 += 1
            level_metrics[level]["step_d1"] += 1
        if step_dist <= 2:
            step_within_2 += 1
            level_metrics[level]["step_d2"] += 1
        if step_dist <= 5:
            step_within_5 += 1

        # Agent match (substring match like A2P's evaluate.py does)
        if gt_agent and pred_agent and gt_agent in pred_agent:
            agent_exact += 1
            level_metrics[level]["agent_exact"] += 1
            if pred_step == gt_step:
                both_exact += 1

    # Compute percentages
    def pct(num, den):
        return f"{num}/{den} ({num / max(den, 1) * 100:.2f}%)" if den > 0 else "N/A"

    def pct_val(num, den):
        return num / max(den, 1) * 100

    # Compute average step distance
    avg_dist = sum(step_distances) / len(step_distances) if step_distances else float('inf')
    median_dist = sorted(step_distances)[len(step_distances) // 2] if step_distances else float('inf')

    # Average trace length
    trace_lens = [gt['num_steps'] for gt in ground_truth.values()]
    avg_trace_len = sum(trace_lens) / len(trace_lens) if trace_lens else 0

    results = {
        "dataset": dataset_name,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "n_parse_error": n_parse_error,
        "step_exact": step_exact,
        "step_exact_pct": pct_val(step_exact, n_total),
        "step_within_1": step_within_1,
        "step_within_1_pct": pct_val(step_within_1, n_total),
        "step_within_2": step_within_2,
        "step_within_2_pct": pct_val(step_within_2, n_total),
        "step_within_5": step_within_5,
        "step_within_5_pct": pct_val(step_within_5, n_total),
        "agent_exact": agent_exact,
        "agent_exact_pct": pct_val(agent_exact, n_total),
        "both_exact": both_exact,
        "both_exact_pct": pct_val(both_exact, n_total),
        "avg_step_distance": avg_dist,
        "median_step_distance": median_dist,
        "avg_trace_length": avg_trace_len,
        "random_baseline": 1 / avg_trace_len * 100 if avg_trace_len > 0 else 0,
        "dist_counts": dict(dist_counts),
        "level_metrics": dict(level_metrics),
    }

    # Print report
    print("=" * 70)
    print(f"  INDEPENDENT EVALUATION: {dataset_name}")
    print("=" * 70)
    print(f"  Total ground-truth files:    {n_total}")
    print(f"  Predictions parsed:          {n_parsed}")
    print(f"  API failures:                {n_failed}")
    print(f"  Parse errors:                {n_parse_error}")
    print(f"  Avg trace length:            {avg_trace_len:.1f} steps")
    print(f"  Random baseline (1/avg_len): {results['random_baseline']:.2f}%")
    print()

    print("  --- Step Accuracy (denominator = all {n_total} files) ---")
    print(f"    Exact (delta=0):    {pct(step_exact, n_total)}")
    print(f"    Within 1 (delta≤1): {pct(step_within_1, n_total)}")
    print(f"    Within 2 (delta≤2): {pct(step_within_2, n_total)}")
    print(f"    Within 5 (delta≤5): {pct(step_within_5, n_total)}")
    print()

    print("  --- Agent Accuracy ---")
    print(f"    Exact agent match:  {pct(agent_exact, n_total)}")
    print()

    print("  --- Joint Accuracy (both agent AND step correct) ---")
    print(f"    Both correct:       {pct(both_exact, n_total)}")
    print()

    print("  --- Step Distance Statistics ---")
    print(f"    Mean distance:      {avg_dist:.2f}")
    print(f"    Median distance:    {median_dist}")
    print()

    print("  --- Step Distance Distribution ---")
    for k in sorted([x for x in dist_counts if isinstance(x, int)]):
        bar = "█" * dist_counts[k]
        print(f"    dist={k:2d}: {dist_counts[k]:3d}  {bar}")
    if "10+" in dist_counts:
        bar = "█" * dist_counts["10+"]
        print(f"    dist=10+: {dist_counts['10+']:3d}  {bar}")
    if "missing" in dist_counts:
        print(f"    missing: {dist_counts['missing']:3d}")
    print()

    print("  --- By Difficulty Level ---")
    for level, m in sorted(level_metrics.items()):
        t = m["total"]
        print(f"    {level}: n={t}, "
              f"step_exact={pct(m['step_exact'], t)}, "
              f"step_d2={pct(m['step_d2'], t)}, "
              f"agent={pct(m['agent_exact'], t)}")
    print()

    return results


def compare_to_claimed(our_results, dataset_key):
    """Compare our metrics against the paper's claimed results."""
    claimed = CLAIMED_RESULTS.get(dataset_key, {})
    if not claimed:
        print(f"  No claimed results for {dataset_key}")
        return

    our_step = our_results["step_exact_pct"]
    our_agent = our_results["agent_exact_pct"]

    print("=" * 70)
    print(f"  COMPARISON VS. PAPER'S CLAIMED RESULTS ({dataset_key})")
    print("=" * 70)
    print(f"  {'Method':<30s} {'Step Acc':>10s} {'vs Ours':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")

    for method, metrics in claimed.items():
        claimed_step = metrics["step"]
        delta = our_step - claimed_step
        sign = "+" if delta >= 0 else ""
        print(f"  {method:<30s} {claimed_step:>9.2f}% {sign}{delta:>8.2f}%")

    print(f"  {'─'*50}")
    print(f"  {'Ours (GPT-4o-mini + A2P)':<30s} {our_step:>9.2f}%")
    print(f"  {'Ours Agent Accuracy':<30s} {our_agent:>9.2f}%")
    print()

    # Ratios
    baseline_step = claimed.get("All-at-Once (baseline)", {}).get("step", 0)
    a2p_claimed = claimed.get("A2P (gpt-oss-120b)", {}).get("step", 0)

    if baseline_step > 0:
        our_ratio = our_step / baseline_step
        claimed_ratio = a2p_claimed / baseline_step
        print(f"  Paper claimed improvement over All-at-Once baseline:")
        print(f"    Claimed: {a2p_claimed:.2f}% / {baseline_step:.2f}% = {claimed_ratio:.2f}×")
        print(f"    Ours:    {our_step:.2f}% / {baseline_step:.2f}% = {our_ratio:.2f}× (vs same baseline)")
        print()

    if a2p_claimed > 0:
        ratio_to_claimed = our_step / a2p_claimed
        print(f"  Our step accuracy vs. paper's A2P claim:")
        print(f"    {our_step:.2f}% / {a2p_claimed:.2f}% = {ratio_to_claimed:.2f}×")
        if ratio_to_claimed >= 0.9:
            print(f"    → Within 10% of claimed A2P performance")
        elif ratio_to_claimed >= 0.7:
            print(f"    → Within 30% of claimed A2P performance")
        else:
            print(f"    → Significantly below claimed A2P performance")
    print()


def main():
    output_dir = Path("outputs")
    data_alg = Path("Who&When/Algorithm-Generated")
    data_hc = Path("Who&When/Hand-Crafted")

    # Find all output files
    eval_files = list(output_dir.glob("*.txt")) if output_dir.exists() else []

    if not eval_files:
        print("No output files found in outputs/")
        sys.exit(1)

    print("╔" + "═" * 68 + "╗")
    print("║" + " INDEPENDENT A2P EVALUATION REPORT".center(68) + "║")
    print("║" + " (RGCD-style metrics)".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    for ef in sorted(eval_files):
        print(f"Processing: {ef}")

        # Determine dataset
        is_alg = "alg_generated" in ef.name
        is_hc = "handcrafted" in ef.name
        if is_alg:
            data_path = data_alg
            dataset_name = f"Algorithm-Generated [{ef.name}]"
            dataset_key = "Algorithm-Generated"
        elif is_hc:
            data_path = data_hc
            dataset_name = f"Hand-Crafted [{ef.name}]"
            dataset_key = "Hand-Crafted"
        else:
            print(f"  Skipping (unknown dataset type): {ef.name}")
            continue

        predictions = parse_predictions(str(ef))
        ground_truth = load_ground_truth(str(data_path))

        results = evaluate(predictions, ground_truth, dataset_name)
        compare_to_claimed(results, dataset_key)

    print("═" * 70)
    print("  EVALUATION COMPLETE")
    print("═" * 70)


if __name__ == "__main__":
    main()

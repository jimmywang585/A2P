#!/bin/bash
# Test script for A2P (Abduct-Act-Predict) scaffolding framework
# This script tests the A2P enhancement to the all_at_once method

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL="gpt-oss-120b"
METHOD="all_at_once"  # Using all_at_once with causal reasoning flag
MAX_TOKENS=20000
BATCH_SIZE=48
OUTPUT_DIR="outputs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Testing A2P Scaffolding Framework${NC}"
echo -e "${GREEN}Model: $MODEL${NC}"
echo -e "${GREEN}Method: $METHOD with --a2p${NC}"
echo -e "${GREEN}Timestamp: $TIMESTAMP${NC}"
echo -e "${GREEN}=========================================${NC}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Full Dataset Test with A2P Scaffolding
echo -e "\n${BLUE}Running on full datasets with A2P scaffolding...${NC}"

# Function to run enhanced method test
run_enhanced_test() {
    local dataset_type=$1
    local is_handcrafted=$2
    local dataset_path=$3
    local output_suffix=$4
    
    echo -e "${YELLOW}Running enhanced method on $dataset_type dataset...${NC}"
    
    # Define output filename for enhanced method
    local output_file="${OUTPUT_DIR}/${METHOD}_${MODEL}_a2p_${output_suffix}.txt"
    
    # Check if result file already exists
    if [ -f "$output_file" ]; then
        echo -e "${YELLOW}Warning: Result file already exists: $output_file${NC}"
        echo "Moving existing file to backup..."
        mv "$output_file" "${output_file}.backup_${TIMESTAMP}"
    fi
    
    echo "Command: python Automated_FA/inference.py --method $METHOD --model $MODEL --is_handcrafted $is_handcrafted --directory_path \"$dataset_path\" --max_tokens $MAX_TOKENS --use_async --batch_size $BATCH_SIZE --a2p"
    
    # Run the enhanced method with A2P scaffolding
    python Automated_FA/inference.py \
        --method "$METHOD" \
        --model "$MODEL" \
        --is_handcrafted "$is_handcrafted" \
        --directory_path "$dataset_path" \
        --max_tokens "$MAX_TOKENS" \
        --use_async \
        --batch_size "$BATCH_SIZE" \
        --a2p
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Completed: Enhanced method on $dataset_type${NC}"
        
        # Run evaluation immediately after generation
        echo -e "${YELLOW}Evaluating enhanced results for $dataset_type...${NC}"
        python Automated_FA/evaluate.py \
            --data_path "$dataset_path" \
            --eval_file "$output_file" 2>&1 | tee "${OUTPUT_DIR}/eval_${output_suffix}_${TIMESTAMP}.log"
    else
        echo -e "${RED}✗ Failed: Enhanced method on $dataset_type${NC}"
        return 1
    fi
}

# Test on Algorithm-Generated Dataset
echo -e "\n${GREEN}========== Algorithm-Generated Dataset ==========${NC}"
run_enhanced_test "Algorithm-Generated" "False" "Who&When/Algorithm-Generated" "alg_generated"

# Test on Hand-Crafted Dataset
echo -e "\n${GREEN}========== Hand-Crafted Dataset ==========${NC}"
run_enhanced_test "Hand-Crafted" "True" "Who&When/Hand-Crafted" "handcrafted"

# Step 2: Comparative Analysis
echo -e "\n${BLUE}Step 2: Comparative Analysis...${NC}"
echo -e "${YELLOW}Comparing baseline vs enhanced method results...${NC}"

# Display baseline results (if they exist)
echo -e "\n${YELLOW}Baseline Results:${NC}"
if [ -f "${OUTPUT_DIR}/all_at_once_${MODEL}_alg_generated.txt" ]; then
    echo "Algorithm-Generated baseline exists"
    python Automated_FA/evaluate.py \
        --data_path "Who&When/Algorithm-Generated" \
        --eval_file "${OUTPUT_DIR}/all_at_once_${MODEL}_alg_generated.txt" 2>&1 | grep -E "Agent Accuracy|Step Accuracy"
fi

if [ -f "${OUTPUT_DIR}/all_at_once_${MODEL}_handcrafted.txt" ]; then
    echo "Hand-Crafted baseline exists"
    python Automated_FA/evaluate.py \
        --data_path "Who&When/Hand-Crafted" \
        --eval_file "${OUTPUT_DIR}/all_at_once_${MODEL}_handcrafted.txt" 2>&1 | grep -E "Agent Accuracy|Step Accuracy"
fi

echo -e "\n${YELLOW}Enhanced Method Results:${NC}"
echo "See evaluation logs above for detailed accuracy metrics"

# Step 3: Generate Summary Report
echo -e "\n${BLUE}Step 3: Generating Summary Report...${NC}"

cat > "${OUTPUT_DIR}/test_summary_${TIMESTAMP}.md" << EOF
# Test Summary for A2P (Abduct-Act-Predict) Scaffolding Framework
**Timestamp:** ${TIMESTAMP}
**Model:** ${MODEL}
**Method:** ${METHOD} with --a2p

## Test Results
- Algorithm-Generated Dataset: COMPLETED
- Hand-Crafted Dataset: COMPLETED

## Output Files
- Enhanced Algorithm-Generated: ${OUTPUT_DIR}/${METHOD}_${MODEL}_a2p_alg_generated.txt
- Enhanced Hand-Crafted: ${OUTPUT_DIR}/${METHOD}_${MODEL}_a2p_handcrafted.txt
- Evaluation Logs: ${OUTPUT_DIR}/eval_*_${TIMESTAMP}.log

## Next Steps
1. Review the evaluation metrics in the logs
2. Compare with baseline results
3. Document improvements in Result.md
EOF

echo -e "${GREEN}Summary report saved to: ${OUTPUT_DIR}/test_summary_${TIMESTAMP}.md${NC}"

# Final Summary
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Testing complete for method: A2P Scaffolding Framework${NC}"
echo -e "${GREEN}All test outputs saved in: $OUTPUT_DIR${NC}"
echo -e "${GREEN}================================================${NC}"

# Display key metrics if available
echo -e "\n${BLUE}Key Metrics Summary:${NC}"
if [ -f "${OUTPUT_DIR}/${METHOD}_${MODEL}_a2p_alg_generated.txt" ]; then
    echo -e "\nAlgorithm-Generated Dataset:"
    python Automated_FA/evaluate.py \
        --data_path "Who&When/Algorithm-Generated" \
        --eval_file "${OUTPUT_DIR}/${METHOD}_${MODEL}_a2p_alg_generated.txt" 2>&1 | tail -5
fi

if [ -f "${OUTPUT_DIR}/${METHOD}_${MODEL}_a2p_handcrafted.txt" ]; then
    echo -e "\nHand-Crafted Dataset:"
    python Automated_FA/evaluate.py \
        --data_path "Who&When/Hand-Crafted" \
        --eval_file "${OUTPUT_DIR}/${METHOD}_${MODEL}_a2p_handcrafted.txt" 2>&1 | tail -5
fi
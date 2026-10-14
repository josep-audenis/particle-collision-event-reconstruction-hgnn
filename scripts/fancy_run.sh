#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

run_config() {
    local script=$1
    local config=$2
    
    echo -e "${BLUE}Running: $script --config $config${NC}"
    
    if python "$script" --config "$config"; then
        echo -e "${GREEN}[OK] SUCCESS: $config${NC}\n"
        return 0
    else
        echo -e "${RED}[FAIL] FAILED: $config${NC}\n"
        return 1
    fi
}

FAILED_CONFIGS=()
SUCCESSFUL_CONFIGS=()

echo "========================================"
echo "Starting experiment runs..."
echo "========================================"
echo ""

echo "=== DFEI Training ==="
configs=(
    "train_DFEI_b_daughters_only.yaml"
    "train_DFEI_cpt.yaml"
    "train_DFEI_full.yaml"
    "train_DFEI_PVassoTrue.yaml"
    "train_DFEI_PVassoIP.yaml"
    "train_DFEI_PVassoDFEI.yaml"
)

for config in "${configs[@]}"; do
    if run_config "../wmpgnn/analysis/trainer.py" "$config"; then
        SUCCESSFUL_CONFIGS+=("$config")
    else
        FAILED_CONFIGS+=("$config")
    fi
done

echo "=== DFEI Evaluation ==="
if run_config "../wmpgnn/analysis/evaluate.py" "evaluate_DFEI.yaml"; then
    SUCCESSFUL_CONFIGS+=("evaluate_DFEI.yaml")
else
    FAILED_CONFIGS+=("evaluate_DFEI.yaml")
fi

echo "=== IFT Training ==="
configs=(
    "train_IFT_PVassoTrue.yaml"
    "train_IFT_PVassoIP.yaml"
    "train_IFT_PVassoDFEI.yaml"
    "train_IFT_PVassoDFEI_DFEI.yaml"
)

for config in "${configs[@]}"; do
    if run_config "../wmpgnn/analysis/trainer.py" "$config"; then
        SUCCESSFUL_CONFIGS+=("$config")
    else
        FAILED_CONFIGS+=("$config")
    fi
done

echo "========================================"
echo "Execution Summary"
echo "========================================"
echo ""

if [ ${#SUCCESSFUL_CONFIGS[@]} -gt 0 ]; then
    echo -e "${GREEN}Successful configs (${#SUCCESSFUL_CONFIGS[@]}):${NC}"
    for config in "${SUCCESSFUL_CONFIGS[@]}"; do
        echo -e "  ${GREEN}[OK]${NC} $config"
    done
    echo ""
fi

if [ ${#FAILED_CONFIGS[@]} -gt 0 ]; then
    echo -e "${RED}Failed configs (${#FAILED_CONFIGS[@]}):${NC}"
    for config in "${FAILED_CONFIGS[@]}"; do
        echo -e "  ${RED}[FAIL]${NC} $config"
    done
    echo ""
    exit 1
else
    echo -e "${GREEN}All configurations executed successfully!${NC}"
    exit 0
fi

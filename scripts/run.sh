#!/bin/bash
python ../wmpgnn/analysis/trainer.py --config train_DFEI_b_daughters_only.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_cpt.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_full.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_PVassoTrue.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_PVassoIP.yaml
python ../wmpgnn/analysis/trainer.py --config train_DFEI_PVassoDFEI.yaml
python ../wmpgnn/analysis/evaluate.py --config evaluate_DFEI.yaml

python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoTrue.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoIP.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoDFEI.yaml
python ../wmpgnn/analysis/trainer.py --config train_IFT_PVassoDFEI_DFEI.yaml

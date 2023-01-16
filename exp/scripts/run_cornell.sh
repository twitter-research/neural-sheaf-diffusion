#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=True \
    --d=4 \
    --dataset=cornell \
    --dropout=0.7 \
    --early_stopping=200 \
    --epochs=500 \
    --folds=10 \
    --hidden_channels=16 \
    --input_dropout=0.2 \
    --layers=2 \
    --lr=0.02 \
    --model=DiagSheaf \
    --sheaf_decay=0.00031764232712732976 \
    --weight_decay=0.0006914841722570725 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True
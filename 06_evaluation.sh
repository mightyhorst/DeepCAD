#!/bin/bash
echo "🕵️‍♀️ Evaluation..."
cd evaluation 
bash run_eval_gen.sh ../proj_log/pretrained/lgan_1000/results/fake_z_ckpt200000_num9000_dec 1000 0

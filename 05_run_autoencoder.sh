#!/bin/bash
echo "🏃‍♂️ Run Autoencoder..."
echo "• run the autoencoder to decode into final CAD sequences"
!python test.py --exp_name pretrained --mode dec --ckpt 1000 --z_path proj_log/pretrained/lgan_1000/results/fake_z_ckpt200000_num9000.h5 -g 0

#!/bin/bash
echo "🏃‍♂️ Run LGAN..."
echo " • run latent GAN to generate fake latent vectors"
python lgan.py --exp_name pretrained --ae_ckpt 1000 --ckpt 200000 --test --n_samples 9000 -g 0

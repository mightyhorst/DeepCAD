#!/bin/bash
echo "🗂️ creating/opening proj_log directory..."
mkdir -p proj_log
cd proj_log
echo "🗑️ Removing old pretrained model..."
rm -rf pretrained
echo "👇 Downloading pretrained model..."
wget http://www.cs.columbia.edu/cg/deepcad/pretrained.tar

echo "🪴 Unzipping pretrained model..."
tar -xzf pretrained.tar && rm pretrained.tar

echo "🗂️ proj_log/pretrained"
echo " + 🗂️ lgan_1000/model"
echo "   + 📄 ckpt_epoch200000.pth"
echo "      • the checkpoint weights for the Latent GAN trained for 200k epochs"
echo " + 🗂️ model "
echo "   + 📄 ckpt_epoch1000.pth"
echo "      • the checkpoint weights for the autoencoder trained for 1k epochs"

ls pretrained 
ls pretrained/model 
ls pretrained/lgan_1000/model

#!/bin/bash
echo "🗑️ Removing old data directory ..."
rm -rf data
echo "👇 Downloading data..."
wget http://www.cs.columbia.edu/cg/deepcad/data.tar
tar -xf data.tar
rm data.tar

echo "🪴 Unzipping cad json..."
cd data
tar -xzf cad_json.tar.gz && rm cad_json.tar.gz

echo "🪴 Unzipping cad vec..."
tar -xzf cad_vec.tar.gz && rm cad_vec.tar.gz

echo "🗂️ The data directory has the following structure:"
echo " + 📁 cad_json: "
echo "      • the json files from Onshape"
echo " + 📁 cad_vec: "
echo "      • the json files converted to vectors"
echo "      • this is produced by running 👉dataset/json2vec.py"
echo " + 📄 train_val_test_split.json "
echo "      • the train and test split"
ls -U | head -5

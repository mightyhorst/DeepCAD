#!/bin/bash
clear
export ENV_NAME="deep-cad-env"
echo "🥝 Creating $ENV_NAME environment..."

# Deactivate the current environment (if any)
echo "🥶 Deactivate current environment..."
echo "👉 to deactivate run --> source ~/miniconda3/etc/profile.d/conda.sh"
echo "👉 to deactivate run --> conda deactivate"

# Remove the environment if it exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "🗑️ Remove the $ENV_NAME environment..."
    conda env remove --name "$ENV_NAME" --yes
fi

# Create the environment from environment.yml
echo "💡 Create the $ENV_NAME environment..."
conda env create --file environment.yml --name "$ENV_NAME"

# Activate the new environment
echo "✅ Activating $ENV_NAME environment..."
echo "👉 to activate run --> conda activate $ENV_NAME"

echo ""
echo "👇 Please run the following command..."
echo "------------------------"
echo "conda activate $ENV_NAME"
echo "------------------------"
echo ""

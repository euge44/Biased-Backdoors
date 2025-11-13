#!/bin/bash

# Automated setup script for WebShop text environment
# Run this from the WebShop directory: ./setup_fixed.sh

set -e  # Exit on error

# Create conda environment
eval "$(conda shell.bash hook)"
conda create -n webshop python=3.8.13 -y

# Activate environment (note: may need to run manually in new shell)
echo "Activate the environment with: conda activate webshop"

# Install dependencies with specific versions
conda activate webshop
pip install charset-normalizer==2.1.1
pip install pydantic==1.7.4
pip install huggingface-hub==0.10.1
pip install transformers==4.19.2
pip install gdown
pip install tqdm rich
pip install rank_bm25
pip install thefuzz
pip install torch==1.11.0
pip install Flask==2.1.2 Werkzeug==2.2.2
pip install gym==0.24.0

# Download data
mkdir -p data
cd data
gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib  # items_shuffle_1000.json
gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu  # items_ins_v2_1000.json
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O  # items_human_ins.json
cd ..

# Make scripts executable
chmod +x run_web_agent_text_env.sh

# Note: Code modifications are required (see comments below)
# 1. In web_agent_site/envs/__init__.py: comment out 'from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv'
# 2. In web_agent_site/engine/engine.py: replace Lucene with BM25 (see previous changes)
# 3. In web_agent_site/engine/goal.py: remove spacy, replace nlp usage with simple split
# 4. In web_agent_site/envs/web_agent_text_env.py: change import to direct

# Run conversion 
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh #TODO: solve issue with run_indexing
cd ..

# Create logging folder + samples of log data => needed for backdoor dataset creation :) 
get_human_trajs () {
  PYCMD=$(cat <<EOF
import gdown
url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
gdown.download_folder(url, quiet=True, remaining_ok=True)
EOF
  )
  python -c "$PYCMD"
}
mkdir -p user_session_logs/
cd user_session_logs/
echo "Downloading 50 example human trajectories..."
get_human_trajs
echo "Downloading example trajectories complete"
cd ..

# Run the environment
echo "Setup complete. Run: conda activate webshop && ./run_web_agent_text_env.sh"

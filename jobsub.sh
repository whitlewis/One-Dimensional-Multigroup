#!/bin/bash

#$ -M wlewis7@nd.edu
#$ -m abe
#$ -pe smp 24        # Specify parallel environment and legal core size
#$ -q long           # Specify queue
#$ -N MultiGroup       # Specify job name

# Required modules
module load python/3.12.15

# Add your local pip install path (adjust version if needed)
setenv PATH "$HOME/.local/bin:$PATH"
setenv PYTHONPATH "$HOME/.local/lib/python3.12/site-packages"

python3 RunSet.py  # Application to execute

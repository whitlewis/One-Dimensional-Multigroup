#$ !/bin/bash

#$ -M wlewis7@nd.edu   # Email address for job notificatio
#$ -m a            # Send mail when job begins, ends and aborts
#$ -pe smp 24        # Specify parallel environment and legal core size
#$ -q long           # Specify queue
#$ -N Batch Multigroup       # Specify job name

module load python      # Required modules

python RunSet.py  # Application to execute

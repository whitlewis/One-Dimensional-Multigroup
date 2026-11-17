#!/bin/bash
#$ -cwd

#$ -M wlewis7@nd.edu
#$ -m a
#$ -pe smp 24
#$ -q long
#$ -N Batch_Multigroup

module load python

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_env="${PYTHON_ENV:-$script_dir/.venv}"

if [[ ! -x "$python_env/bin/python" ]]; then
    echo "Python environment not found: $python_env" >&2
    echo "Create it with: python3 -m venv \"$python_env\"" >&2
    echo "Then install dependencies with: \"$python_env/bin/python\" -m pip install -r \"$script_dir/requirements-crc.txt\"" >&2
    exit 1
fi

cd "$script_dir"
export PYTHONNOUSERSITE=1
exec "$python_env/bin/python" "$script_dir/RunSet.py"

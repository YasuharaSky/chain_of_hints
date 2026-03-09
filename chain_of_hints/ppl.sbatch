#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1       
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL
python ppl.py
python confusion.py

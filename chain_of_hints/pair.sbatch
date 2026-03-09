#!/bin/bash
#SBATCH --job-name=ppl
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1       
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL
python ppl.py

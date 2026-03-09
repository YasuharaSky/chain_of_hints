#!/bin/bash
#SBATCH --job-name=path-stats
#SBATCH --partition=general 
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1       
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL
python overall-stats.py

#!/bin/sh
#SBATCH --job-name=embed
#SBATCH --partition=general
#SBATCH --cpus-per-task=2      
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:1 
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL
#SBATCH --output=logs/run.out   
#SBATCH --error=logs/run.err

python embed2.py
#!/bin/bash
#SBATCH --job-name=tree
#SBATCH --partition=general    
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=12:00:00        
#SBATCH --mail-type=ALL 
#SBATCH --error=logs/tot.err
python tree-of-thought.py

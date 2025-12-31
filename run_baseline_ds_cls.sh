#!/bin/bash
#SBATCH --job-name=swap_train
#SBATCH -t 2-00:00
#SBATCH -o /workspace1/asoto/araymond/swap_pixels/exp_logs/%x_%j.out
#SBATCH -e /workspace1/asoto/araymond/swap_pixels/exp_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=afraymon@uc.cl
#SBATCH --chdir=/home/araymond
#SBATCH --partition=ialab
#SBATCH --nodelist=ventress
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G

# Activa tu env si aplica
# source activate tu_env

cd /workspace1/asoto/araymond/swap_pixels

# 🟢 1. Lógica para definir el nombre del experimento
# Si use_pretrained no está definido, asume false por seguridad
USE_PT=${use_pretrained:-false}

if [ "$USE_PT" = "true" ]; then
    STATUS="pt"
    PT_FLAG="--use_pretrained_encoder" # Flag para Python
else
    STATUS="scratch"
    PT_FLAG="" # Sin flag (false por defecto en tu config)
fi

# 🟢 2. Definimos el nombre COMÚN del experimento (SIN EL SEED)
# Ejemplo: resnet18_swap_pretrained
EXP_NAME="dsprites_${model}_${method}_${STATUS}_full"

echo "Iniciando Job para Experimento: $EXP_NAME | Seed: $seed"
python train.py --dataset dsprites --idg_root /workspace1/asoto/araymond/datasets \
  --idg_split composition \
  --out_dir runs/ \
  --run_name "$EXP_NAME" \
  --inv_factors pos_x \
  --device cuda \
  --encoder $model \
  --z_dim 256 \
  --train_mode $method \
  --epochs 100 \
  --batch_size 256 \
  --num_workers 0 \
  --val_percent 0.1 \
  --log_every 200 \
  --save_every 10 \
  --lr 3e-4 --weight_decay 1e-4 \
  --pair_seed 777 \
  --seed $seed \
  --use_classification \
  --auto_resume \
  \
  $PT_FLAG  # <-- Aquí se inyecta el flag si corresponde

echo "Finished with job $SLURM_JOBID"

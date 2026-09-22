#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2022-2024 ArcaNN developers group <https://github.com/arcann-chem>                     #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2026/02/25
# Last modified: 2026/02/25
#----------------------------------------------
# Project/Account
#SBATCH --account=_R_PROJECT_@_R_ALLOC_
# QoS/Partition/SubPartition
#SBATCH --qos=_R_QOS_
#SBATCH --partition=_R_PARTITION_
#SBATCH -C _R_SUBPARTITION_
# Number of Nodes/MPIperNodes/OpenMPperMPI/GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o MACE_Comp.%j
#SBATCH -e MACE_Comp.%j
# Name of job
#SBATCH -J MACE_Comp
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

#----------------------------------------------
# Files / Variables
#----------------------------------------------


#echo '-<env>----------------------------------------------'
#env | sort
#echo '-</env>---------------------------------------------'

MACE_MODEL_FILE="_R_MACE_MODEL_FILE_"
MACE_MODEL_STYLE="_R_MACE_MODEL_STYLE_"

#----------------------------------------------
# Nothing needed to be changed past this point

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1
# Load the environment depending on the version
module purge
module load arch/h100
module load pytorch-gpu/py3/
conda activate franken

franken.wrap_mace_lammps --model_path=${MACE_MODEL_FILE}


sleep 5
exit 0
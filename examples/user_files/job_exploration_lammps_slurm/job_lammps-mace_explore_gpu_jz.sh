#!/bin/bash
#----------------------------------------------------------------------------------------------------#
#   ArcaNN: Automatic training of Reactive Chemical Architecture with Neural Networks                #
#   Copyright 2023 ArcaNN developers group <https://github.com/arcann-chem>                          #
#                                                                                                    #
#   SPDX-License-Identifier: AGPL-3.0-only                                                           #
#----------------------------------------------------------------------------------------------------#
# Created: 2022/01/01
# Last modified: 2023/09/06
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
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
# Walltime
#SBATCH -t _R_WALLTIME_
# Merge Output/Error
#SBATCH -o LAMMPS_MACE.%j
#SBATCH -e LAMMPS_MACE.%j
# Name of job
#SBATCH -J LAMMPS_MACE
# Email
#SBATCH --mail-type FAIL,BEGIN,END,ALL
#SBATCH --mail-user _R_EMAIL_
#

# Input file (extension is automatically added as .in for INPUT)
MACE_MODEL_VERSION="_R_MACE_VERSION_"
MACE_MODEL_FILES=("_R_MODEL_FILES_")
LAMMPS_IN_FILE="_R_LAMMPS_IN_FILE_"
LAMMPS_LOG_FILE="_R_LAMMPS_LOG_FILE_"
LAMMPS_OUT_FILE="_R_LAMMPS_OUT_FILE_"
EXTRA_FILES=("_R_DATA_FILE_" "_R_PLUMED_FILES_" "_R_RERUN_FILE_")

#----------------------------------------------
# Nothing needed to be changed past this point

# Project Switch and update SCRATCH
PROJECT_NAME=${SLURM_JOB_ACCOUNT:0:3}
eval "$(idrenv -d "${PROJECT_NAME}")"
# Compare PROJECT_NAME and IDRPROJ for inequality
if [[ "${PROJECT_NAME}" != "${IDRPROJ}" ]]; then
    SCRATCH=${NEWSCRATCH/${IDRPROJ}/${PROJECT_NAME}}
fi

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || exit 1

# Check
[ -f "${LAMMPS_IN_FILE}" ] || { echo "${LAMMPS_IN_FILE} does not exist. Aborting..."; exit 1; }

# Load the environment depending on the version
if [ "${SLURM_JOB_QOS:4:3}" == "gpu" ]; then
    if [ "${MACE_MODEL_VERSION}" == "0.3.14" ]; then
        MACE_INSTALL="/lustre/fsn1/worksf/projects/rech/nvs/uht29vt/LAMMPS-SYMMETRIX-PLUMED/env"
        module purge
        module load arch/h100
        module load gcc/12.2.0 cuda/12.8.0 openmpi/4.1.6-cuda
        module load cudnn/9.21.0.82-cuda fftw/3.3.10-mpi-cuda bzip2/1.0.8 ffmpeg/8.1-cuda hdf5/1.12.0-mpi-cuda libpng/1.6.37 netcdf-c/4.7.4-mpi-cuda libjpeg-turbo/2.1.3 gsl/2.7.1 openblas/0.3.20

        conda activate $MACE_INSTALL
        export LD_LIBRARY_PATH="$MACE_INSTALL/lib":$LD_LIBRARY_PATH
	#. /gpfswork/rech/nvs/commun/programs/apps/deepmd-kit/2.1.4-cuda11.6_plumed-2.8.0/etc/profile.d/plumed.sh
    elif [ "${MACE_MODEL_VERSION}" == "2.1" ] ; then
         module purge
    else
        echo "MACE ${MACE_MODEL_VERSION} is not installed on ${SLURM_JOB_QOS}. Aborting..."; exit 1
    fi
elif [ "${SLURM_JOB_QOS:3:4}" == "cpu" ]; then
    echo "GPU on a CPU partition?? Aborting..."; exit 1
else
    echo "There is no ${SLURM_JOB_QOS}. Aborting..."; exit 1
fi

# Set the temporary work directory
export TEMPWORKDIR=${NEWSCRATCH}/JOB-${SLURM_JOBID}
mkdir -p "${TEMPWORKDIR}"
ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}"/JOB-"${SLURM_JOBID}"

cp "${LAMMPS_IN_FILE}" "${TEMPWORKDIR}" && echo "${LAMMPS_IN_FILE} copied successfully"
for f in "${MACE_MODEL_FILES[@]}"; do [ -f "${f}" ] && ln -s "$(realpath "${f}")" "${TEMPWORKDIR}" && echo "${f} linked successfully"; done
for f in "${EXTRA_FILES[@]}"; do [ -f "${f}" ] && cp "${f}" "${TEMPWORKDIR}" && echo "${f} copied successfully"; done

# Go to the temporary work directory
cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }

echo "# [$(date)] Running LAMMPS..."
lmp -k on t $SLURM_CPU_PER_TASK g $SLURM_JOB_GPUS -sf kk -pk kokkos newton on neigh half -in "${LAMMPS_IN_FILE}" -log "${LAMMPS_LOG_FILE}" -screen none > "${LAMMPS_OUT_FILE}" 2>&1
echo "# [$(date)] LAMMPS finished."

# Move back data from the temporary work directory and scratch, and clean-up
if [ -f log.cite ]; then rm log.cite ; fi
find ./ -type l -delete
mv ./* "${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

sleep 2
exit

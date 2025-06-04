#!/bin/bash

# 2 beta minus
# 0 nu
# FINAL_DIR="ANDT_2betaMinus/KY/0nu/gs_to_gs/LNE"
# NUCLEI=("48Ca" "76Ge" "82Se" "96Zr" "100Mo" "110Pd" "116Cd" "124Sn" "128Te" "130Te" "136Xe" "148Nd" "150Nd" "154Sm" "160Gd" "198Pt" "232Th" "238U")
# CONF_FILE_TEMPLATE="config_files/0nu_2betaMinus_template.yaml"
#
# 2 nu
TRANSITION="0->2"
FINAL_DIR="AME2020_ENSDF2025/2betaMinus/2nu/0_to_2"
NUCLEI=("48Ca" "76Ge" "82Se" "96Zr" "100Mo" "110Pd" "116Cd" "124Sn" "128Te" "130Te" "136Xe" "148Nd" "150Nd" "154Sm" "160Gd" "198Pt" "232Th" "238U")
CONF_FILE_TEMPLATE="config_files/2nu_2betaMinus_template.yaml"

# 2EC
# 2 nu
#FINAL_DIR="ANDT_2EC/KY/2nu/gs_to_gs"
#NUCLEI=("36Ar" "40Ca" "50Cr" "54Fe" "58Ni" "64Zn" "74Se" "78Kr" "84Sr" "92Mo" "96Ru" "102Pd" "106Cd" "108Cd" "120Te" "124Xe" "126Xe" "130Ba" "132Ba" "136Ce" "138Ce" "144Sm" "152Gd" "156Dy" "158Dy" "162Er" "164Er" "168Yb" "174Hf" "180W" "184Os" "190Pt" "196Hg")
#CONF_FILE_TEMPLATE="config_files/2nu_2EC_template.yaml"
#

# EC betaPlus
# 2 nu
# FINAL_DIR="ANDT_ECbetaPlus/KY/2nu/gs_to_gs"
# NUCLEI=("50Cr" "58Ni" "64Zn" "74Se" "78Kr" "84Sr" "92Mo" "96Ru" "102Pd" "106Cd" "120Te" "124Xe" "130Ba" "136Ce" "144Sm" "156Dy" "162Er" "168Yb" "174Hf" "184Os" "190Pt" )
# CONF_FILE_TEMPLATE="config_files/2nu_ECbetaPlus_template.yaml"
#
# 0nu
# FINAL_DIR="ANDT_ECbetaPlus/KY/0nu/gs_to_gs/LNE"
# NUCLEI=("50Cr" "58Ni" "64Zn" "74Se" "78Kr" "84Sr" "92Mo" "96Ru" "102Pd" "106Cd" "120Te" "124Xe" "130Ba" "136Ce" "144Sm" "156Dy" "162Er" "168Yb" "174Hf" "184Os" "190Pt" )
# CONF_FILE_TEMPLATE="config_files/0nu_ECbetaPlus_template.yaml"
#

# 2 beta plus
# 0 nu
# FINAL_DIR="ANDT_2betaPlus/KY/0nu/gs_to_gs/LNE"
# NUCLEI=("78Kr" "96Ru" "106Cd" "124Xe" "130Ba" "136Ce")
# CONF_FILE_TEMPLATE="config_files/0nu_2betaPlus_template.yaml"
#
# 2nu
# FINAL_DIR="ANDT_2betaPlus/KY/2nu/gs_to_gs"
# NUCLEI=("78Kr" "96Ru" "106Cd" "124Xe" "130Ba" "136Ce")
# CONF_FILE_TEMPLATE="config_files/2nu_2betaPlus_template.yaml"

for NUC in "${NUCLEI[@]}"
do
    echo "Computing for ${NUC}"
    TMP_DIR="${FINAL_DIR}/${NUC}"
    rm -rf "${TMP_DIR}"
    mkdir -p  "${TMP_DIR}"
    sed -e "s/{PARENT}/${NUC}/g" ${CONF_FILE_TEMPLATE} > ${TMP_DIR}/config.yaml
    sed -i -e "s#{DEST}#${TMP_DIR}#g" ${TMP_DIR}/config.yaml
    sed -i -e "s#{TRANSITION}#${TRANSITION}#g" ${TMP_DIR}/config.yaml

    ./spades/bin/compute_spectra_psfs.py ${TMP_DIR}/config.yaml --energy_unit MeV --distance_unit bohr_radius --qvalues_file /home/stefan/CIFRA/2nubbspectra/data/mass_difference/deltaM_AME2020_ENSDF2025_best2BetaMinus2024.yaml > log.txt 2>&1
    #./spades/bin/compute_spectra_psfs.py ${TMP_DIR}/config.yaml --energy_unit MeV --distance_unit bohr_radius > log.txt 2>&1 # KI q-values by default
    mv log.txt ${TMP_DIR}/.
    echo "Done!"
done

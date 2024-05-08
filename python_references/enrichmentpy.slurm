#!/bin/bash

##SBATCH --time=04:00:00
#SBATCH --job-name=enrichment
#SBATCH -p 1024x28
##SBATCH -p 256x44
##SBATCH -p Instruction
#SBATCH -o enrich.out-%j
#SBATCH -e enrich.err-%j
#SBATCH --mem=510GB
#SBATCH --cpus-per-task=28
#SBATCH --ntasks=1
##SBATCH -n 1
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alohith@ucsc.edu
#SBATCH --workdir=/hb/home/alohith/KStest/

#module load miniconda3
source activate interactive
#conda list -n alohith-base
python --version
#which python
echo ""
#conda info -s
#export PYTHONNOUSERSITE=True

python enrichment.py

#find "$WATCH_DIR" -maxdepth 1 -mindepth 1 -type d ! -name '.*' -print0| while IFS= read -r -d '' dataDir;do


#for dataDir in $(ls $WATCH_DIR);
#do
    (
        echo $dataDir;
        #srun python ~/bin/KSTestAnalysis-main/ksCLI.py -k /hb/home/alohith/KStest/reducedKey.csv -d $dataDir -kt AL_CONSOLIDATED IDname -o $dataDir/data_out -n "/hb/home/alohith/KStest/featRed_horiz_names.txt" -p;
        #srun python ~/bin/KSTestAnalysis-main/ksCompTools/ksSimMat_preprocess.py
        
        #python ~/bin/KSTestAnalysis-main/ksCompTools/ksSimMat_preprocess.py -d FeatureReducedHD/horizConcat/ -s 10 -k /hb/home/alohith/KStest/reducedKey.csv -kt AL_CONSOLIDATED IDname -n /hb/home/alohith/KStest/featRed_horiz_names.txt

    ) 
#done


#srun python ~/bin/APCTool/APC.py -euc -x csv -mi 50000 -co 10000 -l 0.91 -o APCpy_UMAP/PMA+noPMA_horiztrain_full_Euc  -f KStest_py_20240302_FeatuReductions/data/TargetMol_1-27_1uM+10uM_PMA+noPMA_HD_horizconcat-dropna_UMAPreduced.csv -np

#python ~/bin/APCTool/APC.py -x csv -mi 50000 -co 10000 -l 0.95 -o APCpy_UMAP/PMA+noPMA_horiztrain_full_Pearson_lam95 -f KStest_py_20240302_FeatuReductions/data/TargetMol_1-27_1uM+10uM_PMA+noPMA_HD_horizconcat-dropna_UMAPreduced.csv -np

#python ~/bin/APCTool/APC.py -euc  -x csv -mi 50000 -co 10000 -l 0.95 -o APCpy_UMAP/noPMA_longtrain_full_Euc_lam95 -f KStest_py_20240302_FeatuReductions/data/TargetMol_1uM+10uM_noPMA_HD_longconcat_UMAPreduced.csv -np

#python ~/bin/APCTool/APC.py  -x csv -mi 50000 -co 10000 -l 0.95 -o APCpy_UMAP/noPMA_longtrain_ful_Pearson_lam95 -f KStest_py_20240302_FeatuReductions/data/TargetMol_1uM+10uM_noPMA_HD_longconcat_UMAPreduced.csv -np



#srun python ~/bin/KSTestAnalysis-main/ksCLI.py -k /hb/home/alohith/KStest/reducedKey.csv -d /hb/home/alohith/KStest/KStest_py_20240302_FeatuReductions/data -kt AL_CONSOLIDATED IDname -in AL_CONSOLIDATED_KSinclude.lst -o /hb/home/alohith/KStest/KStest_py_20240302_FeatReductions/data_out -n /hb/home/alohith/KStest/HB_FeatRed_fullDatanames_minSet.txt -rp -s 10 



#python ~/bin/KSTestAnalysis-main/ksCLI.py -k /hb/home/alohith/KStest/reducedKey.csv -d /hb/home/alohith/KStest/KStest_py_20240206_FeatReductions/data1 -kt AL_CONSOLIDATED IDname -in AL_CONSOLIDATED_KSinclude.lst -o /hb/home/alohith/KStest/KStest_py_20240206_FeatReductions/data_out_5Mar_ -n /hb/home/alohith/KStest/KStest_py_20240206_FeatReductions/data1_outnames.txt -rp -s 10 

#cd /hb/home/alohith/RotationProjects/LokeyLab-CPanalysis/randomCompGen/
#python /hb/home/alohith/RotationProjects/LokeyLab-CPanalysis/randomCompGen/slurmImport.py
#python slurmError_importModules.py
#python slurmImport.py
#python rand2Ref_20210426.py


#srun python distClassification_10partHoldOut.py KDEunitNormal_AllRefExp_20191018_commonFeatOrder_dist.csv unitNormalNull
#srun python distClassification_10partHoldOut.py KDEnormalOnFeatures_AllRefExp_20191018_commonFeatOrder_dist.csv featureNormalNull
#srun python distClassification_10partHoldOut.py KDEpostFlatten_AllRefExp_20191018_commonFeatOrder_dist.csv flattenKDENull

#srun python distClassification_10partHoldOut.py KDErandom15KSample_1_AllRefExp_20191018_commonFeatOrder_dist.csv random15KSample_1Null
#srun python distClassification_10partHoldOut.py KDErandom15KSample_2_AllRefExp_20191018_commonFeatOrder_dist.csv random15KSample_2Null
#srun python distClassification_10partHoldOut.py KDErandom15KSample_3_AllRefExp_20191018_commonFeatOrder_dist.csv random15KSample_3Null

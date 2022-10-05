# setup 
mkdir tmp
cd tmp
conda activate clim2clim2

# download the geopotential data
wget https://portal.nersc.gov/archive/home/projects/cascade/www/C20C/MOHC/HadGEM3-A-N216/All-Hist/est1/v1-0/day/atmos/zg/r1i1p2/zg_Aday_HadGEM3-A-N216_All-Hist_est1_v1-0_r1i1p2.tar

# create download script for other variables
python ../repos/climateTranslation/climatetranslation/get_data/download/create_c20c_download_script.py HadGEM3 --variable all --experiment all_hist --start 2 --stop 2 --outputfile dl_test_set.sh

bash dl_test_set.sh
rm dl_test_set.sh

mkdir files
mv *.nc files/

# compile other variables into zarr store
python ../repos/climateTranslation/climatetranslation/get_data/netcdfs_to_zarr.py --files $(ls files/*.nc) --zarr_store all_hist_testset_zarr

# unzip geopotential data
tar -xvf zg_Aday_HadGEM3-A-N216_All-Hist_est1_v1-0_r1i1p2.tar
mkdir zg
mv *.nc zg/

# append geopotential data to zarr store
python ../repos/climateTranslation/climatetranslation/get_data/scripts/append_zg.py --files $(ls zg/*.nc) --zarr_store all_hist_testset_zarr

# cleanup and move data
mv all_hist_testset_zarr_z5090_appended /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/all_hist_zarr
rm -rf all_hist_testset_zarr
rm -rf files
rm -rf zg

# get ready for quantile mapping
cd /home/s1205782/geos-fulton/repos/climateTranslation/climatetranslation/qm

# NEED TO EDIT THIS FILE TO CHANGE INPUT DATA
cp scripts/config.yaml /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/qm_config.yaml
#vim /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/qm_config.yaml

# quantile map
python translate.py --config /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/qm_config.yaml --x2x a2b --output_zarr /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/all_hist_qm_to_era_monsoon_zarr

# get ready for unit translation
cd /home/s1205782/geos-fulton/repos/climateTranslation/climatetranslation/unit

# NEED TO EDIT THIS FILE TO CHANGE INPUT DATA
cp /home/s1205782/geos-fulton/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/config.yaml /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/unit_config.yaml
#vim /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/unit_config.yaml

# UNIT translate
python translate.py --config /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/unit_config.yaml --output_zarr /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/v8.2_monsoon_to_era5_40k --checkpoint /home/s1205782/geos-fulton/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/checkpoints/gen_00040000.pt  --x2x b2a --seed 202109061100


# get ready for qm of unit translation
cd /home/s1205782/geos-fulton/repos/climateTranslation/climatetranslation/qm

# NEED TO EDIT THIS FILE TO CHANGE INPUT DATA
cp scripts/config-unit.yaml /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/unitqm_config.yaml
#vim /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/unitqm_config.yaml


# quantile map the unit transformed data
python translate.py --config /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/unitqm_config.yaml --x2x a2b --output_zarr /home/s1205782/geos-fulton/datadrive/hadgem3/test_set/v8.2_monsoon_unit40k_and_qm_to_era5






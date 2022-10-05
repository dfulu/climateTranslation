model=$1
echo "Creating data download script"
python ~/repos/climateTranslation/get_data/get_model_data.py $model --variable all --experiment nat_hist --start 1 --stop 10 --outputfile download.sh

mkdir netcdfs
cd netcdfs

echo "Downloading data"
bash ../download.sh

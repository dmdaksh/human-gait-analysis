# # create data directory if it doesn't exist
# mkdir -p data

# # download data zip file
# wget -O data/input_data_zip_file https://springernature.figshare.com/ndownloader/files/21998373

# # extract data zip file
# unzip data/input_data_zip_file -d data/input_data

# downloading wget and unzip
sudo apt install wget unzip

# download data zip file
wget -O input_data_zip_file https://springernature.figshare.com/ndownloader/files/21998373

# extract data zip file
unzip input_data_zip_file

# rename unzipped file
mv input\ data_SD input_data

# download cloud-tpu-client
# pip3 install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

conda env create -f rgbd_segmentation.yaml

conda activate rgbd_segmentation

sudo apt-get install python-pip
pip install gdown
cd trained_models/
gdown https://drive.google.com/uc?id=1C5-kJv4w3foicEudP3DAjdIXVuzUK7O8
tar -xvzf nyuv2_r34_NBt1D.tar.gz
gdown https://drive.google.com/uc?id=1tviMAEOr-6lJphpluGvdhBDA_FetIR14
tar -xvzf sunrgbd_r34_NBt1D.tar.gz

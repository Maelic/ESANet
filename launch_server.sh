
if [ "$1" = "sunrgbd" ]; then
	sudo python3.7 inference_server.py --dataset sunrgbd --ckpt_path ./trained_models/sunrgbd/r34_NBt1D.pth --depth_scale 1 --raw_depth 
elif [ "$1" = "nyuv2" ]; then
	sudo python3.7 inference_server.py --dataset nyuv2 --ckpt_path ./trained_models/nyuv2/r34_NBt1D.pth --depth_scale 0.1 --raw_depth
else 
	echo "You need to provide the dataset you want to use: sunrgbd or nyuv2"
	exit 1
fi

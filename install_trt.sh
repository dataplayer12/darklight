os="ubuntu2004"
tag="cuda11.6-trt8.4.3.1-ga-20220813"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/*.pub

sudo apt-get update
sudo apt-get install tensorrt

python3 -m pip install numpy
sudo apt-get install python3-libnvinfer-dev

echo "Verifying installation..."
dpkg -l | grep TensorRT

python3 -m pip install --upgrade setuptools pip

python3 -m pip install nvidia-pyindex

python3 -m pip install --upgrade nvidia-tensorrt

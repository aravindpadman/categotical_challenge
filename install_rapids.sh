cp /kaggle/input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz
cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

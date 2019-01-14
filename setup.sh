# https://www.python.org/downloads/mac-osx/

virtualenv -p python3.6 venv

source ./venv/bin/activate

pip install keras==1.2.1
pip install theano==1.0.3 # 0.9.0 for python 3.5
# tensorflow is not working anyway
# pip install tensorflow==0.11 # 0.11
pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py3-none-any.whl
# pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc0-py3-none-any.whl
# pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc1-py3-none-any.whl
# pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc2-py3-none-any.whl
# pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl

pip install numpy==1.15.0

# python train_linkpred.py -d twitter --bases 30 --hidden 16 --l2norm 5e-4 --testing

deactivate

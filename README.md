conda install -c menpo cyvlfeat

pip install -r requirements.txt

For errors in windows:
conda install -c msys2 m2w64-gcc=5.3.0 
conda install libpython

python preprocess.py
python fullEval.py


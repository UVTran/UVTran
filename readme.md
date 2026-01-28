Step 1. Install requirements:
```
pip install -r requirements.txt
```
Step 2. Compile the C++ extension modules:
```
cd /datagen
python g1.py
```

Step 3. Train the model:
```
python trainpre.py --batch_size 128 --lr 1e-4


python trainrefine.py  --batch_size 128 --lr 1e-4
```

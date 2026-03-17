# MAPAGFormer: Multi-Granular Anatomy-Prior Enhanced AGFormer for Accurate 3D Human Pose Estimation

A 3D human pose estimation framework that integrates multi-granular anatomical priors.

---

## Environment

The project is developed under the following environment:

- Python 3.8.10
- PyTorch 2.0.0
- CUDA 12.2

For installation of the project dependencies, please run:

```bash
pip install -r requirements.txt
```
---

## Dataset
## Human3.6M：data/motion3d
Download the fine-tuned Stacked Hourglass detections of preprocessed H3.6M data [here](https://onedrive.live.com/?id=A5438CD242871DF0%21206&resid=A5438CD242871DF0%21206&e=vobkjZ&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdkFkaDBMU2pFT2xnVTdCdVVaY3lhZnU4a3pjP2U9dm9ia2pa&cid=a5438cd242871df0&v=validatepermission) and unzip it to 'data/motion3d'.

Slice the motion clips by running the following python code in directory:data/preprocess

For MAPAGFormer-Base and MAPAGFormer-Large:
```bash
python h36m.py  --n-frames 243
```
For MAPAGFormer-Small:
```bash
 python h36m.py --n-frames 81
```                  
For MAPAGFormer-XSmall:
```bash
python h36m.py --n-frames 27
```    
  ## MPI-INF-3DHP: motion3d
  Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp)  for dataset setup. After preprocessing, the generated .npz files ( and ) should be located at directory.
```bash
data_train_3dhp.npzdata_test_3dhp.npzdata/motion3d 
```

 ---
 
# Training
After dataset preparation, you can train the model as follows:
  ## Human3.6M:
  You can train Human3.6M with the following command:
```bash
python train.py --config <PATH-TO-CONFIG>
```  
  ## MPI-INF-3DHP:
  You can train MPI-INF-3DHP with the following command:
```bash
python train_3dhp.py --config <PATH-TO-CONFIG>
```  
# Evaluation
You can evaluate Human3.6M models by:
```bash
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
For example if MAPAGFormer-s of H.36M is downloaded and put in directory, then we can run:
```bash
python train.py --eval-only --checkpoint checkpoint --checkpoint-file MAPAGFormer-s-h36m.pth.tr --config configs/h36m/MAPAGFormer-small.yaml
```
Similarly, MPI-INF-3DHP can be evaluated as follows:
```bash
python train_3dhp.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```

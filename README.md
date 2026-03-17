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
Slice the motion clips by running the following python code in directory:data/preprocess
For MAPAGFormer-Base and MotionAGFormer-Large:
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
  ## MPI-INF-3DHP: motion3d.
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

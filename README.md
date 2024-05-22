# VaF-LangSplat: Voxel-Aware Fusion Language Gaussian Splatting

## Datasets

In the experiments section of our paper, we primarily utilized two datasets: the 3D-OVS dataset and the LERF dataset.

The 3D-OVS dataset is accessible for download via the following link: [Download 3D-OVS Dataset](https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing) .

The LERF dataset expanded by LangSplat is accessible for download via the following link: [Download Expanded LERF Dataset and COLMAP Data](https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing).



## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models.



### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)



### Software Requirements

- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used VS Code)
- CUDA SDK 11 for PyTorch extensions (we used 11.8)
- C++ Compiler and CUDA SDK must be compatible



### Setup

#### Environment Setup

Our default, provided install method is based on Conda package and environment management:

```
conda env create --file environment.yml
conda activate vaf
```



### QuickStart

Download the pretrained model to `output/`, then simply use

```
python render.py -m output/$CASENAME
# python render.py -m output/lawn
```



## Processing your own Scenes



### Before getting started

Firstly, you need to acquire the following dataset format:

```
<dataset_name>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```



### Environment setup.

Please download the checkpoints of SAM from [here](https://github.com/facebookresearch/segment-anything) to `ckpts/`.



### Pipeline

- **Step 1: Generate Language Feature of the Scenes.** Put the image data into the "input" directory under the `<dataset_name>/`, then run the following code.

  ```
  python preprocess.py --dataset_path $dataset_path 
  ```

  

- **Step 2: Train the Autoencoder and get the lower-dims Feature.**

  ```
  # train the autoencoder
  cd autoencoder
  python train.py --dataset_name $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --output ae_ckpt
  # get the 3-dims language feature of the scene
  python test.py --dataset_name $dataset_path --output
  ```

  

  Our model expect the following dataset structure in the source path location:

  ```
  <dataset_name>
  |---images
  |   |---<image 0>
  |   |---<image 1>
  |   |---...
  |---language_feature
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---language_feature_dim3
  |   |---00_f.npy
  |   |---00_s.npy
  |   |---...
  |---sparse
      |---0
          |---cameras.bin
          |---images.bin
          |---points3D.bin
  ```

​	Note that the full path of `<dataset_name>` should be `3d-ovs/$CASENAME` or `Lerf/$CASENAME`, not simply `$CASENAME`



- **Step 3: Train the VaF-LangSplat.**

  If you want to train on the scene of 3D-OVS, modify `scenes=("$CASENAME")` and then:

  ```
  bash ovs_train.sh
  ```

  If you want to train on the scene of LERF, modify `scenes=("$CASENAME")` and then:

  ```
  bash lerf_train.sh
  ```

- **Step 4: Render the VaF-LangSplat.**

  ```
  python render.py -m output/$CASENAME
  ```




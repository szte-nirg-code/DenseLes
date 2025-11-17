# DenseLes
# 🧠 DenseLes: Multiple Sclerosis Lesion Segmentation

A deep learning project using TensorFlow and Keras for segmenting Multiple Sclerosis (MS) lesions from 3D MRI (NIfTI) data.

## 🚀 Core Methodology

This project tackles the 3D segmentation challenge by employing a **2D slice-by-slice approach**. Instead of a full 3D model, it trains three separate 2D U-Net-like models, one for each anatomical plane:

* **Axial** (Direction 0)
* **Coronal** (Direction 1)
* **Sagittal** (Direction 2)

The model architecture, defined in `model.py`, is a U-Net-style encoder-decoder that uses **DenseNet-inspired blocks**. These blocks promote feature reuse and help capture the complex patterns of MS lesions.

The training process uses the **Tversky loss** function, which is highly effective for imbalanced datasets—a common problem in medical segmentation where lesions are small compared to the surrounding brain tissue.

## ▶️ How to Train

You can run the main training script from your terminal using `train.py`. The script is configured using command-line arguments.

**Example Training Command:**

```bash
python train.py \
    --data_path /path/to/your/data_root_folder \
    --model_name_prefix "ms_lesion_model" \
    --img_size "144,256,160" \
    --batch_size 10 \
    --epochs 500 \
    --patience 20 \
    --split_num 1

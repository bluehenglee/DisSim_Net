# PAplusNet

Welcome to the official repository for the paper: "Beyond Point Annotation: A Weakly Supervised Network Guided by Multi-Level Labels Generated from Four-Point Annotation for Thyroid Nodule Segmentation in Ultrasound Images."
![image](https://github.com/user-attachments/assets/4fec8f5f-8fc8-448e-8b87-0eb978b61806)


## Environment Requirements

To run this project, ensure you have the following software installed:
- Python 3.7 or higher
- PyTorch 1.8 or higher

You can create a virtual environment and install the required packages with:

```bash
# Create a new conda environment
conda create -n your_env_name python=3.7

# Activate the conda environment
conda activate your_env_name

# Install PyTorch and torchvision
pip install torch==1.8.0 torchvision==0.9.0  # Adjust according to your CUDA version
```


## Dataset Preparation
```
The dataset structure required for this network should be organized as follows:
├── main
├── models
├── utils
├── data/          # the dataset folder
    ├── train      # train dataset
        ├── imgs    
        ├── gt      
        └── labels  # point labels .json
    ├── val        # val and test dataset
        ├── imgs
        ├── gt
        └── labels
```

## Dataset Preprocessing
Before training, you need to preprocess the dataset using the point labels to generate the dissimilarity prior and multi-level labels. The generated prior and labels will be saved at the same folder level as the 'imgs'.

Run the following command to preprocess the dataset:
```bash
python utils/data_pre.py --data_name "your prepocess dataset"
```
Replace "your_preprocess_dataset" with the name of your dataset.

## Training the Network
To train the network, use the following command:
```bash
python main/train.py --arch "model name " --dataset "dataset name" --epochs "training epochs"
```

## Inference
To perform inference with the trained model, use the following command:
```bash
python main/inference.py --arch "model name " --train "the model trained on which dataset" --dataset "inference dataset name" --epochs "training epochs"
```
Replace the placeholders:
"model_name": The name of the model architecture.
"trained_model_name": The name of the model that was trained on the dataset.
"inference_dataset_name": The name of the dataset you want to run inference on.
"training_epochs": The number of epochs for which the model was trained.

## License
This project is licensed under the MIT License.

## Citation
If you use this code in your research, please cite our paper:
```
@article{chi2024beyond,
  title={Beyond Point Annotation: A Weakly Supervised Network Guided by Multi-Level Labels Generated from Four-Point Annotation for Thyroid Nodule Segmentation in Ultrasound Image},
  author={Chi, Jianning and Li, Zelan and Wu, Huixuan and Zhang, Wenjun and Huang, Ying},
  journal={arXiv preprint arXiv:2410.19332},
  year={2024}
}
```

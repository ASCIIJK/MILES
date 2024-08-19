# MILES
A new SOTA methods for pre-trained model-based CIL! This is the open source repository for MILES, i.e., MetrIc Learning with Expanable Subspace.
# Preliminary
Before reproducing our experiments, you should prepare two things:
1. Data.
2. Pre-traind backbone.
## Data
Six datasets are included in our experiments, i.e., CIFAR-100, ImageNet-A, Omnibenchmark, CUB-200, FOOD-101 and CARS-196.
1. **CIFAR-100**: it will be automatically downloaded by the code.
2. **ImageNet-A**: Onedrive: [link](https://entuedu-my.sharepoint.com/personal/n2207876b_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL%2Fina%2Ezip&parent=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL&ga=1)
3. **Omnibenchmark**: Onedrive: [link](https://entuedu-my.sharepoint.com/personal/n2207876b_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL%2Fomnibenchmark%2Ezip&parent=%2Fpersonal%2Fn2207876b%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2FRevisitingCIL&ga=1)
4. **CUB-200**: AWS: [link](https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz)
5. **FOOD-101**: AWS: [link](https://s3.amazonaws.com/fast-ai-imageclas/food-101.tgz)
6. **CARS-196**: AWS: [link](https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz)
All links are from open sources. You can download these datasets and put them in the 'data' filefolder. The formats follow ImageFolder.
## Pre-traind backbone
We adopt the ViT-B/16-IN21K as the pre-trained model. You can use the timm library to obtain the pre-trained weight. The weight file please put in './pre_trained_backbone/' folder, i.e., './pre_trained_backbone/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'.


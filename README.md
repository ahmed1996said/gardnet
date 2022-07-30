[![arXiv](https://img.shields.io/badge/arXiv-2205.12902-<COLOR>.svg)](https://arxiv.org/abs/2205.12902)

_Work in progress. Code will be released soon._

# GARDNet: Robust Multi-View Network for Glaucoma Classification in Color Fundus Images
**Authors:** Ahmed Al Mahrooqi, Dmitrii Medvedev, Rand Muhtaseb, Mohammad Yaqub

**Instituion:** Mohamed bin Zayed University of Artificial Intelligence

## :page_facing_up: Abstract
Glaucoma is one of the most severe eye diseases, characterized by rapid progression and leading to irreversible blindness. It is often the case that diagnostics is carried out when one’s sight has already significantly degraded due to the lack of noticeable symptoms at early stage of the disease. Regular glaucoma screenings of the population shall improve early-stage detection, however the desirable frequency of etymological checkups is often not feasible due to the excessive load imposed by manual diagnostics on limited number of specialists. Considering the basic methodology to detect glaucoma is to analyze fundus images for the optic-disc-to-optic-cup ratio, Machine Learning algorithms can offer sophisticated methods for image processing and classification. In our work, we propose an advanced image pre-processing technique combined with a multi-view network of deep classification models to categorize glaucoma. Our Glaucoma Automated Retinal Detection Network (GARDNet) has been successfully tested on Rotterdam EyePACS AIROGS dataset with an AUC of 0.92, and then additionally fine-tuned and tested on RIM-ONE DL dataset with an AUC of 0.9308 outperforming the state- of-the-art of 0.9272.

This work has been accepted at MICCAI 2022 workshop [OMIA9](https://sites.google.com/view/omia9).
### :key: Keywords
Glaucoma Classification, Color Fundus Images. Computer Aided Diagnosis


## :open_file_folder: File Structure 

    /
    ├── configs                 	                # contains experiment configuration .yaml files required to train
    ├── notebooks                 	                # contains .ipynb notebooks for preprocessing and testing
            ├──GradCAM.ipynb                        # GradCAM Visualization script
            ├──Testing Ensemble-AIROGS.ipynb        # Testing script for the ensemble model on AIROGS
            ├──Testing Ensemble-RIM-ONE DL.ipynb    # Testing script for the ensemble model on RIM-ONE DL
            ├──Testing RIM ONE DL.ipynb             # Finetune and testing script on RIM-ONE DL
            ├── bbox_crop.ipynb                     # Preprocessing script for cropping AIROGS using bounding box coords
            └── central_crop.ipynb                  # Preprocessing scrirpt for cropping AIROGS using central crrop
    ├── README.md
    ├── airogs_dataset.py                           # contains dataset class for AIROGS
    ├── early_stopping.py                           # contains script for early stopping
    ├── requirements.txt                            # contains packages and libraries needed to run our code
    ├── run.py                  	                # contains training script 
    └── run_fold.py                                 # contains training script with cross validation
    
## :package: Requirements
You can install all requirements using `pip` by running this command:

``` pip install -r requirements.txt```

Generally speaking, our code uses the following core packages: 
- PyTorch 1.9.0
- [wandb](https://wandb.ai): you need to create an account for logging purposes

## :arrow_forward:	 Training
_Add instructions to run the code_

## :question: Questions?
For all code related questions, please create a GitHub Issue above and our team will respond to you as soon as possible.


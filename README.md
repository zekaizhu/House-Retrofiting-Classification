# Project Build Change

- **Team: Grand Teton**    
- **Team Members: Zach Zhu, Jue Wang, Bhavya Kaushik, Jing Ren**

Directory Structure:

```
├── other code/                           <- other scripts
│   
├── deliverables/                         <- final presentation slide and project report
│   
├── visualizations/                       <- images of visualizations generated through scripts
│   
├── weekly update/                        <- directory for weekly_update.md file
│
├── model/                                <- model related scripts
│   ├── XGBoost_generated.ipynb           <- XGBoost model on generated images
│   ├── XGBoost_real.ipynb                <- XGBoost model on real images
│   ├── LogReg_generated.ipynb            <- logistic regression model on 
│
├── feature/                              <- generate features for modelling
│   ├── feature_Extractor_generated.py    <- extract feautres from generated images of houses 
│   ├── Feature_Extractor_real.py         <- extract feautres from real images of houses 
│   ├── explore_images_generated.ipynb    <- initial EDA of generated images
│   ├── feature_testing_generated.ipynb   <- initial feature testing on generated images
│   ├── feature_testing_real.ipynb        <- initial feature testing on real images
│   ├── link_images_csv.ipynb             <- map generated images with the given data csv file
│
├── data/                                 <- csv files 
│   ├── features_real.csv                 <- features extracted through classical computer vision 
│   ├── labels.csv                        <- manually generated labels of the real images 
└── ...

```
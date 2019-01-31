# Chapter02: Transfer Learning 

Transfer learning is the process of transferring the knowledge gained in one task in a
specific domain to a related task in a similar domain. In the deep learning paradigm,
transfer learning generally refers to the reuse of a pre-trained model as the starting point
for another problem. The problems in computer vision and natural language processing
require a lot of data and computational resources, in order to train meaningful deep
learning models. Transfer learning has gained a lot of importance in the domains of vision
and text, since it alleviates the need for a large amount of training data and training time. In
this chapter, we will use transfer learning to solve a healthcare problem.


#### Goal 

- [x] To leverage transfer learning in solving real world problems
- [x] Look at the achitecture of Standard CNN architectures that can be used for Transfer Learning
- [x] Be comfortable on various facets of performing Transfer learning 

#### Dataset Link
[Data] (https://www.kaggle.com/c/classroom-diabetic-retinopathy-detection-competition/data)

#### Command to execute TransferLearning.py for Training and Holdout validation

```bash

python TransferLearning.py --path '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/Extra/assignment2_train_dataset/' --class_folders '["class0","class1","class2","class3","class4"]' --dim 224 --lr 1e-4 --batch_size 16 --epochs 20 --initial_layers_to_freeze 10 --model InceptionV3 --folds 5 --outdir '/home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/'

```

#### Command to execute TransferLearning_ffd.py for Training and Validation



```bash
python TransferLearning_ffd.py --path '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/Extra/assignment2_train_dataset/' --class_folders '["class0","class1","class2","class3","class4"]' --dim 224 --lr 1e-4 --batch_size 32 --epochs 50 --initial_layers_to_freeze 10 --model InceptionV3 --outdir '/home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/'

```

#### Command to execute TransferLearning_reg.py for Training 
```bash
python TransferLearning_reg.py --path '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/Extra/assignment2_train_dataset/' --class_folders '["class0","class1","class2","class3","class4"]' --dim 224 --lr 1e-4 --batch_size 32 --epochs 5 --initial_layers_to_freeze 10 --model InceptionV3 --folds 5 --outdir '/home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/Regression/'

```
#### Command to execute TransferLearning_reg.py for Validation 

```bash

python TransferLearning_reg.py --path '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/Extra/assignment2_train_dataset/' --class_folders '["class0","class1","class2","class3","class4"]' --dim 224 --lr 1e-4 --batch_size 32 --model InceptionV3 --outdir '/home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/Regression/' --mode validation --model_save_dest --'/home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/Regression/model_dict.pkl' --folds 5

```

**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 







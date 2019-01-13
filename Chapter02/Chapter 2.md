Command to run training

python TransferLearning.py  --path <Data folders path>  --class_folders ["0","1","2","3","4"] --dim 112  --lr 1e-4 --batch_size 16 --epochs 100 --initial_layers_to_freeze 10 --model InceptionV3 --folds 5  --outdir <output directory>

Example:
python TransferLearning.py  --path '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/New/'  --class_folders ["0","1","2","3","4"] --dim 112  --lr 1e-4 --batch_size 16 --epochs 100 --initial_layers_to_freeze 10 --model InceptionV3 --folds 5  --outdir '/home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/'

[x] For limited resources for a quick run - run for smaller number of epochs say 1, and smaller number of folds say 2 

[x] Download Data from the site mentioned in the Chapter 2 (unable to check it since the Chapter is locked)


Command to run Inference on the trained model 

python TransferLearningInference.py --path <path to test data>  --dim 112 --model_save_dest /home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/temp/dict_model.pkl --n_class 5 --outdir '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/temp/'

Example:
python TransferLearningInference.py --path '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/test/'  --dim 112 --model_save_dest /home/santanu/ML_DS_Catalog-/Transfer_Learning_DR/temp/dict_model.pkl --n_class 5 --outdir '/media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/book AI/Diabetic Retinopathy/temp/'
 







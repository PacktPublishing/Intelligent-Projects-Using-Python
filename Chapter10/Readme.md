[x] Captcha Solver 

- Command to train:    
python captcha_solver.py train --dest_train <train images directory> --dest_val <val images directory> --outdir <output directory> --batch_size < batch size> --lr <learning rate> --epochs <epochs> --n_classes < no of characters to classify> --shuffle <shuffle data in each epoch while training> --dim <captcha dimension>

- Example 
python captcha_solver.py train --dest_train '/home/santanu/Downloads/Captcha Generation/captcha_train/' --dest_val '/home/santanu/Downloads/Captcha Generation/captcha_val/' --outdir '/home/santanu/ML_DS_Catalog-/captcha/model/' --batch_size 16 --lr 1e-3 --epochs 20 --n_classes 36 --shuffle True --dim '(40,100,1)'


- Command to run inference/evaluate

python captcha_solver.py evaluate  --model_path  <complete model path> --eval_dest < Directory of images to run inference on> --outdir <output directory> --fetch_target < True/False> 

- Example 

python captcha_solver.py evaluate  --model_path  /home/santanu/ML_DS_Catalog-/captcha/model/captcha_breaker.h5 --eval_dest '/home/santanu/Downloads/Captcha Generation/captcha_test/' --outdir /home/santanu/ML_DS_Catalog-/captcha/ --fetch_target True 


[x]GAN based Captcha Generator 

- Command to train
python captcha_gan.py train --dest_train <directory/file for input images> --outdir <output directory> --dir_flag <whether input is file/directory> --batch_size <batch size> --gen_input_dim <noise dimension> --gen_beta1 <beta_1 of GAN generator> --gen_lr <learning rate of Generator> --dis_input_dim <input dimension of discriminator> --dis_lr <discriminator learning rate> --dis_beta1 <beta_1 of discriminator> --alpha <leak factor of Leaky RELU activation> --epochs <epochs> --smooth_coef <smoothing coeffecient>

- Examle 
python captcha_gan.py train --dest_train '/home/santanu/Downloads/train_32x32.mat' --outdir '/home/santanu/ML_DS_Catalog-/captcha/SVHN/' --dir_flag False --batch_size 100 --gen_input_dim 100 --gen_beta1 0.5 --gen_lr 0.0001 --dis_input_dim '(32,32,3)' --dis_lr 0.001 --dis_beta1 0.5 --alpha 0.2 --epochs 100 --smooth_coef 0.1


- Command to generate Captchas using the Trained network

python captcha_gan.py generate-captcha --gen_input_dim <noise inputdimension> --num_images <number of Captchas to generate> --model_dir <trained model directory> --outdir <output directory> --alpha <Leaky ReLU leak factor>

-Example 
python captcha_gan.py generate-captcha --gen_input_dim 100 --num_images 200 --model_dir '/home/santanu/ML_DS_Catalog-/captcha/' --outdir '/home/santanu/ML_DS_Catalog-/captcha/captcha_for_use/' --alpha 0.2


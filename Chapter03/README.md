# Chapter03: Neural Machine Translation

Machine translation, in simple terms, refers to the translation of text from one language to
another using a computer.With the recent advances in recurrent neural network (RNN) architectures, such
as LSTMs, GRU, and so on, machine translation not only provides an improved quality of
translation, but also the complexity of such systems is far less than those of traditional
systems.


#### Goal 
- [x] To build Machine Translation Model using LSTMs 
- [x] Learn the architectures of the Model and the technical knowhow

#### Dataset Link
[Data] (http://www.manythings.org/anki/)
- [x] Dataset name : fra-eng/fra.txt



#### Command to execute MachineTranslation.py for training and holdout valdiation

```bash

python MachineTranslation.py --path '/home/santanu/ML_DS_Catalog-/Machine Translation/fra-eng/fra.txt' --epochs 20 --batch_size 32 --latent_dim 128 --num_samples 40000 --outdir '/home/santanu/ML_DS_Catalog-/Machine Translation/' --verbose 1 --mode train

```

#### Command to execute MachineTranslation_word2vec.py for Training and Validation



```bash

python MachineTranslation_word2vec.py --path '/home/santanu/ML_DS_Catalog-/Machine Translation/fra-eng/fra.txt' --epochs 20 --batch_size 32 --latent_dim 128 --num_samples 40000 --outdir '/home/santanu/ML_DS_Catalog-/Machine Translation/' --verbose 1 --mode train --embedding_dim 128

```


**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 







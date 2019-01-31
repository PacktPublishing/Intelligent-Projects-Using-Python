# Chapter04: Style Transfer in the Fashion Industry Using using GANs

The concept of style transfer refers to the process of rendering the style of a product into
another product. Imagine that your fashion-crazy friend bought a blue-printed bag and
wanted to get a pair of shoes of a similar print to go with it. Up until 2016, this might not
have been possible, unless they were friends with a fashion designer who would first have
to design a shoe before it was approved for production. With the recent progress in
generative adversarial networks; however, this kind of design process can be carried out
easily.

#### Goal 
- [x] To build a System that con convert handbag edges to Handbag Images 
- [x] Learn the architectures of the Model and the technical knowhow

#### Dataset Link
[Data] (https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz)



#### Command to execute image_split.py for preprocessing image 

```bash

python image_split.py --path /media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/edges2handbags/ --_dir_ train

```

#### Command to execute cycledGAN_edges_to_bags.py to train the GAN and generate images 


```bash

python cycledGAN_edges_to_bags.py process_main  --dataset_dir /media/santanu/9eb9b6dc-b380-486e-b4fd-c424a325b976/edges2handbags/ epochs 100

```

**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 







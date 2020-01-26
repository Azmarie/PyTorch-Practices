###  Fine-Tuning a Pre-Trained Network

The provided code builds upon ResNet20, a state of the art deep network for image classification.
The model has been trained for CIFAR-100 image classification with 100 output classes.
The ResNet20 model has been adapted to solve a (simpler) different task, classifying an image as
one of 10 classes on CIFAR10 dataset.

The code imagenet finetune.py does the following:
* Constructs a deep network. This network starts with ResNet20 up to its average pooling
layer. Then, a new linear classifier .
* Initializes the weights of the ResNet20 portion with the parameters from training on CIFAR10.
* Performs training on only the new layers using CIFAR-10 dataset – all other weights are
fixed to their values learned on ImageNet.

### Environment Setup

To set up the virtual environment, install Anaconda and run the following command
`conda env create -f cmpt726-pytorch-python36.yml`

To activate the virtual environment, run the following command
`source activate cmpt726-pytorch-python36`

### Task

Write a Python function to be used at the end of training that generates HTML output showing each test image and its classification scores. You could produce an HTML table output for example. (You can convert the HTML output to PDF or use screen shots.)


### What I did

- Implemented a python function called `extractCifar10.py` to retrieve and restore images as `.png` file in `data/test`, named `image0.png` and etc. 
- Implemented a python function called `generate_html` showing each test images and the classification scores for 10 classes.
- Ran the provided model training code and save the parameters into `cifar_net.pth` (PATH) in the same dir level as `.py` file.
- Created `testset` and `testloader` similar to `trainset` and `trainloader` with `batch_size = 32`.
- Implemented logic so that if `os.path.exist(PATH)` then just load the saved model, if not, train the model.
- Implemented code to get the test set accuracy.

### How to run my code

Note: Assume you already have the `data` folder ready from the starter code with the data baches inside.

- `cd CIFAR/`
- `python data/extractCifar10.py` to restore CIFAR 10 image dataset
- (optional) if you already have a trained model, it's time to put it under CIFAR/
- `python cifar_finetune.py`, it will check if you have a saved model, if yes then just generate HTML, if no then it will start training, after training finishes, `generate_html` will do its magic.
- Vola, there you have a beautiful table in `result.html` as the results shown before. (Note that you need to have the images in the `data/test` folder to see them. )

### Results

- The first page of my HTML ![HTML](cifar_html.png)
- I got 67% for testset accuracy with `batch_size = 32`

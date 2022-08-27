# PytorchDeepCNN-Classifying-Images-with-Deep-Convolutional-Neural-Networks-w-CelebA
In this notebook,I implemented a CNN on complex CelebA dataset consisting of face images and trained the CNN for smile classification using smile attributes of the pictures.

Below steps were applied through model build-up :

- Download the CelebA Data Set

- Apply data augmentation and different transformations to face images using the torchvision.transforms module.

- Use dataloader to load train,valid and test sets.

- Build a Pytorch Deep CNN model with torch.nn module input with 3×64×64 (the images have three color channels by64*64 square shape).The input data goes through four convolutional layers to make 32, 64, 128, and 256 feature maps using filters with a kernel size of 3×3 and padding of 1 for same padding. The first three convolution layers are followed by max-pooling, P2×2. I included Two dropout layers for regularization:

- Use BCELoss as loss function for my binary classification problem with a single probabilistic output

- Plot loss  and accuracy curves for training and validation sets.

- Compare ground truth values for smile attribute of first 10 pictures with predicted classes belongs to CelebA dataset.

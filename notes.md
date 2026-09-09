## MNIST CNN Results

The toy convolution example showed that a manually defined horizontal edge filter produces strong responses at the top and bottom edges of a white square and near-zero responses in flat regions.

The CNN trained on MNIST reached 99.11% test accuracy after 5 epochs.

Training loss decreased from 0.1259 to 0.0259, showing that the model learned successfully. MNIST is a sanity check dataset, so high accuracy confirms that the CNN architecture, data loading, and training loop are working.

The key concept is that in the toy example we manually wrote a filter, while in the CNN the filters are learned automatically through backpropagation.

## CIFAR-10 Results

The CIFAR-10 VGG-style CNN trained successfully. Training loss and validation loss both decreased over 20 epochs, showing healthy learning. Validation accuracy rose from about 38% at the start to around 89–90% by the end.

The result is strong because CIFAR-10 is much harder than MNIST: it contains small 32×32 RGB images of real objects, with visually similar classes such as cats/dogs and cars/trucks.

There was some bumpiness in validation loss around the middle epochs, but the overall trend improved. After the learning-rate scheduler reduced the learning rate around epoch 12, validation accuracy improved sharply and stabilized near 90%.

The curves do not show serious overfitting because validation loss did not consistently rise while training loss decreased.
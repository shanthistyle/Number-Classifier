I would like to implement an algorithm that classifies an image of a number as the corresponding number. This would be useful in post offices when reading handwritten adresses.

APPROACH:

Let each pixel in an image represent a feature. All images are 28x28 pixels ~ 784 features. We will use K-Nearest Neighbors to classify each input image as a number 0...9. 

1) Plot the training features on a 794-dimension graph
2) For each input image (valFeatures), find k nearest neighbors and their corresponding labels (trainLabels). Return the majority classification of the labels as the classification of the input image. Check this classification with valLabels and compute error rate.
3) After achieving desired error rate on validation set (valFeatures), run classifier on testFeatures.  

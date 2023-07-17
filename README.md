# Neural Style Transfer with VGG19

This project showcases a Neural Style Transfer model built on top of the VGG19 architecture, trained to apply the style of a given image to another while preserving the content of the original image.

## Table of Contents

- [Background](#background)
- [Project Description](#project-description)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Work](#future-work)
- [Sources](#sources)

## Background

Neural Style Transfer is a technique in Deep Learning that blends two images, namely, a "content" image and a "style" image, to create a "generated" image with the content of the former and the style of the latter. This project employs a pre-trained VGG19 model to extract feature maps from different layers of the network, which are then used to create a stylized image.

## Project Description

The model is trained to optimize a loss function that takes into account both the content and the style of the images. It combines the content of one image with the style of another image to generate a new, unique image that could be thought of as a piece of art.

The project includes an interesting feature: 

- A flexible architecture that allows users to experiment with different content and style images to generate unique results. 

The architecture and hyperparameters are robust, yet they offer a lot of scope for tweaking and experimentation.

## Model Architecture

The architecture of the Neural Style Transfer model is based on the VGG19 network, a pre-trained deep convolutional neural network known for its strong feature extraction capabilities. The VGG19 architecture is used to extract the style and content features from the images. A basic VGG19 architecture can be seen below:

![image](https://github.com/DimensionDweller/Neural_Style_Transfer/assets/75709283/80a5db41-a5e9-463d-aef6-4abab0b2d576)



The main parts of the architecture are as follows:

1. **VGG19 Feature Extraction Model**: This is a pre-trained VGG19 model that is used to extract the features from the content and style images. The weights of this model are frozen during training.

2. **Style and Content Losses**: These are the two components of the loss function that the model optimizes. The style loss ensures that the style of the generated image matches the style of the style image, and the content loss ensures that the content of the generated image matches the content of the content image.

The architecture has been chosen to take advantage of the strong feature extraction capabilities of the VGG19 model, and the combination of style and content losses allows the generated image to have the desired style and content.

## Gram Matrix and Style Loss

The style of an image is defined by the correlations between the feature maps in the layers of the VGG19 model. These correlations are captured by the Gram matrix. 

For a given layer $\(l\)$, let $\(F_{li}\)$ be the activation of the $\(i\)-th$ filter at position $\(j\)$ in the layer. The Gram matrix $\(G_{li}\)$ for layer $\(l\)$ is then given by:

$$\[G_{li}^l = \sum_{j} F_{li}^l F_{lj}^l\]$$

This is essentially computing the dot product between the vectorized feature maps $\(i\)$ and $\(j\)$. This can be interpreted as a measure of the correlation between the feature maps. High values in the Gram matrix means that feature maps $\(i\)$ and $\(j\)$ tend to activate together, defining a style for the image.

The style loss for a single layer is defined as the mean squared error for the Gram matrices of the generated and style images. Given the Gram matrices $\(G\)$ for the generated image and $\(A\)$ for the style image, the style loss $\(L_{\text{style}}^l\)$ for layer $\(l\)$ is:

$\[L_{\text{style}}^l = \frac{1}{4N_l^2M_l^2} \sum_{i,j} (G_{ij}^l - A_{ij}^l)^2\]$

where $\(N_l\)$ is the number of feature maps in layer $\(l\)$ and $\(M_l\)$ is the height times the width of the feature map. 

The total style loss is then the sum of the style losses for each layer of interest:

$\[L_{\text{style}} = \sum_l w_l L_{\text{style}}^l\]$

where $\(w_l\)$ are weighting factors for the contribution of each layer to the total style loss.

The total loss function is then a weighted combination of the content loss and the style loss:

$\[L_{\text{total}} = \alpha L_{\text{content}} + \beta L_{\text{style}}\]$

where $\(\alpha\)$ and $\(\beta\)$ are the weights that control the relative importance of content and style, respectively.

## Results

The results of the Neural Style Transfer model are dependent on the choice of content and style images. Here is an example of the type of results that can be obtained:

![image](https://github.com/DimensionDweller/Neural_Style_Transfer/assets/75709283/b44bc6d9-878d-4c33-a96c-ca610d2afb76)
![image](https://github.com/DimensionDweller/Neural_Style_Transfer/assets/75709283/80c727f9-2a8b-4f6d-9fc1-65eaf857f670)



## Future Work and Conclusion

Neural Style Transfer is a fascinating application of Deep Learning that blends the worlds of art and technology. This project provides a robust and flexible framework for creating your own stylized images. In the future, it would be interesting to explore different ways to customize the style transfer process, such as applying different styles to different regions of the image.

In conclusion, this project demonstrates the power of Deep Learning in creating new and unique pieces of art. It's a testament to the flexibility and creativity that Deep Learning algorithms can offer. As always, there's room for improvement and experimentation, and it's exciting to consider the possibilities of what can be created with this model.

## Sources

Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. Retrieved from http://arxiv.org/abs/1508.06576v2

Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. Retrieved from http://arxiv.org/abs/1409.1556v6

Majumdar, S., Bhoi, A., & Jagadeesan, G. (2018). A Comprehensive Comparison between Neural Style Transfer and Universal Style Transfer. Retrieved from http://arxiv.org/abs/1806.00868v1

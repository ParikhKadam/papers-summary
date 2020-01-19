## Related Works
1. C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan,I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.
    - **Summary**: Changing an image, originally correctly classified (e.g. as a lion), in a way imperceptible to human eyes, can cause a DNN to label the image as something else entirely (e.g. mislabeling a lion as a library).
    - **Understanding**: Changing a few pixels in an image of lion can mislead the DNN to classify it as a library. Though the altered image seems like a lion to human, the DNN recognises it as a library.
2. 

## Introduction
It is easy to produce images that are completely unrecognizable to humans (Fig. 1), but that state-of-the-art DNNs believe to be recognizable objects with over 99% confidence (e.g. labeling with certainty that TV static (refers to black and white dots screen on TV) is a motorcycle).

![Figure 1](images/fig1.jpg)

<details>
  <summary>Comparision with [1]</summary>

    This differs from [1] in a sense that, in [1] they modified the pixels of a lion image (the image contained a legit object) and the network misclassified it as library. Whereas, in this paper, the authors take a **garbage image** i.e. image with black and white dots (no legit object), and the **model classifies it as a motorcycle**.
</details>

We use **evolutionary algorithms** or **gradient ascent** to generate images.

We also find that, for MNIST DNNs, **it is not easy to prevent the DNNs from being fooled** by retraining them with fooling images labeled as such. While retrained DNNs learn to classify the negative examples as fooling images, anew batch of fooling images can be produced that fool these new networks, even after many retraining iterations.

Here, the authors experimented with MNIST dataset. They added a new class 10 for garbage images. Prepared a dataset of such garbage images to train the network on it, with an expectation that these are garbage images and stop misclassifying them. As expected, the network learned patterns (after all that's what the ML models do) and stopped misclassifying such images as legit target classes. But now, running the algorithm (mentioned above) with this newly trained model, it generated new garbage images which the model will misclassify. Repeating these steps, it showed no good results.

### Topics discussed
1. Comparison between human vision and DNN-based computer vision
2. Performance of DNNs, in general, across different types of images than the ones they have been trained and traditionally tested on.

## Work

### Pre-trained models used
1. ImageNet DNN - AlexNet + ImageNet
2. MNIST DNN - LeNet + MNIST
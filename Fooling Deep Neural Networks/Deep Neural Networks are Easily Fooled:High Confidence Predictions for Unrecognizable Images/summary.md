## Related Works
1. C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan,I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.
    - **Summary**: Changing an image, originally correctly classified (e.g.  as a lion), in a way imperceptible to human eyes, can cause a DNN to label the image as something else entirely (e.g. mislabeling a lion as a library).
    - **Understanding**: Changing a few pixels in an image of lion can mislead the DNN to classify it as a library. Though the altered image seems like a lion to human, the DNN recognises it as a library.
2. 

## Introduction
It is easy to produce images that are completely unrecognizable to humans (Fig. 1), but that state-of-the-art DNNs believe to be recognizable objects with over 99% confidence (e.g. labeling with certainty that TV static (refers to black and white dots screen on TV) is a motorcycle).

![Figure 1](images/fig1.jpg)

### Comparision with [1]
This differs from [1] in a sense that, in [1] they modified the pixels of a lion image (the image contained a legit object) and the network misclassified it as library. Whereas, in this paper, the authors take a **grabage image** i.e. image with black and white dots (no legit object), and the **model classifies it as a motorcycle**.
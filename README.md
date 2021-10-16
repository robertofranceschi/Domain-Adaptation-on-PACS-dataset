# Domain adaptation on PACS dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d05ErjIoe4qO3AH9x9qO6YIi_XcV1paT?usp=sharing)

Custom implementation of DANN, a Domain adaptation algorithm, on PACS dataset [2] using a modified version of Alexnet [1]. 

![Network architecure](/images/dann_architecture.jpg)

## Problem description
In many ML task we often assumes that the training data are representative of the underlying distribution. However, if the inputs at test time differ
significantly from the training data, the model might not perform very well. 

> Domain adaptation is the ability to apply an algorithm trained in one or more "source domains" to a different (but related) "target domain". Domain adaptation is a subcategory of transfer learning. This scenario arises when we aim at learning from a source data distribution a well performing model on a different target data distribution. (Wikipedia)

In order to tackle the issue, a modified version of the AlexNet [1] is used allowing not only to classify input images in the source domain but also to transfer this capability to the target domain. 

### Dataset
For this anaysis the PACS dataset [2] is used. It contains overall 9991 images, splittd unevenly among:
- 7 classes (Dog, Elephant, Giraffe, Guitar, Horse, House, Person) and 
- 4 domains: `Art painting`, `Cartoon`, `Photo` and `Sketch`.

In the following figure we can appreciate the different data distribution among domains.

![example](/images/example_PACSdata_horse.jpg)
<p align = "center">
Sample images from the PACS dataset for the class 'horse' one for each domain, from left to right photo, art painting, cartoon and sketch.
</p>

#### Download dataset

The PACS dataset is available in the following repo:
```python
  # Clone github repository with data
  if not os.path.isdir('./Homework3-PACS'):
    !git clone https://github.com/MachineLearning2020/Homework3-PACS
```

## 👨‍💻 Implementation 
Regarding the neural network, the AlexNet network has been used with pretrained weights on ImageNet, and the domain classifier has been created as a copy of the FC part of AlexNet (with only two output neurons) attached at the end of the feature extractor (after the pooling layer). The domain classifier has also been initialized with the same weights as the AlexNet Fully Connected part, except for the output layer.

For easiness, in this work no data augmentation has been performed on the training dataset, since the main focus has been put on the relative difference in performance between the domain adaptation approach and the vanilla one. Though, input data is still being transformed by standardizing each channel according to mean and standard deviations of PyTorch models pretrained on ImageNet and adapting their size to the input required by AlexNet, which is 224x224 pixels (using a `transforms.CenterCrop`).

🔗 Details about the experiments: [spreadsheet experiments](https://docs.google.com/spreadsheets/d/1uLhNkXpfvKClKMzDB2up0mOgv7D9yjEpBaQuIOw4xbw).

▶ Further details about discussion and results are available in the [project report](./report.pdf).

---

### References

[1] Krizhevsky, Alex et al. “ImageNet classification with deep convolutional neural networks.” Communications of the ACM 60 (2012): 84 - 90.<br>
[2] Zhou, Kaiyang et al. “Deep Domain-Adversarial Image Generation for Domain Generalisation.” ArXiv abs/2003.06054 (2020)<br>
[3] Ganin Y. et al. “Domain-adversarial training of neural networks.” Journal of Machine Learning Research (2016)<br>
[4] Li, Da et al. “Deeper, Broader and Artier Domain Generalization.” 2017 IEEE International Conference on Computer Vision (ICCV) (2017): 5543-5551.<br>
[5] PyTorch implementation of Domain-Adversarial Neural Networks (DANN) by @fungtion - [github](https://github.com/fungtion/DANN).

# Domain adaptation on PACS dataset
Implementation of Domain adaptation on PACS dataset using a modified version of Alexnet.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d05ErjIoe4qO3AH9x9qO6YIi_XcV1paT?usp=sharing)

## Dataset
The PACS dataset is available in the following repo:
```
  # Clone github repository with data
  if not os.path.isdir('./Homework3-PACS'):
    !git clone https://github.com/MachineLearning2020/Homework3-PACS
```

Useful links: 
- [Report](/report.pdf)
- [Spreadsheet results](https://docs.google.com/spreadsheets/d/1uLhNkXpfvKClKMzDB2up0mOgv7D9yjEpBaQuIOw4xbw)


---

### References

[1] Krizhevsky, Alex et al. “ImageNet classification with deep convolutional neural networks.” Communications of the ACM 60 (2012): 84 - 90.<br>
[2] Ganin Y. et al. “Domain-adversarial training of neural networks.” Journal of Machine Learning Research (2016)<br>
[3] Li, Da et al. “Deeper, Broader and Artier Domain Generalization.” 2017 IEEE International Conference on Computer Vision (ICCV) (2017): 5543-5551.<br>
[4] PyTorch implementation of Domain-Adversarial Neural Networks (DANN) by @fungtion - [github](https://github.com/fungtion/DANN).

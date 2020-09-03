# CoNCRA: A Convolutional Neural Network Code Retrieval Approach

This repository is the official implementation of [CoNCRA: A Convolutional Neural Network Code Retrieval Approach](https://arxiv.org/abs/2030.12345). 

Our source code its an adaptation of: https://github.com/codekansas/keras-language-modeling

We propose a technique for semantic code search: A Convolutional Neural Network approach to code retrieval (CoNCRA). Our technique aims to find the code snippet that most closely matches the developer's intent, expressed in natural language. We evaluated our approach's efficacy on a dataset composed of questions and code snippets collected from Stack Overflow. Our preliminary results showed that our technique, which prioritizes local interactions (words nearby), improved the state-of-the-art (SOTA) by 5% on average, retrieving the most relevant code snippets in the top 3 (three) positions by almost 80% of the time.

![Illustration of the joint embedding technique for code retrieval.](https://github.com/mrezende/concra/blob/master/images/joint_embedding-article.png)

## Requirements

We ran our experiments at Google colab. The notebooks and source code to run our models is available at [**notebooks**](https://github.com/mrezende/concra/tree/master/notebooks) folder.



## Training

To train the model(s) in the paper, execute the following notebooks:

* train_cnn_stack_over_flow_qa.ipynb
* train_shared_cnn_stack_over_flow_qa.ipynb
* train_unif_embedding_stack_over_flow_qa.ipynb


## Evaluation

To evaluate my model on [StaQC Dataset](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset), run the following notebooks:

* evaluate_best_stack_over_flow_qa.ipynb
* evaluate_stack_over_flow_qa.ipynb



## Pre-trained Models

You can download pretrained models here:

- [CoNCRA Model](https://github.com/mrezende/concra/blob/master/models/weights/weights_epoch_ca8cf5_SharedConvolutionModelWithBatchNormalization.h5) trained on StaQC Dataset using margin 0.05, 4000 filters and kernel size 2. 



## Results

Our model achieves the following performance on :

### [StaQC Dataset](https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset)

| Model name         |  MRR  | Top 1 Accuracy |
| ------------------ |---------------- | -------------- |
| CoNCRA Model       |     0.701         |      57.7%       |
| [Unif](https://arxiv.org/abs/1905.03813)       |     0.675         |      53.9%       |
| Embedding Model       |     0.637         |      49.3%       |



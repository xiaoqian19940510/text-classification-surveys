# Text Classification papers（文本分类资料总结）

This repository contains resources for Natural Language Processing (NLP) with a focus on the task of Text Classification. The content is mainly from paper 《A Survey on Text Classification: From Shallow to Deep Learning》

该repository主要总结自然语言处理（NLP）中文本分类任务的资料。内容主要来自论文《A Survey on Text Classification: From Shallow to Deep Learning》。

# Table of Contents

<details>

<summary><b>Expand Table of Contents</b></summary><blockquote><p align="justify">

- [Surveys](#Surveys)
- [Shallow Learning Models](#Shallow-Learning-Models)
- [Deep Learning Models](#Deep-Learning-Models)
- [Datasets](#Datasets)
- [Tools and Repos](#tools-and-repos)
</p></blockquote></details>

---


## Surveys


### 2020


<details>
<summary>1. <a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey on Text Classification: From Shallow to Deep Learning</a> by<i> Qian Li, Hao Peng, Jianxin Li, Congying Xia, Renyu Yang, Lichao Sun, Philip S. Yu, Lifang He
</i></summary><blockquote><p align="justify">
Text classification is the most fundamental and essential task in natural language processing. The last decade has seen a surge of research in this area due to the unprecedented success of deep learning. Numerous methods, datasets, and evaluation metrics have been proposed in the literature, raising the need for a comprehensive and updated survey. This paper fills the gap by reviewing the state of the art approaches from 1961 to 2020, focusing on models from shallow to deep learning. We create a taxonomy for text classification according to the text involved and the models used for feature extraction and classification. We then discuss each of these categories in detail, dealing with both the technical developments and benchmark datasets that support tests of predictions. A comprehensive comparison between different techniques, as well as identifying the pros and cons of various evaluation metrics are also provided in this survey. Finally, we conclude by summarizing key implications, future research directions, and the challenges facing the research area.
</p></blockquote></details>

### 2019


<details>
<summary>1. <a href="https://arxiv.org/pdf/1904.08067.pdf">Text Classification Algorithms: A Survey</a> by<i> Kamran Kowsari, Kiana Jafari Meimandi, Mojtaba Heidarysafa, Sanjana Mendu, Laura E. Barnes, Donald E. Brown 
</i></summary><blockquote><p align="justify">
In recent years, there has been an exponential growth in the number of complex documents and texts that require a deeper understanding of machine learning methods to be able to accurately classify texts in many applications. Many machine learning approaches have achieved surpassing results in natural language processing. The success of these learning algorithms relies on their capacity to understand complex models and non-linear relationships within data. However, finding suitable structures, architectures, and techniques for text classification is a challenge for researchers. In this paper, a brief overview of text classification algorithms is discussed. This overview covers different text feature extractions, dimensionality reduction methods, existing algorithms and techniques, and evaluations methods. Finally, the limitations of each technique and their application in the real-world problem are discussed.
</p></blockquote></details>



<details>
<summary>2. <a href="https://arxiv.org/pdf/2004.03705.pdf">Deep Learning Based Text Classification: A Comprehensive Review</a> by<i> Shervin Minaee, Nal Kalchbrenner, Erik Cambria, Narjes Nikzad, Meysam Chenaghlu, Jianfeng Gao </i></summary><blockquote><p align="justify">
Deep learning based models have surpassed classical machine learning based approaches in various text classification tasks, including sentiment analysis, news categorization, question answering, and natural language inference. In this work, we provide a detailed review of more than 150 deep learning based models for text classification developed in recent years, and discuss their technical contributions, similarities, and strengths. We also provide a summary of more than 40 popular datasets widely used for text classification. Finally, we provide a quantitative analysis of the performance of different deep learning models on popular benchmarks, and discuss future research directions.
</p></blockquote></details>



## Shallow Learning Models
[:arrow_up:](#table-of-contents)

### 1961 

<details>
<summary>1. <a href="https://dl.acm.org/doi/10.1145/321075.321084">Naïve Bayes (NB)</a> (<a href="https://github.com/Gunjitbedi/Text-Classification">{Github}</a>) </summary><blockquote><p align="justify">
</p></blockquote></details>


## Deep Learning Models
[:arrow_up:](#table-of-contents)

### 2014



<details>
<summary>1. <a href="https://www.aclweb.org/anthology/D14-1181.pdf">Convolutional Neural Networks for Sentence Classification</a> TextCNN (<a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras">Github</a>)</summary><blockquote><p align="justify">
We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.
</p></blockquote></details>



## Data
[:arrow_up:](#table-of-contents)

* <a href="http://www.cs.cornell.edu/people/pabo/movie-review-data/">Movie Review (MR)</a></summary><blockquote><p align="justify">
The MR is a movie review dataset, each of which correspondsto a sentence. The corpus has 5,331 positive data and 5,331 negative data. 10-fold cross-validationby random splitting is commonly used to test MR.
</p></blockquote></details>

* <a href="http://www.cs.uic.edu/∼liub/FBS/sentiment-analysis.html">Stanford Sentiment Treebank (SST)</a></summary><blockquote><p align="justify">
The SST [175] is an extension of MR. It has two cate-gories. SST-1 with fine-grained labels with five classes. It has 8,544 training texts and 2,210 testtexts, respectively. Furthermore, SST-2 has 9,613 texts with binary labels being partitioned into6,920 training texts, 872 development texts, and 1,821 testing texts.
</p></blockquote></details>

## Tools and Repos



<details>
<summary><a href="https://github.com/ahsi/Multilingual_Event_Extraction">CMU Multilingual Event Extractor</a></summary><blockquote><p align="justify">
Python code to run ACE-style event extraction on English, Chinese, or Spanish texts 
</p></blockquote></details>

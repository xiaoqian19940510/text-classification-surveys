# Text Classification papers/surveys（文本分类资料综述总结）更新中...

This repository contains resources for Natural Language Processing (NLP) with a focus on the task of Text Classification. The content is mainly from paper 《A Survey on Text Classification: From Shallow to Deep Learning》

该repository主要总结自然语言处理（NLP）中文本分类任务的资料。内容主要来自文本分类综述论文《A Survey on Text Classification: From Shallow to Deep Learning》。

# Table of Contents

<b>Contents 目录</b></summary><blockquote><p align="justify">

- [Surveys（综述论文）](#Surveys)
- [Shallow Learning Models（浅层学习模型）](#Shallow-Learning-Models)
- [Deep Learning Models（深度学习模型）](#Deep-Learning-Models)
- [Datasets（数据集）](#Datasets)
- [Evaluation Metrics（评价指标）](#Evaluation-Metrics)
- [Tools and Repos（工具与资源）](#tools-and-repos)
</p></blockquote></details>

---


## Surveys(综述论文)


#### 2020

<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey on Text Classification: From Shallow to Deep Learning（文本分类综述：从浅层模型到深度模型）</a> by<i> Qian Li, Hao Peng, Jianxin Li, Congying Xia, Renyu Yang, Lichao Sun, Philip S. Yu, Lifang He
</a></summary><blockquote><p align="justify">
Text classification is the most fundamental and essential task in natural language processing. The last decade has seen a surge of research in this area due to the unprecedented success of deep learning. Numerous methods, datasets, and evaluation metrics have been proposed in the literature, raising the need for a comprehensive and updated survey. This paper fills the gap by reviewing the state of the art approaches from 1961 to 2020, focusing on models from shallow to deep learning. We create a taxonomy for text classification according to the text involved and the models used for feature extraction and classification. We then discuss each of these categories in detail, dealing with both the technical developments and benchmark datasets that support tests of predictions. A comprehensive comparison between different techniques, as well as identifying the pros and cons of various evaluation metrics are also provided in this survey. Finally, we conclude by summarizing key implications, future research directions, and the challenges facing the research area.
  文本分类是自然语言处理中最基本的任务。由于深度学习的空前成功，过去十年中该领域的研究激增。已有的文献提出了许多方法，数据集和评估指标，从而需要对这些内容进行全面的总结。本文回顾1961年至2020年的文本分类方法，重点是从浅层学习到深度学习的模型。根据所涉及的文本以及用于特征提取和分类的模型创建用于文本分类的分类法。然后，详细讨论这些类别中的每一个类别，涉及支持预测测试的技术发展和基准数据集。并提供了不同技术之间的全面比较，确定了各种评估指标的优缺点。最后，通过总结关键含义，未来的研究方向以及研究领域面临的挑战进行总结。
</p></blockquote></details>


## Shallow Learning Models(浅层学习模型)
[:arrow_up:](#table-of-contents)

#### 1961 

<details/>
<summary/>
<a href="https://dl.acm.org/doi/10.1145/321075.321084">Automatic indexing: An experimental inquiry 采用贝叶斯公式进行文本分类</a> (<a href="https://github.com/Gunjitbedi/Text-Classification">{Github}</a>) </summary><blockquote><p align="justify">
  This inquiry examines a technique for automatically classifying (indexing) documents according to their subject content. The task, in essence, is to have a computing machine read a document and on the basis of the occurrence of selected clue words decide to which of many subject categories the document in question belongs. This paper describes the design, execution and evaluation of a modest experimental study aimed at testing empirically one statistical technique for automatic indexing.
  这个调查研究了一种根据主题内容自动分类(索引)文档的技术。本质上，任务是让计算机读取文档，并根据所选线索词的出现情况来决定所涉文档属于哪个主题类别。本文描述了一个适度的实验研究的设计、执行和评估，旨在实证地测试一种统计技术的自动索引。
</p></blockquote></details>

#### 1967
<details/>
<summary/>
<a href="https://dl.acm.org/doi/10.1145/321075.321084">Nearest neighbor pattern classification (k-nearest neighbor classification,KNN)</a> (<a href="https://github.com/raimonbosch/knn.classifier">{Github}</a>) </summary><blockquote><p align="justify">
  The nearest neighbor decision rule assigns to an unclassified sample point the classification of the nearest of a set of previously classified points. This rule is independent of the underlying joint distribution on the sample points and their classifications, and hence the probability of errorRof such a rule must be at least as great as the Bayes probability of errorR^{\ast}--the minimum probability of error over all decision rules taking underlying probability structure into account. However, in a large sample analysis, we will show in theM-category case thatR^{\ast} \leq R \leq R^{\ast}(2 --MR^{\ast}/(M-1)), where these bounds are the tightest possible, for all suitably smooth underlying distributions. Thus for any number of categories, the probability of error of the nearest neighbor rule is bounded above by twice the Bayes probability of error. In this sense, it may be said that half the classification information in an infinite sample set is contained in the nearest neighbor.
  最近邻决策规则将一组已分类点中最接近的分类分配给一个未分类的样本点。这个规则独立于底层的联合分布的采样点及其分类的概率,因此error R概率这样的规则必须至少一样伟大的贝叶斯概率error R ^ {\ ast},可以使误差概率最小决策规则考虑潜在的概率结构。然而，在一个大型样本分析中，我们将在它们-类别的情况下显示R^{\ast} \leq R \leq R^{\ast}/(2—MR^{\ast}/(M-1))，其中对于所有适当平滑的底层分布，这些边界是可能最紧的。因此，对于任意数量的类别，最近邻规则的错误概率是贝叶斯错误概率的两倍以上。从这个意义上说，可以说一个无限样本集中一半的分类信息都包含在最近邻中。
</p></blockquote></details>

#### 1984
<details/>
<summary/>
<a href="https://dblp.org/img/paper.dark.empty.16x16.png">Classification and Regression Trees (CART)</a> (<a href="https://github.com/sayantann11/all-classification-templetes-for-ML">{Github}</a>) </summary><blockquote><p align="justify">
  分类与回归树CART是由Loe Breiman等人在1984年提出的，自提出后被广泛的应用。CART既能用于分类也能用于回归，和决策树相比较，CART把选择最优特征的方法从信息增益（率）换成了基尼指数。
</p></blockquote></details>


#### 1993
<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">C4.5: Programs for Machine Learning (C4.5)</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  C4.5算法是由Ross Quinlan开发的用于产生决策树的算法，该算法是对Ross Quinlan之前开发的ID3算法的一个扩展。C4.5算法主要应用于统计分类中，主要是通过分析数据的信息熵建立和修剪决策树。
</p></blockquote></details>

#### 1998
<details/>
<summary/>
<a href="https://xueshu.baidu.com/usercenter/paper/show?paperid=58aa6cfa340e6ae6809c5deadd07d88e&site=xueshu_se">Text categorization with Support Vector Machines: Learning with many relevant features (SVM)</a> (<a href="https://github.com/Gunjitbedi/Text-Classification">{Github}</a>) </summary><blockquote><p align="justify">
  This paper explores the use of Support Vector Machines (SVMs) for learning text classifiers from examples. It analyzes the particular properties of learning with text data and identifies why SVMs are appropriate for this task. Empirical results support the theoretical findings. SVMs achieve substantial improvements over the currently best performing methods and behave robustly over a variety of different learning tasks. Furthermore they are fully automatic, eliminating the need for manual parameter tuning.
  本文通过实例探讨了支持向量机在文本分类器学习中的应用。它分析了文本数据学习的特殊属性，并确定了支持向量机为什么适合这项任务。实证结果支持理论发现。支持向量机在当前性能最好的方法中取得了实质性的改进，并且在各种不同的学习任务中表现得很稳健。此外，它们是全自动的，不需要手动调整参数。
</p></blockquote></details>

#### 2001
<details/>
<summary/>
 <a href="https://link.springer.com/article/10.1023%2FA%3A1010933404324">Random Forests (RF)</a> (<a href="https://github.com/hexiaolang/RandomForest-In-text-classification">{Github}</a>) </summary><blockquote><p align="justify">
  Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost (Y. Freund & R. Schapire, Machine Learning: Proceedings of the Thirteenth International conference, ***, 148–156), but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.
  随机森林是树预测器的组合，因此每棵树都依赖于独立采样的随机向量的值，并且对森林中的所有树具有相同的分布。森林的泛化误差随着森林中树木数量的增加而收敛到一个极限。树分类器的森林泛化误差取决于森林中单个树的强度和它们之间的相关性。使用随机选择的特征来分割每个节点产生的错误率与Adaboost (Y. Freund & R. Schapire，机器学习:第十三届国际会议论文集，***，148-156)不相上下，但在噪声方面更健壮。内部估计监视错误、强度和相关性，这些用于显示对分割中使用的特性数量增加的响应。内部估计也被用来衡量变量的重要性。这些思想也适用于回归。
</p></blockquote></details>

## Deep Learning Models 深度学习模型
[:arrow_up:](#table-of-contents)

#### 2011
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D14-1181.pdf">Semi-supervised recursive autoencoders forpredicting sentiment distributions</a> RAE (<a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras">Github</a>)</summary><blockquote><p align="justify">
We introduce a novel machine learning frame-work based on recursive autoencoders for sentence-level prediction of sentiment labeldistributions. Our method learns vector spacerepresentations for multi-word phrases. In sentiment prediction tasks these represen-tations outperform other state-of-the-art ap-proaches on commonly used datasets, such asmovie reviews, without using any pre-definedsentiment lexica or polarity shifting rules. Wealso  evaluate  the  model’s  ability to predict sentiment distributions on a new dataset basedon confessions from the experience project. The dataset consists of personal user storiesannotated with multiple labels which, whenaggregated, form a multinomial distributionthat captures emotional reactions. Our algorithm can more accurately predict distri-butions over such labels compared to severalcompetitive baselines.
  提出了一种基于递归自动编码器的句子级情绪标签分布预测机器学习框架。我们的方法学习多词短语的向量空间扫描。在情绪预测任务中，这些代表在不使用任何预定义的情绪词汇或极性转换规则的情况下，在常用数据集(如电影评论)上胜过其他先进的应用程序。我们还评估了该模型预测情绪分布的能力，基于经验项目的告白。数据集由带有多个标签的个人用户故事组成，当聚合时，这些故事就形成了一个多项分布，可以捕捉人们的情绪反应。与几个有竞争力的基线相比，我们的算法可以更准确地预测这些标签的销量。
</p></blockquote></details>

#### 2012
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D12-1110/">Semantic compositionality through recursive matrix-vector spaces</a> MV-RNN (<a href="https://github.com/github-pengge/MV_RNN">Github</a>)</summary><blockquote><p align="justify">
Single-word vector space models have been very successful at learning lexical information. However, they cannot capture the compositional meaning of longer phrases, preventing them from a deeper understanding of language. We introduce a recursive neural network (RNN) model that learns compositional vector representations for phrases and sentences of arbitrary syntactic type and length. Our model assigns a vector and a matrix to every node in a parse tree: the vector captures the inherent meaning of the constituent, while the matrix captures how it changes the meaning of neighboring words or phrases. This matrix-vector RNN can learn the meaning of operators in propositional logic and natural language. The model obtains state of the art performance on three different experiments: predicting fine-grained sentiment distributions of adverb-adjective pairs; classifying sentiment labels of movie reviews and classifying semantic relationships such as cause-effect or topic-message between nouns using the syntactic path between them.
  单词向量空间模型在学习词汇信息方面非常成功。然而，他们不能捕捉长短语的组成意义，阻碍了他们对语言的更深的理解。我们介绍了一个递归神经网络(RNN)模型，学习组合向量表示的短语和句子的任意句法类型和长度。我们的模型为解析树中的每个节点分配一个向量和一个矩阵:向量捕获组成部分的内在含义，而矩阵捕获它如何改变邻近单词或短语的含义。该矩阵向量神经网络可以学习算子在命题逻辑和自然语言中的意义。该模型通过三个不同的实验得到艺术表现的状态:预测副词-形容词对的精细情绪分布;对电影评论的情感标签进行分类，利用名词之间的句法路径对名词之间的因果关系、主题信息等语义关系进行分类。
</p></blockquote></details>

#### 2013
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D13-1170/">Recursive deep models for semantic compositionality over a sentiment treebank</a> RNTN (<a href=" https://github.com/pondruska/DeepSentiment">Github</a>)</summary><blockquote><p align="justify">
Semantic word spaces have been very useful but cannot express the meaning of longer phrases in a principled way. Further progress towards understanding compositionality in tasks such as sentiment detection requires richer supervised training and evaluation resources and more powerful models of composition. To remedy this, we introduce a Sentiment Treebank. It includes fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and presents new challenges for sentiment composition-ality. To address them, we introduce the Recursive Neural Tensor Network. When trained on the new treebank, this model outperforms all previous methods on several metrics. It pushes the state of the art in single sentence positive/negative classification from 80% up to 85.4%. The accuracy of predicting fine-grained sentiment labels for all phrases reaches 80.7%, an improvement of 9.7% over bag of features baselines. Lastly, it is the only model that can accurately capture the effects of negation and its scope at various tree levels for both positive and negative phrases.
  语义词空间已经非常有用，但不能以一种原则的方式表达较长的短语的意义。在情感检测等任务中进一步理解作文性，需要更丰富的监督训练和评估资源以及更强大的作文模型。为了解决这个问题，我们引入了一个情绪树银行。它为11,855个句子的解析树中的215,154个短语包含了细粒度的情绪标签，并对情绪的组合提出了新的挑战。为了解决这个问题，我们引入递归神经张量网络。当在新的treebank上进行训练时，这个模型在几个指标上都优于之前所有的方法。它将单句肯定/否定分类的技术水平从80%提升到85.4%。预测所有短语的精细情绪标签的准确率达到80.7%，比包特征基线提高了9.7%。最后，该模型是唯一能够准确地捕捉否定语句在不同树级上的影响及其范围的模型。
</p></blockquote></details>

#### 2014
 <details/>
<summary/>
  <a href="http://proceedings.mlr.press/v32/le14.html">Distributed representations of sentences and documents</a> Paragraph-Vec (<a href="https://github.com/inejc/paragraph-vectors">Github</a>)</summary><blockquote><p align="justify">
Many machine learning algorithms require the input to be represented as a fixed length feature vector. When it comes to texts, one of the most common representations is bag-of-words. Despite their popularity, bag-of-words models have two major weaknesses: they lose the ordering of the words and they also ignore semantics of the words. For example, "powerful," "strong" and "Paris" are equally distant. In this paper, we propose an unsupervised algorithm that learns vector representations of sentences and text documents. This algorithm represents each document by a dense vector which is trained to predict words in the document. Its construction gives our algorithm the potential to overcome the weaknesses of bag-of-words models. Empirical results show that our technique outperforms bag-of-words models as well as other techniques for text representations. Finally, we achieve new state-of-the-art results on several text classification and sentiment analysis tasks.
  许多机器学习算法要求输入以固定长度的特征向量表示。说到文本，最常见的一种表现形式是词汇袋。尽管词包模型很流行，但它们有两个主要的弱点:它们失去了单词的顺序，而且它们还忽略了单词的语义。例如，“powerful”、“strong”和“Paris”的距离一样远。在本文中，我们提出了一种学习句子和文本文档的向量表示的无监督算法。该算法通过密集向量来表示每个文档，密集向量被训练用来预测文档中的单词。它的构造使我们的算法有可能克服单词包模型的缺点。实验结果表明，我们的技术在文本表示方面优于词汇袋模型和其他技术。最后，我们在几个文本分类和情绪分析任务上取得了最新的结果。
</p></blockquote></details>

#### 2014
 <details/>
<summary/>
  <a href="https://doi.org/10.3115/v1/p14-1062">A convolutional neural network for modelling sentences</a> DCNN (<a href="https://github.com/kinimod23/ATS_Project">Github</a>)</summary><blockquote><p align="justify">
The ability to accurately represent sentences is central to language understanding. We describe a convolutional architecture dubbed the Dynamic Convolutional Neural Network (DCNN) that we adopt for the semantic modelling of sentences. The network uses Dynamic k-Max Pooling, a global pooling operation over linear sequences. The network handles input sentences of varying length and induces a feature graph over the sentence that is capable of explicitly capturing short and long-range relations. The network does not rely on a parse tree and is easily applicable to any language. We test the DCNN in four experiments: small scale binary and multi-class sentiment prediction, six-way question classification and Twitter sentiment prediction by distant supervision. The network achieves excellent performance in the first three tasks and a greater than 25% error reduction in the last task with respect to the strongest baseline.
  准确表达句子的能力是语言理解的核心。我们描述了一种称为动态卷积神经网络(DCNN)的卷积架构，我们将其用于句子的语义建模。该网络使用动态k-Max池，一种线性序列上的全局池操作。该网络处理不同长度的输入句子，并在句子上归纳出一个特征图，能够明确地捕捉短关系和长关系。该网络不依赖于解析树，并且很容易适用于任何语言。我们通过四组实验对DCNN进行了测试:小尺度二值多类情绪预测、六向问题分类和推特远程监控情绪预测。该网络在前三个任务中都取得了优异的性能，在最后一个任务中相对于最强基线的误差减少了25%以上。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D14-1181.pdf">Convolutional Neural Networks for Sentence Classification</a> TextCNN (<a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras">Github</a>)</summary><blockquote><p align="justify">
We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.
  本文报告了一系列卷积神经网络(CNN)的实验，这些网络是在预先训练好的单词向量上训练的，用于句子级别的分类任务。我们证明了一个简单的CNN具有小的超参数调优和静态向量在多个基准测试中取得了很好的结果。通过微调学习特定任务向量可以进一步提高性能。此外，我们还建议对架构进行简单修改，以允许同时使用特定于任务的和静态向量。本文讨论的CNN模型在7项任务中的4项上改进了现有的技术，其中包括情绪分析和问题分类。
</p></blockquote></details>



## Data 数据
[:arrow_up:](#table-of-contents)

#### Sentiment Analysis (SA) 情感分析
SA is the process of analyzing and reasoning the subjective text withinemotional color. It is crucial to get information on whether it supports a particular point of view fromthe text that is distinct from the traditional text classification that analyzes the objective content ofthe text. SA can be binary or multi-class. Binary SA is to divide the text into two categories, includingpositive and negative. Multi-class SA classifies text to multi-level or fine-grained labels. 

情感分析是对带有情感色彩的主观文本进行分析和推理的过程。从文本中获取是否支持特定观点的信息是至关重要的，而传统的文本分类是分析文本的客观内容。情感分析可以是二进制的或多类的。二元情感分类是将文本分为正反两类。多类情感分类将文本分类为多层或细粒度的标签。

<details/>
<summary/> <a href="http://www.cs.cornell.edu/people/pabo/movie-review-data/">Movie Review (MR) 电影评论数据集</a></summary><blockquote><p align="justify">
The MR is a movie review dataset, each of which correspondsto a sentence. The corpus has 5,331 positive data and 5,331 negative data. 10-fold cross-validationby random splitting is commonly used to test MR. MR是一个影评数据集，每个影评是一个句子。该语料库有正样本和负样本各5331个。十折交叉验证常用来测试MR。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.cs.uic.edu/∼liub/FBS/sentiment-analysis.html">Stanford Sentiment Treebank (SST) 斯坦福情感库</a></summary><blockquote><p align="justify">
The SST [175] is an extension of MR. It has two cate-gories. SST-1 with fine-grained labels with five classes. It has 8,544 training texts and 2,210 testtexts, respectively. Furthermore, SST-2 has 9,613 texts with binary labels being partitioned into6,920 training texts, 872 development texts, and 1,821 testing texts.
 SST是MR的扩展版本，一共有两种类型。SST-1有五个类别标签，有8,544个训练文本和2,210个测试文本。SST-2有两个类别，共9,613个文本，被划分为6,920个训练文本、872个开发文本和1,821个测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.cs.pitt.edu/mpqa/">The Multi-Perspective Question Answering (MPQA)多视角问答数据集</a></summary><blockquote><p align="justify">
The MPQA is an opinion dataset. It has two class labels and also an MPQA dataset of opinion polarity detection sub-tasks.MPQA includes 10,606 sentences extracted from news articles from various news sources. It shouldbe noted that it contains 3,311 positive texts and 7,293 negative texts without labels of each text.
 MPQA是一个观点数据集。它有两个类标签和一个MPQA数据集的观点极性检测子任务。MPQA包含了10,606个句子，这些句子是从各种新闻来源的新闻文章中提取的。应当指出的是，其中有3311个正样本和7293个负样本，每个文本没有标签。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/kdd/DiaoQWSJW14">IMDB reviews IMDB评论</a></summary><blockquote><p align="justify">
The IMDB review is developed for binary sentiment classification of filmreviews with the same amount in each class. It can be separated into training and test groups onaverage, by 25,000 comments per group.
 IMDB影评是对每类影评进行相同数量的二元情感分类。它平均可以被分成训练组和测试组，每组有25000条评论。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/emnlp/TangQL15">Yelp reviews Yelp评论</a></summary><blockquote><p align="justify">
The Yelp review is summarized from the Yelp Dataset Challenges in 2013,2014, and 2015. This dataset has two categories. Yelp-2 of these were used for negative and positiveemotion classification tasks, including 560,000 training texts and 38,000 test texts. Yelp-5 is used todetect fine-grained affective labels with 650,000 training and 50,000 test texts in all classes.
 Yelp评论数据来自于2013年、2014年和2015年的Yelp数据集挑战。这个数据集有两个类别。其中的Yelp-2被用于消极和积极情绪分类任务，包括560,000篇训练文本和38,000篇测试文本。使用Yelp-5对65万篇训练文本和5万篇测试文本进行精细的情感标签检测。
</p></blockquote></details>

<details/>
<summary/> <a href="https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products">Amazon Reviews (AM) 亚马逊评论数据集</a></summary><blockquote><p align="justify">
The AM is a popular corpus formed by collecting Amazon websiteproduct reviews [190]. This dataset has two categories. The Amazon-2 with two classes includes 3,600,000 training sets and 400,000 testing sets. Amazon-5, with five classes, includes 3,000,000 and650,000 comments for training and testing.
 AM是一个收集亚马逊网站产品评论的流行语料库。这个数据集有两个类别。带有两个类的Amazon-2包括360万个训练集和40万个测试集。Amazon-5有五个类，包含300万条和65万条培训和测试评论。
</p></blockquote></details>


#### News Classification (NC) 新闻分类数据集
News content is one of the most crucial information sources which hasa critical influence on people. The NC system facilitates users to get vital knowledge in real-time.News classification applications mainly encompass: recognizing news topics and recommendingrelated news according to user interest. The news classification datasets include 20NG, AG, R8, R52,Sogou, and so on. Here we detail several of the primary datasets.

新闻内容是影响人们生活的最重要的信息来源之一。数控系统便于用户实时获取重要知识。新闻分类应用主要包括:识别新闻话题，根据用户兴趣推荐相关新闻。新闻分类数据集包括20NG、AG、R8、R52、Sogou等。这里我们详细介绍了几个主要数据集。

<details/>
<summary/>
<a href="http://ana.cachopo.org/datasets-for-single-label-text-categorization">20 Newsgroups (20NG)</a></summary><blockquote><p align="justify">
 The 20NG is a newsgroup text dataset. It has 20 categories withthe same number of each category and includes 18,846 texts.
 20NG是一个新闻组文本数据集。它有20个类别，每个类别的数目相同，包括18,846个文本。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html">AG News (AG)</a></summary><blockquote><p align="justify">
The AG News is a search engine for news from academia, choosingthe four largest classes. It uses the title and description fields of each news. AG contains 120,000texts for training and 7,600 texts for testing.
 AG新闻是一个搜索来自学术界的新闻的引擎，选择了四个最大的类别。它使用每个新闻的标题和描述字段。AG包含120,000篇训练文本和7,600篇测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://www.cs.umb.edu/~smimarog/textmining/datasets/">R8 and R52</a></summary><blockquote><p align="justify">
R8 and R52 are two subsets which are the subset of Reuters. R8 has 8categories, divided into 2,189 test files and 5,485 training courses. R52 has 52 categories, split into6,532 training files and 2,568 test files.
 R8和R52是两个子集，它们是Reuters的子集。R8有8个类别，分为2189个测试文件和5485个训练文件。R52有52个类别，分为6,532个训练文件和2,568个测试文件。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/conf/cncl/SunQXH19.bib">Sogou News (Sogou) 搜狗新闻</a></summary><blockquote><p align="justify">
The Sogou News combines two datasets, including SogouCA andSogouCS news sets. The label of each text is the domain names in the URL.
 搜狗新闻结合了两个数据集，包括SogouCA和sogoucs新闻集。每个文本的标签是URL中的域名。
</p></blockquote></details>

### Topic Labeling (TL) 话题标签
The topic analysis attempts to get the meaning of the text by defining thesophisticated text theme. The topic labeling is one of the essential components of the topic analysistechnique, intending to assign one or more subjects for each document to simplify the topic analysis.

主题分析试图通过对文本主题的界定来获得文本的意义。话题标签是主题分析技术的重要组成部分之一，用于为每个文档分配一个或多个主题，以简化主题分析。

<details/>
<summary/> <a href="https://dblp.org/rec/journals/semweb/LehmannIJJKMHMK15.bib">DBpedia</a></summary><blockquote><p align="justify">
The DBpedia is a large-scale multi-lingual knowledge base generated usingWikipedia’s most ordinarily used infoboxes. It publishes DBpedia each month, adding or deletingclasses and properties in every version. DBpedia’s most prevalent version has 14 classes and isdivided into 560,000 training data and 70,000 test data. 
 DBpedia是使用wikipedia最常用的信息框生成的大型多语言知识库。它每月发布DBpedia，在每个版本中添加或删除类和属性。DBpedia最流行的版本有14个类，分为560,000个训练数据和70,000个测试数据。
</p></blockquote></details>

<details/>
<summary/> <a href="http://davis.wpi.edu/xmdv/datasets/ohsumed.html">Ohsumed</a></summary><blockquote><p align="justify">
The Ohsumed belongs to the MEDLINE database. It includes 7,400 texts andhas 23 cardiovascular disease categories. All texts are medical abstracts and are labeled into one ormore classes.
 Ohsumed属于MEDLINE数据库。它包括7400个文本，有23个心血管疾病类别。所有文本都是医学摘要，并被标记为一个或多个类。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Yahoo answers (YahooA) 雅虎问答</a></summary><blockquote><p align="justify">
The YahooA is a topic labeling task with 10 classes. It includes140,000 training data and 5,000 test data. All text contains three elements, being question titles,question contexts, and best answers, respectively.
 YahooA是一个包含10个类的主题标记任务。它包括140,000个训练数据和5,000个测试数据。所有文本包含三个元素，分别是问题标题、问题上下文和最佳答案。
</p></blockquote></details>


### Question Answering (QA) 问答
The QA task can be divided into two types: the extractive QA and thegenerative QA. The extractive QA gives multiple candidate answers for each question to choosewhich one is the right answer. Thus, the text classification models can be used for the extractiveQA task. The QA discussed in this paper is all extractive QA. The QA system can apply the textclassification model to recognize the correct answer and set others as candidates. The questionanswering datasets include SQuAD, MS MARCO, TREC-QA, WikiQA, and Quora [209]. Here wedetail several of the primary datasets.

QA任务可以分为两种类型:抽取型QA和生成型QA。抽取式QA为每个问题提供多个候选答案，以选择哪一个是正确答案。因此，文本分类模型可以用于抽取式QA任务。本文所讨论的质量保证都是抽取式质量保证。QA系统可以应用文本分类模型来识别正确答案，并将其他答案设置为考生。问答的数据集包括SQuAD、MS MARCO、treci - qa、WikiQA和Quora。这里我们详细介绍了几个主要的数据集。


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Stanford Question Answering Dataset (SQuAD） 斯坦福问答数据集</a></summary><blockquote><p align="justify">
The SQuAD is a set of question and answer pairs obtained from Wikipedia articles. The SQuAD has two categories. SQuAD1.1 contains 536 pairs of 107,785 Q&A items. SQuAD2.0 combines 100,000 questions in SQuAD1.1 with morethan 50,000 unanswerable questions that crowd workers face in a form similar to answerable questions.
这个数据集是一组从维基百科文章中获得的问题和答案。球队分为两类。SQuAD1.1包含536对107,785个问答题。SQuAD2.0将SQuAD1.1中的100,000个问题与5万多个无法回答的问题组合在一起，这些问题以类似于可回答问题的形式出现。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">MS MARCO</a></summary><blockquote><p align="justify">
The MS MARCO contains questions and answers. The questions and part ofthe answers are sampled from actual web texts by the Bing search engine. Others are generative. Itis used for developing generative QA systems released by Microsoft.
MS MARCO包含了问题和答案。这些问题和部分答案都是由Bing搜索引擎从实际的网络文本中抽取的，其他的则是生成式的。它用于开发由微软发布的生成式QA系统。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">TREC-QA</a></summary><blockquote><p align="justify">
The TREC-QA includes 5,452 training texts and 500 testing texts. It has two versions. TREC-6 contains 6 categories, and TREC-50 has 50 categories.
TREC-QA包括5,452个训练文本和500个测试文本。它有两个版本。TREC-6包含6个类别，TREC-50包含50个类别。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">WikiQA</a></summary><blockquote><p align="justify">
The WikiQA dataset includes questions with no correct answer, which needs toevaluate the answer.
WikiQA数据集包含没有正确答案的问题，需要对答案进行评估。
</p></blockquote></details>



### Natural Language Inference (NLI) 自然语言推理
NLI is used to predict whether the meaning of one text canbe deduced from another. Paraphrasing is a generalized form of NLI. It uses the task of measuringthe semantic similarity of sentence pairs to decide whether one sentence is the interpretation ofanother. The NLI datasets include SNLI, MNLI, SICK, STS, RTE, SciTail, MSRP, etc. Here we detailseveral of the primary datasets.

NLI是用来预测一个文本的意思是否可以从另一个文本中推断出来。释义是NLI的一种概括形式。它的任务是测量句子对的语义相似度，以决定一个句子是否是另一个句子的解释。NLI数据集包括SNLI, MNLI, SICK, STS, RTE, SciTail, MSRP等。这里我们详细介绍了几个主要数据集。

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">The Stanford Natural Language Inference (SNLI)</a></summary><blockquote><p align="justify">
The SNLI is generally applied toNLI tasks. It contains 570,152 human-annotated sentence pairs, including training, development,and test sets, which are annotated with three categories: neutral, entailment, and contradiction.
  SNLI一般应用于各种任务。它包含570,152对人注释句子，包括训练集、开发集和测试集，并以中性句、隐含句和矛盾句三大类注释.
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Multi-Genre Natural Language Inference (MNLI)</a></summary><blockquote><p align="justify">
The Multi-NLI is an expansion of SNLI, embracing a broader scope of written and spoken text genres. It includes 433,000 sentencepairs annotated by textual entailment labels.
  Multi-NLI是SNLI的扩展，包括更广泛的书面和口语文本类型。它包括433,000个句子，并附有文本蕴涵标签。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Sentences Involving Compositional Knowledge (SICK)</a></summary><blockquote><p align="justify">
The SICK contains almost10,000 English sentence pairs. It consists of neutral, entailment and contradictory labels.
The SICK包含近10,000对英语句子。它由中性、含蓄和矛盾的标签构成。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Microsoft Research Paraphrase (MSRP)</a></summary><blockquote><p align="justify">
The MSRP consists of sentence pairs, usuallyfor the text-similarity task. Each pair is annotated by a binary label to discriminate whether theyare paraphrases. It respectively includes 1,725 training and 4,076 test sets.
MSRP由句子对组成，通常用于文本相似任务。每一对都由一个二进制标签注释，以区分它们是否是意译。分别包含1725个训练集和4076个测试集。
</p></blockquote></details>



### Dialog Act Classification (DAC) 对话行为分类
A dialog act describes an utterance in a dialog based on semantic,pragmatic, and syntactic criteria. DAC labels a piece of a dialog according to its category of meaningand helps learn the speaker’s intentions. It is to give a label according to dialog. Here we detailseveral of the primary datasets, including DSTC 4, MRDA, and SwDA.

对话行为描述基于语义、语用和句法标准的对话中的话语。DAC根据对话的意义类别给对话片贴上标签，并帮助了解讲话者的意图。根据对话给一个标签。这里我们详细介绍了几个主要数据集，包括DSTC 4、MRDA和SwDA。

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Dialog State Tracking Challenge 4 (DSTC 4)</a></summary><blockquote><p align="justify">
The DSTC 4 is used for dialog act classi-fication. It has 89 training classes, 24,000 training texts, and 6,000 testing texts.
DSTC 4用于对话行为分类。它有89个培训班，24000个培训文本，6000个测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">ICSI Meeting Recorder Dialog Act (MRDA)</a></summary><blockquote><p align="justify">
The MRDA is used for dialog act classifi-cation. It has 5 training classes, 51,000 training texts, 11,000 testing texts, and 11,000 validation texts.
MRDA用于对话行为分类。它有5个培训类、51,000个培训文本、11,000个测试文本和11,000个验证文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Switchboard Dialog Act (SwDA)</a></summary><blockquote><p align="justify">
The SwDA is used for dialog act classification. It has43 training classes, 1,003,000 training texts, 19,000 testing texts and 112,000 validation texts.
SwDA用于对话行为分类。它有43个培训课程，1,003,000个培训文本，19,000个测试文本和112,000个验证文本。
</p></blockquote></details>


### Multi-label datasets 多标签数据集
In multi-label classification, an instance has multiple labels, and each la-bel can only take one of the multiple classes. There are many datasets based on multi-label textclassification. It includes Reuters, Education, Patent, RCV1, RCV1-2K, AmazonCat-13K, BlurbGen-reCollection, WOS-11967, AAPD, etc. Here we detail several of the main datasets.

在多标签分类中，一个实例有多个标签，每个la-bel只能获取多个类中的一个。有许多基于多标签文本分类的数据集。包括Reuters、Education、Patent、RCV1、RCV1-2k、AmazonCat-13K、BlurbGen-reCollection, WOS-11967、AAPD等。这里我们详细介绍了几个主要的数据集。

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Reuters news</a></summary><blockquote><p align="justify">
The Reuters is a popularly used dataset for text classification fromReuters financial news services. It has 90 training classes, 7,769 training texts, and 3,019 testingtexts, containing multiple labels and single labels. There are also some Reuters sub-sets of data,such as R8, BR52, RCV1, and RCV1-v2.
Reuters是一个广泛使用的数据集，用于Reuters财经新闻服务的文本分类。它有90个训练类别，7,769个训练文本和3,019个测试文本，包含多个标签和单个标签。还有一些Reuters数据子集，如R8、BR52、RCV1和RCV1-v2。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Patent Dataset</a></summary><blockquote><p align="justify">
The Patent Dataset is obtained from USPTO1, which is a patent system gratingU.S. patents containing textual details such title and abstract. It contains 100,000 US patents awardedin the real-world with multiple hierarchical categories.
该数据集来自美国专利系统USPTO1。专利包含文本细节，如标题和摘要。它包含了现实世界中获得的10万项美国专利，具有多重等级类别。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Reuters Corpus Volume I (RCV1) and RCV1-2K</a></summary><blockquote><p align="justify">
The RCV1 is collected from Reuters News articles from 1996-1997, which is human-labeled with 103 categories. It consists of 23,149 training and 784,446 testing texts, respectively. The RCV1-2K dataset has the same features as the RCV1. However, the label set of RCV1-2K has been expanded with some new labels. It contains2456 labels.
RCV1收集自1996-1997年Reuters News的新闻文章，被人为标记为103个类别。它分别由23,149个培训文本和784,446个测试文本组成。RCV1- 2k数据集具有与RCV1相同的特征。然而，RCV1-2K的标签集已经扩展了一些新的标签。它包含2456标签。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Web of Science (WOS-11967)</a></summary><blockquote><p align="justify">
The WOS-11967 is crawled from the Web of Science,consisting of abstracts of published papers with two labels for each example. It is shallower, butsignificantly broader, with fewer classes in total.
WOS-11967抓取自Web of Science，其中包含已发表论文的摘要，每个例子有两个标签。它较浅，但明显较宽，类总数较少。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Arxiv Academic Paper Dataset (AAPD)</a></summary><blockquote><p align="justify">
The AAPD is a large dataset in the computer science field for the multi-label text classification from website2. It has 55,840 papers, including the abstract and the corresponding subjects with 54 labels in total. The aim is to predict the corresponding subjects of each paper according to the abstract.
AAPD是计算机科学领域中用于website2多标签文本分类的大型数据集。论文55,840篇，包括摘要及相应科目，共计54个标签。目的是根据摘要预测每篇论文对应的主题。
</p></blockquote></details>


### Others 其他
There are some datasets for other applications, such as Geonames toponyms, Twitter posts,and so on.

还有一些用于其他应用程序的数据集，比如Geonames toponyms、Twitter帖子等等。

## Evaluation Metrics 评价指标
[:arrow_up:](#table-of-contents)

In terms of evaluating text classification models, accuracy and F1 score are the most used to assessthe text classification methods. Later, with the increasing difficulty of classification tasks or theexistence of some particular tasks, the evaluation metrics are improved. For example, evaluationmetrics such as P@K and Micro-F1 are used to evaluate multi-label text classification performance,and MRR is usually used to estimate the performance of QA tasks.

### Single-label metrics 单标签评价指标
Single-label text classification divides the text into one of the most likelycategories applied in NLP tasks such as QA, SA, and dialogue systems [9]. For single-label textclassification, one text belongs to just one catalog, making it possible not to consider the relationsamong labels. Here we introduce some evaluation metrics used for single-label text classificationtasks.

单标签文本分类将文本分成一个最可能应用于NLP任务的类别，如QA、SA和对话系统。对于单标签文本分类，一个文本只属于一个目录，使得不考虑标签之间的关系成为可能。在这里，我们介绍一些评价指标用于单标签文本分类任务。


<details/>
<summary/> Accuracy and Error Rate</summary><blockquote><p align="justify">
Accuracy and Error Rate are the fundamental metrics for a text classi-fication model. The Accuracy and Error Rate are respectively defined as
</p></blockquote></details>

<details/>
<summary/> Precision, Recall and F1</summary><blockquote><p align="justify">
These are vital metrics utilized for unbalanced test sets regardless ofthe standard type and error rate. For example, most of the test samples have a class label. F1 is theharmonic average of Precision and Recall. Accuracy, Recall, and F1 as defined
  
  The desired results will be obtained when the accuracy, F1 and recall value reach 1. On the contrary,when the values become 0, the worst result is obtained. For the multi-class classification problem,the precision and recall value of each class can be calculated separately, and then the performanceof the individual and whole can be analyzed.
</p></blockquote></details>

<details/>
<summary/>Exact Match (EM)</summary><blockquote><p align="justify">
The EM is a metric for QA tasks measuring the prediction that matches all theground-truth answers precisely. It is the primary metric utilized on the SQuAD dataset.

</p></blockquote></details>


<details/>
<summary/>Mean Reciprocal Rank (MRR)</summary><blockquote><p align="justify">
The EM is a metric for QA tasks measuring the prediction that matches all theground-truth answers precisely. It is the primary metric utilized on the SQuAD dataset.
</p></blockquote></details>

<details/>
<summary/>Hamming-loss (HL)</summary><blockquote><p align="justify">
The HL assesses the score of misclassified instance-label pairs wherea related label is omitted or an unrelated is predicted.
</p></blockquote></details>


### Multi-label metrics 多标签评价指标
Compared with single-label text classification, multi-label text classifica-tion divides the text into multiple category labels, and the number of category labels is variable. These metrics are designed for single label text classification, which are not suitable for multi-label tasks. Thus, there are some metrics designed for multi-label text classification.

<details/>
<summary/>Micro−F1</summary><blockquote><p align="justify">
The Micro−F1 is a measure that considers the overall accuracy and recall of alllabels. The Micro−F1is defined as
</p></blockquote></details>


<details/>
<summary/>Macro−F1</summary><blockquote><p align="justify">
The Macro−F1 calculates the average F1 of all labels. Unlike Micro−F1, which setseven weight to every example, Macro−F1 sets the same weight to all labels in the average process. Formally, Macro−F1is defined as
</p></blockquote></details>

In addition to the above evaluation metrics, there are some rank-based evaluation metrics forextreme multi-label classification tasks, including P@K and NDCG@K.

<details/>
<summary/>Precision at Top K (P@K)</summary><blockquote><p align="justify">
The P@K is the precision at the top k. ForP@K, each text has a set of L ground truth labels Lt={l0,l1,l2...,lL−1}, in order of decreasing probability Pt=p0,p1,p2...,pQ−1.The precision at k is
</p></blockquote></details>

<details/>
<summary/>Normalized Discounted Cummulated Gains (NDCG@K)</summary><blockquote><p align="justify">
The NDCG at k is
</p></blockquote></details>


## Tools and Repos
[:arrow_up:](#table-of-contents)


<details>
<summary><a href="https://github.com/Tencent/NeuralNLP-NeuralClassifier">NeuralClassifier</a></summary><blockquote><p align="justify">
腾讯的开源NLP项目
</p></blockquote></details>


<details>
<summary><a href="https://github.com/nocater/baidu_nlp_project2">baidu_nlp_project2</a></summary><blockquote><p align="justify">
百度NLP项目
</p></blockquote></details>



<details>
<summary><a href="https://github.com/TianWuYuJiangHenShou/textClassifier">Multi-label</a></summary><blockquote><p align="justify">
多标签文本分类项目
</p></blockquote></details>

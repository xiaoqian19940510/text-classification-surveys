# Text Classification papers/surveys（文本分类资料综述总结）更新中...

This repository contains resources for Natural Language Processing (NLP) with a focus on the task of Text Classification. The content is mainly from paper 《A Survey on Text Classification: From Shallow to Deep Learning》
（该repository主要总结自然语言处理（NLP）中文本分类任务的资料。内容主要来自文本分类综述论文《A Survey on Text Classification: From Shallow to Deep Learning》）


# Table of Contents

- [Surveys](#surveys)
- [Deep Learning Models](#deep-learning-models)
- [Shallow Learning Models](#shallow-learning-models)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Research Challenges](#future-research-challenges)
- [Tools and Repos](#tools-and-repos)
</p></blockquote></details>



# Surveys
[:arrow_up:](#table-of-contents)

<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey on Text Classification: From Shallow to Deep Learning,2020</a> by<i> Qian Li, Hao Peng, Jianxin Li, Congying Xia, Renyu Yang, Lichao Sun, Philip S. Yu, Lifang He
</a></summary><blockquote><p align="justify">
Text classification is the most fundamental and essential task in natural language processing. The last decade has seen a surge of research in this area due to the unprecedented success of deep learning. Numerous methods, datasets, and evaluation metrics have been proposed in the literature, raising the need for a comprehensive and updated survey. This paper fills the gap by reviewing the state of the art approaches from 1961 to 2020, focusing on models from shallow to deep learning. We create a taxonomy for text classification according to the text involved and the models used for feature extraction and classification. We then discuss each of these categories in detail, dealing with both the technical developments and benchmark datasets that support tests of predictions. A comprehensive comparison between different techniques, as well as identifying the pros and cons of various evaluation metrics are also provided in this survey. Finally, we conclude by summarizing key implications, future research directions, and the challenges facing the research area.
  
  文本分类综述：文本分类是自然语言处理中最基本，也是最重要的任务。由于深度学习的成功，在过去十年里该领域的相关研究激增。 鉴于已有的文献已经提出了许多方法，数据集和评估指标，因此更加需要对上述内容进行全面的总结。本文通过回顾1961年至2020年的最新方法填补来这一空白，主要侧重于从浅层学习模型到深度学习模型。本文首先根据方法所涉及的文本，以及用于特征提取和分类的模型，构建了一个对不同方法进行分类的规则。然后本文将详细讨论每一种类别的方法，涉及该方法相关预测技术的发展和基准数据集。此外，本综述还提供了不同方法之间的全面比较，并确定了各种评估指标的优缺点。最后，本文总结了该研究领域的关键影响因素，未来研究方向以及所面临的挑战。
  
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/picture1.png)

![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/picture2.png)

</p></blockquote></details>



# Deep Learning Models
[:arrow_up:](#table-of-contents)

#### 2020
 <details/>
<summary/>
  <a href="https://transacl.org/ojs/index.php/tacl/article/view/1853">Spanbert: Improving pre-training by representing and predicting spans</a>  --- SpanBERT--- by<i> Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy
</a>(<a href="https://github.com/facebookresearch/SpanBERT">Github</a>)</summary><blockquote><p align="justify">
We present SpanBERT, a pre-training method that is designed to better represent and predict spans of text. Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary representations to predict the entire content of the masked span, without relying on the individual token representations within it. SpanBERT consistently outperforms BERT and our better-tuned baselines, with substantial gains on span selection tasks such as question answering and coreference resolution. In particular, with the same training data and model size as BERT-Large, our single model obtains 94.6% and 88.7% F1 on SQuAD 1.1 and 2.0 respectively. We also achieve a new state of the art on the OntoNotes coreference resolution task (79.6% F1), strong performance on the TACRED relation extraction benchmark, and even gains on GLUE.
  
  主要贡献：Span Mask机制，不再对随机的单个token添加mask，随机对邻接分词添加mask；Span Boundary Objective(SBO)训练，使用分词边界表示预测被添加mask分词的内容；一个句子的训练效果更好。
  
  本文提出了一种名为SpanBERT的预训练方法，旨在更好地表示和预测文本范围。本文的方法通过在BERT模型上进行了以下改进：(1)在进行掩膜操作（masking）时，对随机的一定范围内的词语进行掩膜覆盖，而不是单个随机的词语。(2)训练掩膜边界表征来预测掩膜覆盖的整个内容，而不依赖于其中的单个词语表示。SpanBERT的表现始终优于BERT和本文优化后的基线方法，并且在范围选择任务(如问题回答和指代消解)上取得了实质性的进展。特别地，在训练数据和模型尺寸与BERT- large相同的情况下，本文的单模型在SQuAD 1.1和2.0上分别取得了94.6%和88.7%地F1分数。此外还达到了OntoNotes指代消解任务(79.6% F1)的最优效果，同时在TACRED关系提取基准测试上的展现了强大的性能，并且在GLUE数据集上也取得了性能提升。
 
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/picture3.png)

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://openreview.net/forum?id=H1eA7AEtvS">ALBERT: A lite BERT for self-supervised learning of language representations</a> --- ALBERT--- by<i> Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut
</a> (<a href="https://github.com/google-research/ALBERT">Github</a>)</summary><blockquote><p align="justify">
Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems,  we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT~\citep{devlin2018bert}. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and \squad benchmarks while having fewer parameters compared to BERT-large. The code and the pretrained models are available at https://github.com/google-research/ALBERT.
  
  论文主要贡献：瘦身版BERT，全新的参数共享机制。对embedding因式分解，隐层embedding带有上线文信息；跨层参数共享，全连接和attention层都进行参数共享，效果下降，参数减少，训练时间缩短；句间连贯 
  
  在对自然语言表示进行预训练时增加模型大小通常会提高下游任务的性能。然而，在某种程度上由于GPU/TPU的内存限制和训练时间的增长，进一步的提升模型规模变得更加困难。为了解决这些问题，本文提出了两种参数缩减技术来降低内存消耗，并提高BERT的训练速度。全面的实验表明，本文的方法能够让模型在规模可伸缩性方面远优于BERT。本文还使用了一种对句子间连贯性进行建模的自监督损失函数，并证明这种方法对多句子输入的下游任务确实有帮助。本文的最佳模型在GLUE, RACE和SQuAD数据集上取得了新的最佳效果，并且参数量低于BERT-large。代码和预训练模型已经开源到github：https://github.com/google-research/ALBERT.
</p></blockquote></details>

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Roberta: A robustly optimized BERT pretraining approach</a> --- Roberta--- by<i> Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
</a>  (<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements. We release our models and code.
  
  主要贡献：更多训练数据、更大batch size、训练时间更长；去掉NSP；训练序列更长；动态调整Masking机制，数据拷贝十份，每句话会有十种不同的mask方式。 
  
 语言模型的预训练能带来显著的性能提升，但详细比较不同的预训练方法仍然具有挑战性，这是因为训练的计算开销很大，并且通常是在不同大小的非公开数据集上进行的，此外正如本文将说明的，超参数的选择对最终结果有很大的影响。本文提出了一项针对BERT预训练的复制研究，该研究仔细测试了许多关键超参数和训练集大小对预训练性能的影响。实验发现BERT明显训练不足，并且在经过预训练后可以达到甚至超过其后发布的每个模型的性能。本文最好的模型在GLUE，RACE和SQuAD数据集上达到了SOTA效果。这些结果突出了以前被忽略的设计选择的重要性，并对最近一些其他文献所提出的性能增长提出了质疑。模型和代码已经开源到github。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding">Xlnet: Generalized autoregressive pretraining for language understanding</a> --- Xlnet--- by<i> Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R. Salakhutdinov, Quoc V. Le
</a>  (<a href="https://github.com/zihangdai/xlnet">Github</a>)</summary><blockquote><p align="justify">
With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment setting, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.
  
  主要贡献：采用自回归（AR）模型替代自编码（AE）模型，解决mask带来的负面影响；双流自注意力机制；引入transformer-xl，解决超长序列的依赖问题；采用相对位置编码 
  
 凭借对双向上下文进行建模的能力，与基于自回归语言模型的预训练方法（GPT）相比，基于像BERT这种去噪自编码的预训练方法能够达到更好的性能。然而，由于依赖于使用掩码(masks)去改变输入，BERT忽略了屏蔽位置之间的依赖性并且受到预训练与微调之间差异的影响。结合这些优缺点，本文提出了XLNet，这是一种通用的自回归预训练方法，其具有以下优势：（1）通过最大化因式分解次序的概率期望来学习双向上下文，（2）由于其自回归公式，克服了BERT的局限性。此外，XLNet将最先进的自回归模型Transformer-XL的创意整合到预训练中。根据经验性测试，XLNet在20个任务上的表现优于BERT，并且往往有大幅度提升，并在18个任务中实现最先进的结果，包括问答，自然语言推理，情感分析和文档排序。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/picture4.png)  
  
</p></blockquote></details>




 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P19-1441/">Multi-task deep neural networks for natural language understanding</a> --- MT-DNN--- by<i> Xiaodong Liu, Pengcheng He, Weizhu Chen, Jianfeng Gao
</a>  (<a href="https://github.com/namisan/mt-dnn">Github</a>)</summary><blockquote><p align="justify">
In this paper, we present a Multi-Task Deep Neural Network (MT-DNN) for learning representations across multiple natural language understanding (NLU) tasks. MT-DNN not only leverages large amounts of cross-task data, but also benefits from a regularization effect that leads to more general representations to help adapt to new tasks and domains. MT-DNN extends the model proposed in Liu et al. (2015) by incorporating a pre-trained bidirectional transformer language model, known as BERT (Devlin et al., 2018). MT-DNN obtains new state-of-the-art results on ten NLU tasks, including SNLI, SciTail, and eight out of nine GLUE tasks, pushing the GLUE benchmark to 82.7% (2.2% absolute improvement) as of February 25, 2019 on the latest GLUE test set. We also demonstrate using the SNLI and SciTail datasets that the representations learned by MT-DNN allow domain adaptation with substantially fewer in-domain labels than the pre-trained BERT representations. Our code and pre-trained models will be made publicly available.
  
  主要贡献：多任务学习机制训练模型，提高模型的泛化性能。 
  
  本文提出了一种多任务深度神经网络 (MT-DNN) ，用于跨多种自然语言理解任务（NLU）的学习表示。MT-DNN 不仅利用大量跨任务数据，而且得益于一种正则化效果，这种效果可以帮助产生更通用的表示，从而有助于扩展到新的任务和领域。MT-DNN 扩展引入了一个预先训练的双向转换语言模型BERT。MT-DNN在十个自然语言处理任务上取得了SOTA的成果，包括SNLI、SciTail和九个GLUE任务中的八个，将GLUE的baseline提高到了82.7 % (2.2 %的绝对改进)。在SNLI和Sc-iTail数据集上的实验证明，与预先训练的BERT表示相比，MT-DNN学习到的表示可以在域内标签数据较少的情况下展现更好的领域适应性。代码和预先训练好的模型将进行开源。
  
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/picture5.png)  
  
</p></blockquote></details>

  


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/n19-1423">BERT: pre-training of deep bidirectional transformers for language understanding</a> --- BERT--- by<i> Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
</a> (<a href="https://github.com/google-research/bert">Github</a>)</summary><blockquote><p align="justify">
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
  
  主要贡献：BERT是双向的Transformer block连接，增加词向量模型泛化能力，充分描述字符级、词级、句子级关系特征。真正的双向encoding，Masked LM类似完形填空；transformer做encoder实现上下文相关，而不是bi-LSTM，模型更深，并行性更好；学习句子关系表示，句子级负采样 
  
本文介绍了一种新的语言表示模型BERT，它表示Transformers的双向编码器表示。与最近的语言表示模型不同(Peters et al., 2018; Radford et al., 2018)，BERT通过在所有层的上下文联合调节来预训练深层双向表示。因此，只需一个额外的输出层就可以对预先训练好的BERT表示进行微调，以便为各种任务创建最先进的模型，例如问答和语言推断，而无需基本的任务特定架构修改。BERT概念简单，经验丰富。它在11项自然语言处理任务中获得了最新的技术成果，包括将GLUE的基准值提高到80.4%(7.6%的绝对改进)、多项准确率提高到86.7%(5.6%的绝对改进)、将SQuAD v1.1的问答测试F1基准值提高到93.2(1.5的绝对改进)，以及将SQuAD v2.0测试的F1基准值提高到83.1（5.1的绝对改进）。
  
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/picture6.png)  
  
</p></blockquote></details>

  

 <details/>
<summary/>
  <a href="https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4725">Graph convolutional networks for text classification</a> --- TextGCN---  by<i> Liang Yao, Chengsheng Mao, Yuan Luo
</a>(<a href="https://github.com/yao8839836/text_gcn">Github</a>)</summary><blockquote><p align="justify">
Text classification is an important and classical problem in natural language processing. There have been a number of studies that applied convolutional neural networks (convolution on regular grid, e.g., sequence) to classification. However, only a limited number of studies have explored the more flexible graph convolutional neural networks (convolution on non-grid, e.g., arbitrary graph) for the task. In this work, we propose to use graph convolutional networks for text classification. We build a single text graph for a corpus based on word co-occurrence and document word relations, then learn a Text Graph Convolutional Network (Text GCN) for the corpus. Our Text GCN is initialized with one-hot representation for word and document, it then jointly learns the embeddings for both words and documents, as supervised by the known class labels for documents. Our experimental results on multiple benchmark datasets demonstrate that a vanilla Text GCN without any external word embeddings or knowledge outperforms state-of-the-art methods for text classification. On the other hand, Text GCN also learns predictive word and document embeddings. In addition, experimental results show that the improvement of Text GCN over state-of-the-art comparison methods become more prominent as we lower the percentage of training data, suggesting the robustness of Text GCN to less training data in text classification.
  
  主要贡献：构建基于文本和词的异构图，在GCN上进行半监督文本分类，包含文本节点和词节点，document-word边的权重是TF-IDF，word-word边的权重是PMI，即词的共现频率。 
  
文本分类是自然语言处理中的一个重要而经典的问题。已经有很多研究将卷积神经网络 (规则网格上的卷积，例如序列) 应用于分类。然而，只有个别研究探索了将更灵活的图卷积神经网络(在非网格上卷积，如任意图)应用到该任务上。在这项工作中，本文提出使用图卷积网络（GCN）来进行文本分类。基于词的共现关系和文档词的关系，本文为整个语料库构建单个文本图，然后学习用于语料库的文本图卷积网络(text GCN)。本文的text-GCN首先对词语和文本使用one-hot编码进行初始化，然后在已知文档类标签的监督下联合学习单词和文本的嵌入（通过GCN网络传播）。我们的模型在多个基准数据集上的实验结果表明，一个没有任何外部词或知识嵌入的普通text-GCN在性能上优于最先进的文本分类方法。另一方面，Text -GCN也学习词的预测和文档嵌入。实验结果表明，当降低训练数据的百分比时，文本GCN相对于现有比较方法的改进更加显著，说明在文本分类中，文本GCN对较少的训练数据具有鲁棒性。
  
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure7.png)  

</p></blockquote></details>



#### 2018

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d18-1380">Multi-grained attention network for aspect-level sentiment classification</a> --- MGAN --- by<i> Feifan Fan, Yansong Feng, Dongyan Zhao
</a>(<a href="https://github.com/songyouwei/ABSA-PyTorch">Github</a>)</summary><blockquote><p align="justify">
We propose a novel multi-grained attention network (MGAN) model for aspect level sentiment classification. Existing approaches mostly adopt coarse-grained attention mechanism, which may bring information loss if the aspect has multiple words or larger context. We propose a fine-grained attention mechanism, which can capture the word-level interaction between aspect and context. And then we leverage the fine-grained and coarse-grained attention mechanisms to compose the MGAN framework. Moreover, unlike previous works which train each aspect with its context separately, we design an aspect alignment loss to depict the aspect-level interactions among the aspects that have the same context. We evaluate the proposed approach on three datasets: laptop and restaurant are from SemEval 2014, and the last one is a twitter dataset. Experimental results show that the multi-grained attention network consistently outperforms the state-of-the-art methods on all three datasets. We also conduct experiments to evaluate the effectiveness of aspect alignment loss, which indicates the aspect-level interactions can bring extra useful information and further improve the performance.
  
  主要贡献：多粒度注意力网络，结合粗粒度和细粒度注意力来捕捉aspect和上下文在词级别上的交互；aspect对齐损失来描述拥有共同上下文的aspect之间的aspect级别上的相互影响。
  
本文提出了一种新颖的多粒度注意力网络模型，用于方面级（aspect-level）情感分类。现有的方法多采用粗粒度注意机制，如果有多个词或较大的上下文，可能会造成信息丢失。因此作者提出了一种精细的注意力机制，可以捕捉到方面和上下文之间的字级交互。然后利用细粒度和粗粒度的注意机制来组成MGAN框架。此外，与之前用上下文分别训练每个方面的工作不同，作者设计了一个方面级对齐损失来描述具有相同上下文的方面之间的方面级交互。作者在三个数据集上评估提出的方法:SemEval 2014包含笔记本销售评价和餐厅评价，以及 twitter数据集。实验结果表明，在这三个数据集上，多粒度注意力网络的性能始终优于现有的方法。本文还进行了实验来评估方面对齐丢失的有效性，表明方面级交互可以带来额外的有用信息，并进一步提高性能。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure8.png)  
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d18-1350">Investigating capsule networks with dynamic routing for text classification</a> --- TextCapsule --- by<i> Min Yang, Wei Zhao, Jianbo Ye, Zeyang Lei, Zhou Zhao, Soufei Zhang
</a>(<a href="https://github.com/andyweizhao/capsule_text_classification">Github</a>)</summary><blockquote><p align="justify">
In this study, we explore capsule networks with dynamic routing for text classification. We propose three strategies to stabilize the dynamic routing process to alleviate the disturbance of some noise capsules which may contain “background” information or have not been successfully trained. A series of experiments are conducted with capsule networks on six text classification benchmarks. Capsule networks achieve state of the art on 4 out of 6 datasets, which shows the effectiveness of capsule networks for text classification. We additionally show that capsule networks exhibit significant improvement when transfer single-label to multi-label text classification over strong baseline methods. To the best of our knowledge, this is the first work that capsule networks have been empirically investigated for text modeling.
  
在这项研究中，本文探索了用于文本分类，具有动态路由的胶囊网络。本文提出了三种策略来稳定动态路由的过程，以减轻某些可能包含“背景”信息，或尚未成功训练的噪声胶囊的影响。作者在六个文本分类基准数据集上对胶囊网络进行了一系列实验。 胶囊网络在6个数据集中的4个上达到了SOTA效果，这表明了胶囊网络在文本分类任务中的有效性。 本文还展示了当通过强基线方法将单标签文本分类转换为多标签文本分类时，胶囊网络表现出显着的性能提升。 据作者所知，这项工作是第一次经过经验研究将胶囊网络用于文本建模任务。
  
    ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure9.png)  
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.24963/ijcai.2018/584">Constructing narrative event evolutionary graph for script event prediction</a> --- SGNN ---  by<i> Zhongyang Li, Xiao Ding, Ting Liu
</a>(<a href="https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018">Github</a>)</summary><blockquote><p align="justify">
Script event prediction requires a model to predict the subsequent event given an existing event context. Previous models based on event pairs or event chains cannot make full use of dense event connections, which may limit their capability of event prediction. To remedy this, we propose constructing an event graph to better utilize the event network information for script event prediction. In particular, we first extract narrative event chains from large quantities of news corpus, and then construct a narrative event evolutionary graph (NEEG) based on the extracted chains. NEEG can be seen as a knowledge base that describes event evolutionary principles and patterns. To solve the inference problem on NEEG, we present a scaled graph neural network (SGNN) to model event interactions and learn better event representations. Instead of computing the representations on the whole graph, SGNN processes only the concerned nodes each time, which makes our model feasible to large-scale graphs. By comparing the similarity between input context event representations and candidate event representations, we can choose the most reasonable subsequent event. Experimental results on widely used New York Times corpus demonstrate that our model significantly outperforms state-of-the-art baseline methods, by using standard multiple choice narrative cloze evaluation.

脚本事件预测需要模型在已知现有事件的上下文信息的情况下，预测对应的上下文事件。以往的模型主要基于事件对或事件链，这种模式无法充分利用事件之间的密集连接，在某种程度上这会限制模型对事件的预测能力。为了解决这个问题，本文提出构造一个事件图来更好地利用事件网络信息进行脚本事件预测。特别是，首先从大量新闻语料库中提取叙事事件链，然后根据提取的事件链来构建一个叙事事件进化图（narrative event evolutionary graph ，NEEG）。 NEEG可以看作是描述事件进化原理和模式的知识库。为了解决NEEG上的推理问题，本文提出了可放缩图神经网络（SGNN）来对事件之间的交互进行建模，并更好地学习事件的潜在表示。 SGNN每次都只处理相关的节点，而不是在整个图的基础上计算特征信息，这使本文提出的模型能在大规模图上进行计算。通过比较输入上下文事件的特征表示与候选事件特征表示之间的相似性，模型可以选择最合理的后续事件。在广泛使用的《纽约时报》语料库上的实验结果表明，通过使用标准的多选叙述性完形填空评估，本文的模型明显优于最新的基准方法。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure10.png)  
    
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/C18-1330/">SGM: sequence generation model for multi-label classification</a> --- SGM ---  by<i> Pengcheng Yang, Xu Sun, Wei Li, Shuming Ma, Wei Wu, Houfeng Wang
</a>(<a href="https://github.com/lancopku/SGM">Github</a>)</summary><blockquote><p align="justify">
Multi-label classification is an important yet challenging task in natural language processing. It is more complex than single-label classification in that the labels tend to be correlated. Existing methods tend to ignore the correlations between labels. Besides, different parts of the text can contribute differently for predicting different labels, which is not considered by existing models. In this paper, we propose to view the multi-label classification task as a sequence generation problem, and apply a sequence generation model with a novel decoder structure to solve it. Extensive experimental results show that our proposed methods outperform previous work by a substantial margin. Further analysis of experimental results demonstrates that the proposed methods not only capture the correlations between labels, but also select the most informative words automatically when predicting different labels.
  
多标签分类是NLP任务中一项重要而又具有挑战性的任务。相较于单标签分类，由于多个标签之间趋于相关，因而更加复杂。现有方法倾向于忽略标签之间的相关性。此外，文本的不同部分对于预测不同的标签可能有不同的贡献，然而现有模型并未考虑这一点。在本文中，作者提出将多标签分类任务视为序列生成问题，并用具有新颖解码器结构的序列生成模型来解决该问题。大量的实验结果表明，本文提出的方法在很大程度上优于以前的工作。 对实验结果的进一步分析表明，本文的方法不仅可以捕获标签之间的相关性，而且可以在预测不同标签时自动选择具有最多信息的单词。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure11.png)  
  
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1216/">Joint embedding of words and labelsfor text classification</a> --- LEAM ---  by<i> Guoyin Wang, Chunyuan Li, Wenlin Wang, Yizhe Zhang, Dinghan Shen, Xinyuan Zhang, Ricardo Henao, Lawrence Carin
</a>(<a href="https://github.com/guoyinwang/LEAM">Github</a>)</summary><blockquote><p align="justify">
Word embeddings are effective intermediate representations for capturing semantic regularities between words, when learning the representations of text sequences. We propose to view text classification as a label-word joint embedding problem: each label is embedded in the same space with the word vectors. We introduce an attention framework that measures the compatibility of embeddings between text sequences and labels. The attention is learned on a training set of labeled samples to ensure that, given a text sequence, the relevant words are weighted higher than the irrelevant ones. Our method maintains the interpretability of word embeddings, and enjoys a built-in ability to leverage alternative sources of information, in addition to input text sequences. Extensive results on the several large text datasets show that the proposed framework outperforms the state-of-the-art methods by a large margin, in terms of both accuracy and speed.
  
在对文本序列进行表征学习时，单词嵌入是捕获单词之间语义规律的有效中间表示。在本文中，作者提出将文本分类视为标签与单词的联合嵌入问题：每个标签与单词向量一起嵌入同一向量空间。 本文引入了一个注意力框架，该框架可测量文本序列和标签之间嵌入的兼容性。该注意力框架在带有标签标记的数据集上进行训练，以确保在给定文本序列的情况下，相关单词的权重高于不相关单词的权重。 本文的方法保持了单词嵌入的可解释性，并且还具有内置的能力来利用替代信息源，来作为输入文本序列信息的补充。 在几个大型文本数据集上的大量实验结果表明，本文所提出的框架在准确性和速度上都大大优于目前的SOTA方法。
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure12.png)  
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1031/">Universal language model fine-tuning for text classification</a> --- ULMFiT ---  by<i> Jeremy Howard, Sebastian Ruder
</a>(<a href="http://nlp.fast.ai/category/classification.html">Github</a>)</summary><blockquote><p align="justify">
Inductive transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific modifications and training from scratch. We propose Universal Language Model Fine-tuning (ULMFiT), an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model. Our method significantly outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets. Furthermore, with only 100 labeled examples, it matches the performance of training from scratch on 100 times more data. We open-source our pretrained models and code.
  
 归纳迁移学习在CV领域大放异彩，但并未广泛应用于NLP领域，NLP领域的现有方法仍然需要针对特定任务进行模型修改并从头开始训练。 因此本文提出了通用语言模型微调（Universal Language Model Fine-tuning ，ULMFiT），一种可以应用于NLP中任何任务的高效率迁移学习方法，并介绍了对语言模型进行微调的关键技术。 本文的方法在六个文本分类任务上的性能明显优于SOTA技术，在大多数数据集上的错误率降低了18-24％。 此外，在只有100个带标签样本的情况下，本文的方法能于在100倍数据上从头训练的性能相匹配。相关的预先训练的模型和代码已开源。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure13.png)  
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://dl.acm.org/doi/10.1145/3178876.3186005">Large-scale hierarchical text classification withrecursively regularized deep graph-cnn</a> --- DGCNN --- by<i> Hao Peng, Jianxin Li, Yu He, Yaopeng Liu, Mengjiao Bao, Lihong Wang, Yangqiu Song, Qiang Yang
</a>(<a href="https://github.com/HKUST-KnowComp/DeepGraphCNNforTexts">Github</a>)</summary><blockquote><p align="justify">
Text classification to a hierarchical taxonomy of topics is a common and practical problem. Traditional approaches simply use bag-of-words and have achieved good results. However, when there are a lot of labels with different topical granularities, bag-of-words representation may not be enough. Deep learning models have been proven to be effective to automatically learn different levels of representations for image data. It is interesting to study what is the best way to represent texts. In this paper, we propose a graph-CNN based deep learning model to first convert texts to graph-of-words, and then use graph convolution operations to convolve the word graph. Graph-of-words representation of texts has the advantage of capturing non-consecutive and long-distance semantics. CNN models have the advantage of learning different level of semantics. To further leverage the hierarchy of labels, we regularize the deep architecture with the dependency among labels. Our results on both RCV1 and NYTimes datasets show that we can significantly improve large-scale hierarchical text classification over traditional hierarchical text classification and existing deep models.
  
将文本分类按主题进行层次分类是一个常见且实际的问题。传统方法仅使用单词袋（bag-of-words）并取得了良好的效果。但是，当有许多具有不同的主题粒度标签时，词袋的表征能力可能不足。鉴于深度学习模型已被证明可以有效地自动学习图像数据的不同表示形式，因此值得研究哪种方法是文本表征学习的最佳方法。在本文中，作者提出了一种基于graph-CNN的深度学习模型，该模型首先将文本转换为单词图，然后使用图卷积运算对词图进行卷积。将文本表示为图具有捕获非连续和长距离语义信息的优势。 CNN模型的优势在于可以学习不同级别的语义信息。为了进一步利用标签的层次结构，本文使用标签之间的依赖性来对深度网络结构进行正则化。在RCV1和NYTimes数据集上的结果表明，与传统的分层文本分类和现有的深度模型相比，本文的方法在大规模的分层文本分类任务上有显著提升。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/n18-1202">Deep contextualized word rep-resentations</a> --- ELMo --- by<i> Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer
</a>(<a href="https://github.com/flairNLP/flair">Github</a>)</summary><blockquote><p align="justify">
We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.
  
本文介绍了一种新型的深层上下文词表示形式，该模型既可以对以下信息进行建模（1）单词使用方法的复杂特征（例如语法和语义） （2）这些用法在语言上下文之间的变化方式（即建模多义性）。 本文的词向量是深度双向语言模型（biLM）内部状态的学习函数，双向语言模型已在大型文本语料库上进行了预训练。实验证明了可以很容易地将这些表示形式添加到现有模型中，并在六个具有挑战性的NLP问题上（包括问题回答，文本蕴含和情感分析）显著改善目前的SOTA。 本文还进行了一项分析，该分析表明探索预训练网络的深层内部信息至关重要，这有助于下游模型混合不同类型的半监督信号。
</p></blockquote></details>


#### 2017
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D17-1047/">Recurrent Attention Network on Memory for Aspect Sentiment Analysis</a> --- RAM --- by<i> Peng Chen, Zhongqian Sun, Lidong Bing, Wei Yang
</a>(<a href="https://github.com/songyouwei/ABSA-PyTorch">Github</a>)</summary><blockquote><p align="justify">
We propose a novel framework based on neural networks to identify the sentiment of opinion targets in a comment/review. Our framework adopts multiple-attention mechanism to capture sentiment features separated by a long distance, so that it is more robust against irrelevant information. The results of multiple attentions are non-linearly combined with a recurrent neural network, which strengthens the expressive power of our model for handling more complications. The weighted-memory mechanism not only helps us avoid the labor-intensive feature engineering work, but also provides a tailor-made memory for different opinion targets of a sentence. We examine the merit of our model on four datasets: two are from SemEval2014, i.e. reviews of restaurants and laptops; a twitter dataset, for testing its performance on social media data; and a Chinese news comment dataset, for testing its language sensitivity. The experimental results show that our model consistently outperforms the state-of-the-art methods on different types of data.
  
本文提出了一种基于神经网络的新框架，以识别评论中意见目标的情绪。本文的框架采用多注意机制来捕获相距较远的情感特征，因此对于不相关的信息鲁棒性更高。多重注意力的结果与循环神经网络（RNN）进行非线性组合，从而增强了模型在处理更多并发情况时的表达能力。加权内存机制不仅避免了工作量大的特征工程工作，而且还为句子的不同意见目标提供了对应的记忆特征。在四个数据集上的实验验证了模型的优点：两个来自SemEval2014，该数据集包含了例如餐馆和笔记本电脑的评论信息; 一个Twitter数据集，用于测试其在社交媒体数据上的效果；以及一个中文新闻评论数据集，用于测试其语言敏感性。实验结果表明，本文的模型在不同类型的数据上始终优于SOTA方法。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure14.png) 
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d17-1169">Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm</a> --- DeepMoji --- by<i> Bjarke Felbo, Alan Mislove, Anders Søgaard, Iyad Rahwan, Sune Lehmann
</a>(<a href="https://github.com/bfelbo/DeepMoji">Github</a>)</summary><blockquote><p align="justify">
NLP tasks are often limited by scarcity of manually annotated data. In social media sentiment analysis and related tasks, researchers have therefore used binarized emoticons and specific hashtags as forms of distant supervision. Our paper shows that by extending the distant supervision to a more diverse set of noisy labels, the models can learn richer representations. Through emoji prediction on a dataset of 1246 million tweets containing one of 64 common emojis we obtain state-of-the-art performance on 8 benchmark datasets within emotion, sentiment and sarcasm detection using a single pretrained model. Our analyses confirm that the diversity of our emotional labels yield a performance improvement over previous distant supervision approaches.
  
NLP任务通常受到缺乏人工标注数据的限制。 因此，在社交媒体情绪分析和相关任务中，研究人员已使用二值化表情符号和特定的主题标签作为远程监督的形式。 本文的研究表明，通过将远程监督扩展到更多种类的嘈杂标签，可以让模型学习到更丰富的特征表示。 通过对包含64种常见表情符号之一的12.46亿条推文数据集进行表情符号预测，可以使用单个预训练模型在情绪，情感和嘲讽检测的8个基准数据集上获得最优效果。 本文的分析进一步证明，与以前的远程监督方法相比，情感标签的多样性可以显著提高模型效果。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure15.png) 
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.ijcai.org/Proceedings/2017/568">Interactive attention networks for aspect-level sentiment classification</a> --- IAN --- by<i> Dehong Ma, Sujian Li, Xiaodong Zhang, Houfeng Wang
</a>(<a href="https://github.com/songyouwei/ABSA-PyTorch">Github</a>)</summary><blockquote><p align="justify">
Aspect-level sentiment classification aims at identifying the sentiment polarity of specific target in its context. Previous approaches have realized the importance of targets in sentiment classification and developed various methods with the goal of precisely modeling thier contexts via generating target-specific representations. However, these studies always ignore the separate modeling of targets. In this paper, we argue that both targets and contexts deserve special treatment and need to be learned their own representations via interactive learning. Then, we propose the interactive attention networks (IAN) to interactively learn attentions in the contexts and targets, and generate the representations for targets and contexts separately. With this design, the IAN model can well represent a target and its collocative context, which is helpful to sentiment classification. Experimental results on SemEval 2014 Datasets demonstrate the effectiveness of our model.
  
方面级别（aspect-level）的情感分类旨在识别特定目标在其上下文中的情感极性。 先前的方法已经意识到情感目标在情感分类中的重要性，并开发了各种方法，目的是通过生成特定于目标的表示来对上下文进行精确建模。 但是，这些研究始终忽略了目标的单独建模。 在本文中，作者认为目标和上下文都应受到特殊对待，需要通过交互式学习来学习它们自己的特征表示。 因此，作者提出了交互式注意力网络（interactive attention networks , IAN），以交互方式学习上下文和目标中的注意力信息，并分别生成目标和上下文的特征表示。 通过这种设计，IAN模型可以很好地表示目标及其搭配上下文，这有助于情感分类。 在SemEval 2014数据集上的实验结果证明了本文模型的有效性。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure16.png) 
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/P17-1052">Deep pyramid convolutional neural networks for text categorization</a> --- DPCNN --- by<i> Rie Johnson, Tong Zhang
</a>(<a href="https://github.com/Cheneng/DPCNN">Github</a>)</summary><blockquote><p align="justify">
This paper proposes a low-complexity word-level deep convolutional neural network (CNN) architecture for text categorization that can efficiently represent long-range associations in text. In the literature, several deep and complex neural networks have been proposed for this task, assuming availability of relatively large amounts of training data. However, the associated computational complexity increases as the networks go deeper, which poses serious challenges in practical applications. Moreover, it was shown recently that shallow word-level CNNs are more accurate and much faster than the state-of-the-art very deep nets such as character-level CNNs even in the setting of large training data. Motivated by these findings, we carefully studied deepening of word-level CNNs to capture global representations of text, and found a simple network architecture with which the best accuracy can be obtained by increasing the network depth without increasing computational cost by much. We call it deep pyramid CNN. The proposed model with 15 weight layers outperforms the previous best models on six benchmark datasets for sentiment classification and topic categorization.
  
本文提出了一种用于文本分类的低复杂度的词语级深度卷积神经网络（CNN）架构，该架构可以有效地对文本中的远程关联进行建模。在以往的研究中，已经有多种复杂的深度神经网络已经被用于该任务，当然前提是可获得相对大量的训练数据。然而随着网络的深入，相关的计算复杂性也会增加，这对网络的实际应用提出了严峻的挑战。此外，最近的研究表明，即使在设置大量训练数据的情况下，较浅的单词级CNN也比诸如字符级CNN之类的深度网络更准确，且速度更快。受这些发现的启发，本文仔细研究了单词级CNN的深度化以捕获文本的整体表示，并找到了一种简单的网络体系结构，在该体系结构下，可以通过增加网络深度来获得最佳精度，且不会大量增加计算成本。相应的模型被称为深度金字塔CNN（pyramid-CNN）。在情感分类和主题分类任务的六个基准数据集上，本文提出的具有15个权重层的模型优于先前的SOTA模型。
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure17.png) 
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://openreview.net/forum?id=rJbbOLcex">Topicrnn: A recurrent neural network with long-range semantic dependency</a> --- TopicRNN ---  by<i> Adji B. Dieng, Chong Wang, Jianfeng Gao, John Paisley
</a>(<a href="https://github.com/dangitstam/topic-rnn">Github</a>)</summary><blockquote><p align="justify">
In this paper, we propose TopicRNN, a recurrent neural network (RNN)-based language model designed to directly capture the global semantic meaning relating words in a document via latent topics. Because of their sequential nature, RNNs are good at capturing the local structure of a word sequence – both semantic and syntactic – but might face difficulty remembering long-range dependencies. Intuitively, these long-range dependencies are of semantic nature. In contrast, latent topic models are able to capture the global underlying semantic structure of a document but do not account for word ordering. The proposed TopicRNN model integrates the merits of RNNs and latent topic models: it captures local (syntactic) dependencies using an RNN and global (semantic) dependencies using latent topics. Unlike previous work on contextual RNN language modeling, our model is learned end-to-end. Empirical results on word prediction show that TopicRNN outperforms existing contextual RNN baselines. In addition, TopicRNN can be used as an unsupervised feature extractor for documents. We do this for sentiment analysis on the IMDB movie review dataset and report an error rate of 6.28%. This is comparable to the state-of-the-art 5.91% resulting from a semi-supervised approach. Finally, TopicRNN also yields sensible topics, making it a useful alternative to document models such as latent Dirichlet allocation.
  
本文提出了TopicRNN，这是一个基于循环神经网络（RNN）的语言模型，旨在通过潜在主题直接捕获文档中与单词相关的全局语义。由于RNN的顺序结构特点，其擅长捕获单词序列的局部结构（包括语义和句法），但可能难以记住长期依赖关系。直观上，这些远程依赖关系具有语义性质。相反，潜在主题模型（latent topic models）能够捕获文档的全局基础语义结构，但不考虑单词顺序。因此本文提出的TopicRNN模型整合了RNN和潜在主题模型的优点：它使用RNN捕获局部（语法）依赖关系，并使用潜在主题捕获全局（语义）依赖关系。与先前基于上下文RNN的语言建模工作不同，本文的模型是端到端模型。单词预测的经验性结果表明，TopicRNN优于现有的上下文RNN基线。此外，TopicRNN可以用作文档的无监督特征提取器。本文以情感分析任务为例在IMDB电影评论数据集上测试，错误率为6.28％，这可与半监督方法产生的最优效果5.91％相媲美。最后，TopicRNN还提出了有探索意义的主题，使其成为诸如潜在Dirichlet分配之类的文档模型的有用替代方法。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure18.png) 
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://openreview.net/forum?id=r1X3g2_xl">Adversarial training methods for semi-supervised text classification</a> --- Miyato et al. ---  by<i> Takeru Miyato, Andrew M. Dai, Ian Goodfellow
</a>(<a href="https://github.com/tensorflow/models/tree/master/adversarial_text">Github</a>)</summary><blockquote><p align="justify">
Adversarial training provides a means of regularizing supervised learning algorithms while virtual adversarial training is able to extend supervised learning algorithms to the semi-supervised setting. However, both methods require making small perturbations to numerous entries of the input vector, which is inappropriate for sparse high-dimensional inputs such as one-hot word representations. We extend adversarial and virtual adversarial training to the text domain by applying perturbations to the word embeddings in a recurrent neural network rather than to the original input itself. The proposed method achieves state of the art results on multiple benchmark semi-supervised and purely supervised tasks. We provide visualizations and analysis showing that the learned word embeddings have improved in quality and that while training, the model is less prone to overfitting.
  
 对抗训练提供了一种对有监督学习算法进行正则化的方法，而虚拟对抗训练能够将监督学习算法扩展到半监督条件下。 但是，这两种方法都需要对输入向量的众多条目进行较小的扰动，这对于稀疏的高维输入（例如：独一热单词编码）是不合适的。 通过对循环神经网络中的单词嵌入（而不是原始输入本身）进行扰动，本文将对抗性和虚拟对抗性训练扩展到文本域。 本文所提出的方法在多个基准半监督任务和纯监督任务上均达到了SOTA效果。可视化和分析结果表明，学习的单词嵌入的质量有所提高，并且在模型在训练过程中更加不容易过拟合。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure19.png) 
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/e17-2068">Bag of tricks for efficient text classification</a> --- FastText ---  by<i> Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov
</a>(<a href="https://github.com/SeanLee97/short-text-classification">Github</a>)</summary><blockquote><p align="justify">
This paper explores a simple and efficient baseline for text classification. Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation. We can train fastText on more than one billion words in less than ten minutes using a standard multicore CPU, and classify half a million sentences among 312K classes in less than a minute.
  
本文探讨了一种简单有效的文本分类基准。实验表明，本文的快速文本分类器fastText在准确性方面可以与深度学习分类器相提并论，而训练和预测速度要快多个数量级。 可以使用标准的多核CPU在不到十分钟的时间内在超过十亿个单词的数据集上训练fastText，并在一分钟之内对属于312K个类别的50万个句子进行分类。
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure20.png) 
</p></blockquote></details>



#### 2016

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d16-1053">Long short-term memory-networks for machine reading</a> --- LSTMN ---  by<i> Jianpeng Cheng, Li Dong, Mirella Lapata
</a>(<a href="https://github.com/JRC1995/Abstractive-Summarization">Github</a>)</summary><blockquote><p align="justify">
In this paper we address the question of how to render sequence-level networks better at handling structured input. We propose a machine reading simulator which processes text incrementally from left to right and performs shallow reasoning with memory and attention. The reader extends the Long Short-Term Memory architecture with a memory network in place of a single memory cell. This enables adaptive memory usage during recurrence with neural attention, offering a way to weakly induce relations among tokens. The system is initially designed to process a single sequence but we also demonstrate how to integrate it with an encoder-decoder architecture. Experiments on language modeling, sentiment analysis, and natural language inference show that our model matches or outperforms the state of the art.
  
 在本文中，作者解决了如何在处理结构化输入时更好地呈现序列网络的问题。 本文提出了一种机器阅读模拟器，该模拟器可以从左到右递增地处理文本，并通过记忆和注意力进行浅层推理。 阅读器使用存储网络代替单个存储单元来对LSTM结构进行扩展。 这可以在神经注意力循环计算时启用自适应内存使用，从而提供一种弱化token之间关系的方法。 该系统最初设计为处理单个序列，但本文还将演示如何将其与编码器-解码器体系结构集成。 在语言建模，情感分析和自然语言推理任务上的实验表明，本文的模型与SOTA相媲美，甚至优于目前的SOTA。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure21.png) 
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.ijcai.org/Abstract/16/408">Recurrent neural network for text classification with multi-task learning</a> --- Multi-Task --- by<i> Pengfei Liu, Xipeng Qiu, Xuanjing Huang
</a>(<a href="https://github.com/baixl/text_classification">Github</a>)</summary><blockquote><p align="justify">
Neural network based methods have obtained great progress on a variety of natural language processing tasks. However, in most previous works, the models are learned based on single-task supervised objectives, which often suffer from insufficient training data. In this paper, we use the multi-task learning framework to jointly learn across multiple related tasks. Based on recurrent neural network, we propose three different mechanisms of sharing information to model text with task-specific and shared layers. The entire network is trained jointly on all these tasks. Experiments on four benchmark text classification tasks show that our proposed models can improve the performance of a task with the help of other related tasks.
  
 基于神经网络的方法已经在各种自然语言处理任务上取得了长足的进步。然而在以往的大多数工作中，都是基于有监督的单任务目标进行模型训练，而这些目标通常会受训练数据不足的困扰。 在本文中，作者使用多任务学习框架来共同学习多个相关任务（相对于多个任务的训练数据可以共享）。本文提出了三种不同的基于递归神经网络的信息共享机制，以针对特定任务和共享层对文本进行建模。 整个网络在这些任务上进行联合训练。在四个基准文本分类任务的实验表明，模型在某一任务下的性能可以在其他任务的帮助下得到提升。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure22.png) 
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/n16-1174">Hierarchical attention networks for document classification</a> --- HAN --- by<i> Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy
</a>(<a href="https://github.com/richliao/textClassifier">Github</a>)</summary><blockquote><p align="justify">
We propose a hierarchical attention networkfor document classification.  Our model hastwo distinctive characteristics: (i) it has a hier-archical structure that mirrors the hierarchicalstructure of documents; (ii) it has two levelsof attention mechanisms applied at the word-and sentence-level, enabling it to attend dif-ferentially to more and less important con-tent when constructing the document repre-sentation. Experiments conducted on six largescale text classification tasks demonstrate thatthe proposed architecture outperform previousmethods by a substantial margin. Visualiza-tion of the attention layers illustrates that themodel selects qualitatively informative wordsand sentences.
  
本文提出了一种用于文档分类的层次注意力网络。该模型具有两个鲜明的特征：（1）具有分层模型结构，能反应对应层次的文档结构； （2）它在单词和句子级别上应用了两个级别的注意机制，使它在构建文档表征时可以有区别地对待或多或少的重要内容。在六个大型文本分类任务上进行的实验表明，本文所提出的分层体系结构在很大程度上优于先前的方法。此外，注意力层的可视化说明该模型定性地选择了富有主要信息的词和句子。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure23.png) 
</p></blockquote></details>


#### 2015

 <details/>
<summary/>
  <a href="http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification">Character-level convolutional networks for text classification</a> --- CharCNN --- by<i> Xiang Zhang, Junbo Zhao, Yann LeCun
</a>(<a href="https://github.com/mhjabreel/CharCNN">Github</a>)</summary><blockquote><p align="justify">
This article offers an empirical exploration on the use of character-level convolutional networks (ConvNets) for text classification. We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks.
  
本文提出了通过字符级卷积网络（ConvNets）进行文本分类的实证研究。 本文构建了几个大型数据集，以证明字符级卷积网络可以达到SOTA结果或者得到具有竞争力的结果。 可以与传统模型（例如bag of words，n-grams 及其 TFIDF变体）以及深度学习模型（例如基于单词的ConvNets和RNN）进行比较。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure24.png) 
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.3115/v1/p15-1150">Improved semantic representations from tree-structured long short-term memory networks</a> --- Tree-LSTM ---  by<i> Kai Sheng Tai, Richard Socher, Christopher D. Manning
</a>(<a href="https://github.com/stanfordnlp/treelstm">Github</a>)</summary><blockquote><p align="justify">
Because  of  their  superior  ability  to  pre-serve   sequence   information   over   time,Long  Short-Term  Memory  (LSTM)  net-works,   a  type  of  recurrent  neural  net-work with a more complex computationalunit, have obtained strong results on a va-riety  of  sequence  modeling  tasks.Theonly underlying LSTM structure that hasbeen  explored  so  far  is  a  linear  chain.However,  natural  language  exhibits  syn-tactic properties that would naturally com-bine words to phrases.  We introduce theTree-LSTM, a generalization of LSTMs totree-structured network topologies.  Tree-LSTMs  outperform  all  existing  systemsand strong LSTM baselines on two tasks:predicting the semantic relatedness of twosentences  (SemEval  2014,  Task  1)  andsentiment  classification  (Stanford  Senti-ment Treebank).
  
由于具有较强的序列长期依赖保存能力，具有更复杂的计算单元的长短时记忆网络（LSTM）在各种序列建模任务上都取得了出色的结果。然而，现有研究探索过的唯一底层LSTM结构是线性链。由于自然语言具有句法属性， 因此可以自然地将单词与短语结合起来。 本文提出了Tree-LSTM，它是LSTM在树形拓扑网络结构上的扩展。 Tree-LSTM在下面两个任务上的表现优于所有现有模型以及强大的LSTM基准方法：预测两个句子的语义相关性（SemEval 2014，任务1）和情感分类（Stanford情感树库）。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure25.png) 
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.3115/v1/p15-1162">Deep unordered composition rivals syntactic methods for text classification</a> --- DAN ---  by<i> Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber, Hal Daumé III
</a>(<a href="https://github.com/miyyer/dan">Github</a>)</summary><blockquote><p align="justify">
Many  existing  deep  learning  models  fornatural language processing tasks focus onlearning thecompositionalityof their in-puts, which requires many expensive com-putations. We present a simple deep neuralnetwork that competes with and, in somecases,  outperforms  such  models  on  sen-timent  analysis  and  factoid  question  an-swering tasks while taking only a fractionof the training time.  While our model issyntactically-ignorant, we show significantimprovements over previous bag-of-wordsmodels by deepening our network and ap-plying a novel variant of dropout.  More-over, our model performs better than syn-tactic models on datasets with high syn-tactic variance.  We show that our modelmakes similar errors to syntactically-awaremodels, indicating that for the tasks we con-sider, nonlinearly transforming the input ismore important than tailoring a network toincorporate word order and syntax.
  
 现有的许多用于自然语言处理任务的深度学习模型都专注于学习不同输入的语义合成性， 然而这需要许多昂贵的计算。本文提出了一个简单的深度神经网络，它在情感分析和事实类问题解答任务上可以媲美，并且在某些情况下甚至胜过此类模型，并且只需要少部分训练事件。 尽管本文的模型对语法并不敏感， 但通过加深网络并使用一种新型的辍学变量，模型相较于以前的单词袋模型上表现出显著的改进。 此外，本文的模型在具有高句法差异的数据集上的表现要比句法模型更好。 实验表明，本文的模型与语法感知模型存在相似的错误，表明在本文所考虑的任务中，非线性转换输入比定制网络以合并单词顺序和语法更重要。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure26.png) 
</p></blockquote></details>


 <details/>
<summary/>
  <a href="http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745">Recurrent convolutional neural networks for text classification</a> --- TextRCNN ---  by<i> Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao
</a>(<a href="https://github.com/roomylee/rcnn-text-classification">Github</a>)</summary><blockquote><p align="justify">
Text classification is a foundational task in many NLP applications. Traditional text classifiers often rely on many human-designed features, such as dictionaries, knowledge bases and special tree kernels. In contrast to traditional methods, we introduce a recurrent convolutional neural network for text classification without human-designed features. In our model, we apply a recurrent structure to capture contextual information as far as possible when learning word representations, which may introduce considerably less noise compared to traditional window-based neural networks. We also employ a max-pooling layer that automatically judges which words play key roles in text classification to capture the key components in texts. We conduct experiments on four commonly used datasets. The experimental results show that the proposed method outperforms the state-of-the-art methods on several datasets, particularly on document-level datasets.
  
 文本分类是众多NLP应用中的一项基本任务。 传统的文本分类器通常依赖于许多人工设计的特征工程，例如字典，知识库和特殊的树形内核。 与传统方法相比，本文引入了循环卷积神经网络来进行文本分类，而无需手工设计的特征或方法。 在本文的模型中，当学习单词表示时，本文应用递归结构来尽可能地捕获上下文信息，相较于传统的基于窗口的神经网络，这种方法带来的噪声更少。 本文还采用了一个最大池化层，该层可以自动判断哪些单词在文本分类中起关键作用，以捕获文本中的关键组成部分。 本文在四个常用数据集进行了实验， 实验结果表明，本文所提出的模型在多个数据集上，特别是在文档级数据集上，优于最新方法。
</p></blockquote></details>



#### 2014
 <details/>
<summary/>
  <a href="http://proceedings.mlr.press/v32/le14.html">Distributed representations of sentences and documents</a> --- Paragraph-Vec ---  by<i> Quoc Le, Tomas Mikolov
</a>(<a href="https://github.com/inejc/paragraph-vectors">Github</a>)</summary><blockquote><p align="justify">
Many machine learning algorithms require the input to be represented as a fixed length feature vector. When it comes to texts, one of the most common representations is bag-of-words. Despite their popularity, bag-of-words models have two major weaknesses: they lose the ordering of the words and they also ignore semantics of the words. For example, "powerful," "strong" and "Paris" are equally distant. In this paper, we propose an unsupervised algorithm that learns vector representations of sentences and text documents. This algorithm represents each document by a dense vector which is trained to predict words in the document. Its construction gives our algorithm the potential to overcome the weaknesses of bag-of-words models. Empirical results show that our technique outperforms bag-of-words models as well as other techniques for text representations. Finally, we achieve new state-of-the-art results on several text classification and sentiment analysis tasks.
  
许多机器学习算法要求将输入表示为固定长度的特征向量。当涉及到文本时，词袋模型是最常见的表示形式之一。 尽管非常流行，但词袋模型有两个主要缺点：丢失了单词的顺序信息，并且也忽略了单词的语义含义。 例如在词袋中，“powerful”，“strong”和“Paris”的距离相等（但根据语义含义，显然“powerful”和”strong”的距离应该更近）。 因此在本文中，作者提出了一种无监督算法，用于学习句子和文本文档的向量表示。该算法用一个密集矢量来表示每个文档，经过训练后该向量可以预测文档中的单词。 它的构造使本文的算法可以克服单词袋模型的缺点。实验结果表明，本文的技术优于词袋模型以及其他用于文本表示的技术。 最后，本文在几个文本分类和情感分析任务上获得了SOTA效果。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure27.png)
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.3115/v1/p14-1062">A convolutional neural network for modelling sentences</a> --- DCNN ---  by<i> Nal Kalchbrenner, Edward Grefenstette, Phil Blunsom
</a>(<a href="https://github.com/kinimod23/ATS_Project">Github</a>)</summary><blockquote><p align="justify">
The ability to accurately represent sentences is central to language understanding. We describe a convolutional architecture dubbed the Dynamic Convolutional Neural Network (DCNN) that we adopt for the semantic modelling of sentences. The network uses Dynamic k-Max Pooling, a global pooling operation over linear sequences. The network handles input sentences of varying length and induces a feature graph over the sentence that is capable of explicitly capturing short and long-range relations. The network does not rely on a parse tree and is easily applicable to any language. We test the DCNN in four experiments: small scale binary and multi-class sentiment prediction, six-way question classification and Twitter sentiment prediction by distant supervision. The network achieves excellent performance in the first three tasks and a greater than 25% error reduction in the last task with respect to the strongest baseline.
  
准确的句子表征能力对于理解语言至关重要。 本文提出了一种被称为动态卷积神经网络（Dynamic Convolutional Neural Network , DCNN）的卷积体系结构，用来对句子的语义建模。 网络使用一种线性序列上的全局池化操作，称为动态k-Max池化。网络处理长度可变的输入句子，并通过句子来生成特征图， 该特征图能够显式捕获句中的短期和长期关系。 该网络不依赖于语法分析树，并且很容易适用于任何语言。 本文在四个实验中测试了DCNN：小规模的二类和多类别情感预测，六向问题分类以及通过远程监督的Twitter情感预测。 相对于目前效果最好的基准工作，本文的网络在前三个任务中标系出色的性能，并且在最后一个任务中将错误率减少了25％以上。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure28.png)
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D14-1181.pdf">Convolutional Neural Networks for Sentence Classification</a> --- TextCNN --- by<i> Yoon Kim
</a>(<a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras">Github</a>)</summary><blockquote><p align="justify">
We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.
  
本文研究了在卷积神经网络（CNN）上进行的一系列实验，这些卷积神经网络在针对句子级别分类任务的预训练单词向量的基础上进行了训练。 实验证明，几乎没有超参数调整和静态矢量的简单CNN在多个基准上均能实现出色的结果。 通过微调来学习针对特定任务的单词向量可进一步提高性能。 此外，本文还提出了对体系结构进行简单的修改，以让模型能同时使用针对特定任务的单词向量和静态向量。 本文讨论的CNN模型在7个任务中的4个上超过了现有的SOTA效果，其中包括情感分析和问题分类。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure29.png)
</p></blockquote></details>

#### 2013
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D13-1170/">Recursive deep models for semantic compositionality over a sentiment treebank</a> --- RNTN --- by<i> Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, Christopher Potts
</a>(<a href=" https://github.com/pondruska/DeepSentiment">Github</a>)</summary><blockquote><p align="justify">
Semantic word spaces have been very useful but cannot express the meaning of longer phrases in a principled way. Further progress towards understanding compositionality in tasks such as sentiment detection requires richer supervised training and evaluation resources and more powerful models of composition. To remedy this, we introduce a Sentiment Treebank. It includes fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and presents new challenges for sentiment composition-ality. To address them, we introduce the Recursive Neural Tensor Network. When trained on the new treebank, this model outperforms all previous methods on several metrics. It pushes the state of the art in single sentence positive/negative classification from 80% up to 85.4%. The accuracy of predicting fine-grained sentiment labels for all phrases reaches 80.7%, an improvement of 9.7% over bag of features baselines. Lastly, it is the only model that can accurately capture the effects of negation and its scope at various tree levels for both positive and negative phrases.
  
尽管语义词空间在语义表征方面效果很好，但却不能从原理上表达较长短语的含义。在诸如情绪检测等任务中的词语组合性理解方向的改进需要更丰富的监督训练和评估资源， 以及更强大的合成模型。为了解决这个问题，本文引入了一个情感树库。它在11,855个句子的语法分析树中包含215,154个短语的细粒度情感标签，并在情感组成性方面提出了新挑战。为了解决这些问题，本文引入了递归神经张量网络。在新的树库上进行训练后，该模型在多个评价指标上效果优于之前的所有方法。它使单句正/负分类的最新技术水平从80％上升到85.4％。预测所有短语的细粒度情感标签的准确性达到80.7％，相较于基准工作提高了9.7％。此外，它也是是唯一一个可以在正面和负面短语的各个树级别准确捕获消极影响及其范围的模型。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure30.png)
</p></blockquote></details>


#### 2012
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D12-1110/">Semantic compositionality through recursive matrix-vector spaces</a> --- MV-RNN --- by<i> Richard Socher, Brody Huval, Christopher D. Manning, Andrew Y. Ng
</a>(<a href="https://github.com/github-pengge/MV_RNN">Github</a>)</summary><blockquote><p align="justify">
Single-word vector space models have been very successful at learning lexical information. However, they cannot capture the compositional meaning of longer phrases, preventing them from a deeper understanding of language. We introduce a recursive neural network (RNN) model that learns compositional vector representations for phrases and sentences of arbitrary syntactic type and length. Our model assigns a vector and a matrix to every node in a parse tree: the vector captures the inherent meaning of the constituent, while the matrix captures how it changes the meaning of neighboring words or phrases. This matrix-vector RNN can learn the meaning of operators in propositional logic and natural language. The model obtains state of the art performance on three different experiments: predicting fine-grained sentiment distributions of adverb-adjective pairs; classifying sentiment labels of movie reviews and classifying semantic relationships such as cause-effect or topic-message between nouns using the syntactic path between them.
  
 基于单个词的向量空间模型在学习词汇信息方面非常成功。但是，它们无法捕获较长短语的组成含义，从而阻止了它们更深入理解地理解语言。本文介绍了一种循环神经网络（RNN）模型，该模型学习任意句法类型和长度的短语或句子的成分向量表示。本文的模型为解析树中的每个节点分配一个向量和一个矩阵：其中向量捕获成分的固有含义，而矩阵捕获其如何改变相邻单词或短语的含义。该矩阵-向量RNN可以学习命题逻辑和自然语言中算子的含义。该模型在三种不同的实验中均获得了SOTA效果：预测副词-形容词对的细粒度情绪分布；对电影评论的情感标签进行分类，并使用名词之间的句法路径对名词之间的因果关系或主题消息等语义关系进行分类。
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure31.png)
</p></blockquote></details>

#### 2011
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D11-1014/">Semi-supervised recursive autoencoders forpredicting sentiment distributions</a> --- RAE --- by<i> Richard Socher, Jeffrey Pennington, Eric H. Huang, Andrew Y. Ng, Christopher D. Manning
</a>(<a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras">Github</a>)</summary><blockquote><p align="justify">
We introduce a novel machine learning frame-work based on recursive autoencoders for sentence-level prediction of sentiment labeldistributions. Our method learns vector spacerepresentations for multi-word phrases. In sentiment prediction tasks these represen-tations outperform other state-of-the-art ap-proaches on commonly used datasets, such asmovie reviews, without using any pre-definedsentiment lexica or polarity shifting rules. Wealso  evaluate  the  model’s  ability to predict sentiment distributions on a new dataset basedon confessions from the experience project. The dataset consists of personal user storiesannotated with multiple labels which, whenaggregated, form a multinomial distributionthat captures emotional reactions. Our algorithm can more accurately predict distri-butions over such labels compared to severalcompetitive baselines.
  
 本文介绍了一种新颖地基于递归自动编码器机器学习框架，用于句子级地情感标签分布预测。 本文的方法学习多词短语的向量空间表示。在情感预测任务中，这些表示优于常规数据集（例如电影评论）上的其他最新方法，而无需使用任何预定义的情感词典或极性转换规则。 本文还将根据经验项目上的效果来评估模型在新数据集上预测情绪分布的能力。数据集由带有多个标签的个人用户故事组成，这些标签汇总后形成捕获情感反应的多项分布。 与其他几个具有竞争力的baseline相比，本文提出的算法可以更准确地预测此类标签的分布。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure32.png)
   
</p></blockquote></details>




# Shallow Learning Models
[:arrow_up:](#table-of-contents)

#### 2017
 <details/>
<summary/>
  <a href="http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree">Lightgbm: A highly efficient gradient boosting decision tree</a> --- LightGBM --- by<i> Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu
</a>(<a href="https://github.com/creatist/text_classify">Github</a>)</summary><blockquote><p align="justify">
Gradient Boosting Decision Tree (GBDT) is a popular machine learning algorithm, and has quite a few effective implementations such as XGBoost and pGBRT. Although many engineering optimizations have been adopted in these implementations, the efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large. A major reason is that for each feature, they need to scan all the data instances to estimate the information gain of all possible split points, which is very time consuming. To tackle this problem, we propose two novel techniques: \emph{Gradient-based One-Side Sampling} (GOSS) and \emph{Exclusive Feature Bundling} (EFB). With GOSS, we exclude a significant proportion of data instances with small gradients, and only use the rest to estimate the information gain. We prove that, since the data instances with larger gradients play a more important role in the computation of information gain, GOSS can obtain quite accurate estimation of the information gain with a much smaller data size. With EFB, we bundle mutually exclusive features (i.e., they rarely take nonzero values simultaneously), to reduce the number of features. We prove that finding the optimal bundling of exclusive features is NP-hard, but a greedy algorithm can achieve quite good approximation ratio (and thus can effectively reduce the number of features without hurting the accuracy of split point determination by much). We call our new GBDT implementation with GOSS and EFB \emph{LightGBM}. Our experiments on multiple public datasets show that, LightGBM speeds up the training process of conventional GBDT by up to over 20 times while achieving almost the same accuracy.
  
 梯度提升决策树（GBDT）是一种流行的机器学习算法，并且具有大量的高效实现（变体），例如XGBoost和pGBRT。尽管在这些实现中采用了许多工程优化方法，但是当特征维数较大且数据量较大时，模型效率和模型的可伸缩性仍不令人满意。一个主要原因是对于每个维度的特征，模型都需要扫描所有数据实例以估计所有可能的树分支点的信息增益， 这一步非常消耗事件。为了解决这个问题，本文提出了两种新颖的技术：基于梯度的单边采样 （GOSS）和Exclusive Feature Bundling （EFB）。使用GOSS，排除了大部分的小梯度数据样本，而仅使用其余部分来估计信息增益。作者证明，由于具有较大梯度的数据实例在信息增益的计算中起着更重要的作用，因此GOSS可以在数据量较小的情况下获得相当准确的信息增益估计。使用EFB来捆绑互斥的属性（即这两个属性很少同时采用非零值），以减少特征维度。实验证明，找到专有特征的最佳捆绑是NP难的，但是贪婪算法可以达到相当好的近似率（因此可以有效地减少特征维度，而不会严重损害树分支点的准确性）。本文将新的GBDT实施称为GOSS和EFB-LightGBM。在多个公共数据集上的实验表明，LightGBM将传统GBDT的训练过程加快了20倍以上，同时达到了几乎相同的准确性。
</p></blockquote></details>

#### 2016
 <details/>
<summary/>
  <a href="https://dl.acm.org/doi/10.1145/2939672.2939785">Xgboost: A scalable tree boosting system</a> --- XGBoost ---  by<i> Tianqi Chen, Carlos Guestrin
</a>(<a href="https://xgboost.readthedocs.io/en/latest">Github</a>)</summary><blockquote><p align="justify">
Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.
  
 提升树是一种高效且广泛使用的机器学习方法。本文描述了一种称为XGBoost的可扩展端到端提升树系统，该系统已被数据科学家广泛使用，以在许多机器学习竞赛中获得最优结果。 本文为稀疏数据和加权分位数草图提出了一种新的稀疏感知算法，用来对树进行近似的学习。 更重要的是，本文提供关于缓存访问模式，数据压缩和切片的研究，以构建可伸缩的提升树系统。 结合以上这些方法，XGBoost可以使用比现有系统少得多的资源来对数十亿个样本进行建模。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/figure33.png)
</p></blockquote></details>


#### 2001
<details/>
<summary/>
 <a href="https://link.springer.com/article/10.1023%2FA%3A1010933404324"> --- Random Forests (RF) --- by<i> Leo Breiman 
</a></a> (<a href="https://github.com/hexiaolang/RandomForest-In-text-classification">{Github}</a>) </summary><blockquote><p align="justify">
  Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost (Y. Freund & R. Schapire, Machine Learning: Proceedings of the Thirteenth International conference, ***, 148–156), but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.
  
随机森林是多个决策树预测器的组合，其中每棵树都取决于独立采样的随机向量的值，并且森林中的所有树都具有相同的分布。森林的泛化误差收敛随着森林中树的数量的增加而达到极限。由决策树构成的随机森林的泛化误差取决于森林中各个树的强度以及它们之间的相关性。使用随机选择的功能来分割每个节点所产生的错误率可与Adaboost相比，但对噪声的鲁棒性更强。内在的估计监视误差，强度和相关性，同时也能反应出模型对特征数量增加的响应。内部估计也用于衡量变量的重要性。这些方法不仅适用于分类，同样也适用于回归。
</p></blockquote></details>

#### 1998
<details/>
<summary/>
<a href="https://xueshu.baidu.com/usercenter/paper/show?paperid=58aa6cfa340e6ae6809c5deadd07d88e&site=xueshu_se">Text categorization with Support Vector Machines: Learning with many relevant features (SVM)</a>  by<i> JOACHIMS,T.
</a> (<a href="https://github.com/Gunjitbedi/Text-Classification">{Github}</a>) </summary><blockquote><p align="justify">
  This paper explores the use of Support Vector Machines (SVMs) for learning text classifiers from examples. It analyzes the particular properties of learning with text data and identifies why SVMs are appropriate for this task. Empirical results support the theoretical findings. SVMs achieve substantial improvements over the currently best performing methods and behave robustly over a variety of different learning tasks. Furthermore they are fully automatic, eliminating the need for manual parameter tuning.
  
本文探讨了使用支持向量机（SVM）从样本中学习文本分类器的方法。 它分析了使用文本数据进行学习的特殊属性，并确定了SVM为什么适合此任务。 实证结果证明了理论分析的正确性。 SVM相较于当前效果最好的方法有巨大提升，并在各种不同的学习任务中表现出色。 此外，该模型是全自动的，无需手动进行参数调整。
</p></blockquote></details>

#### 1993
<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">C4.5: Programs for Machine Learning (C4.5)</a> by<i> Steven L. Salzberg 
</a>  (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  C4.5算法是由Ross Quinlan开发的用于产生决策树的算法，该算法是对Ross Quinlan之前开发的ID3算法的一个扩展。C4.5算法主要应用于统计分类中，主要是通过分析数据的信息熵建立和修剪决策树。
</p></blockquote></details>

#### 1984
<details/>
<summary/>
<a href="https://dblp.org/img/paper.dark.empty.16x16.png">Classification and Regression Trees (CART)</a> by<i> Chyon-HwaYeh
</a> (<a href="https://github.com/sayantann11/all-classification-templetes-for-ML">{Github}</a>) </summary><blockquote><p align="justify">
  分类与回归树CART是由Loe Breiman等人在1984年提出的，自提出后被广泛的应用。CART既能用于分类也能用于回归，和决策树相比较，CART把选择最优特征的方法从信息增益（率）换成了基尼指数。
</p></blockquote></details>

#### 1967
<details/>
<summary/>
<a href="https://dl.acm.org/doi/10.1145/321075.321084">Nearest neighbor pattern classification (k-nearest neighbor classification,KNN)</a> by<i> M. E. Maron
</a> (<a href="https://github.com/raimonbosch/knn.classifier">{Github}</a>) </summary><blockquote><p align="justify">
  The nearest neighbor decision rule assigns to an unclassified sample point the classification of the nearest of a set of previously classified points. This rule is independent of the underlying joint distribution on the sample points and their classifications, and hence the probability of errorRof such a rule must be at least as great as the Bayes probability of errorR^{\ast}--the minimum probability of error over all decision rules taking underlying probability structure into account. However, in a large sample analysis, we will show in theM-category case thatR^{\ast} \leq R \leq R^{\ast}(2 --MR^{\ast}/(M-1)), where these bounds are the tightest possible, for all suitably smooth underlying distributions. Thus for any number of categories, the probability of error of the nearest neighbor rule is bounded above by twice the Bayes probability of error. In this sense, it may be said that half the classification information in an infinite sample set is contained in the nearest neighbor.
  
  最近邻决策规则将待分类样本点的一组已分类样本点中的最近点的类别分配给该分类点。 该规则与样本点及其类别的基础联合分布无关，因此，该规则的错误概率R必须至少与贝叶斯分类错误概率R ^ {\ ast}一样大，即最小错误概率（一种在所有决策规则中都考虑了潜在的概率结构）。 但是，在大量样本分析中，在在M类情况下（R ^ {\ ast} \ leq R \ leq R ^ {\ ast}（2 --MR ^ {\ ast} /（M-1））（应该是一个错误概率））对于所有适当平滑的基础分布，这些界限都可能是最严格的。 因此，对于任何数量的类别，最近邻居规则的错误概率都以两倍的贝叶斯错误概率为界。 从这个意义上讲，可以说无限样本集中分类信息的一半包含在最近的邻居中。
</p></blockquote></details>


#### 1961 

<details/>
<summary/>
<a href="https://dl.acm.org/doi/10.1145/321075.321084">Automatic indexing: An experimental inquiry</a> by<i> M. E. Maron
</a> (<a href="https://github.com/Gunjitbedi/Text-Classification">{Github}</a>) </summary><blockquote><p align="justify">
  This inquiry examines a technique for automatically classifying (indexing) documents according to their subject content. The task, in essence, is to have a computing machine read a document and on the basis of the occurrence of selected clue words decide to which of many subject categories the document in question belongs. This paper describes the design, execution and evaluation of a modest experimental study aimed at testing empirically one statistical technique for automatic indexing.
  
本文研究了一种根据文档的主题内容自动分类（索引）文档的技术。 该技术本质上是让计算机读取文档，并根据所选线索词的出现频率来确定所讨论文档属于多个主题类别中的哪一个。 本文介绍了如何设计，执行和评估相关的实验研究，以对自动索引统计技术进行实证检验。
</p></blockquote></details>





# Datasets
[:arrow_up:](#table-of-contents)

#### Sentiment Analysis (SA) 情感分析
SA is the process of analyzing and reasoning the subjective text withinemotional color. It is crucial to get information on whether it supports a particular point of view fromthe text that is distinct from the traditional text classification that analyzes the objective content ofthe text. SA can be binary or multi-class. Binary SA is to divide the text into two categories, includingpositive and negative. Multi-class SA classifies text to multi-level or fine-grained labels. 

情感分析（Sentiment Analysis，SA）是在情感色彩中对主观文本进行分析和推理的过程。 通过分析文本来判断作者是否支持特定观点的信息至关重要，这与分析文本客观内容的传统文本分类任务不同。 SA可以是二分类也可以是多分类。 Binary SA将文本分为两类，包括肯定和否定。 多类SA将文本分类为多级或细粒度更高的不同标签。

<details/>
<summary/> <a href="http://www.cs.cornell.edu/people/pabo/movie-review-data/">Movie Review (MR) 电影评论数据集</a></summary><blockquote><p align="justify">
The MR is a movie review dataset, each of which correspondsto a sentence. The corpus has 5,331 positive data and 5,331 negative data. 10-fold cross-validationby random splitting is commonly used to test MR. 
  
  MR是电影评论数据集， 其中每个样本对应一个句子。 语料库有5,331个积极样本和5,331个消极样本。 该数据集通常通过随机划分的10折交叉验证来验证模型效果。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.cs.uic.edu/∼liub/FBS/sentiment-analysis.html">Stanford Sentiment Treebank (SST) 斯坦福情感库</a></summary><blockquote><p align="justify">
The SST [175] is an extension of MR. It has two cate-gories. SST-1 with fine-grained labels with five classes. It has 8,544 training texts and 2,210 testtexts, respectively. Furthermore, SST-2 has 9,613 texts with binary labels being partitioned into6,920 training texts, 872 development texts, and 1,821 testing texts.
  
 SST是MR的扩展，它有两个不同版本。 SST-1带有五类细粒度标签。 它分别具有8,544个训练文本和2,210个测试文本。 SST-2则有两类标签，9,613个的文本，这些文本被分为6,920个训练文本，872个开发文本和1,821个测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.cs.pitt.edu/mpqa/">The Multi-Perspective Question Answering (MPQA)多视角问答数据集</a></summary><blockquote><p align="justify">
The MPQA is an opinion dataset. It has two class labels and also an MPQA dataset of opinion polarity detection sub-tasks.MPQA includes 10,606 sentences extracted from news articles from various news sources. It shouldbe noted that it contains 3,311 positive texts and 7,293 negative texts without labels of each text.
  
 MPQA是意见数据集。 它有两个类别标签，还有一个MPQA意见极性检测子任务数据集. MPQA包括从各种来源的新闻文章中提取的10606个句子。 应当指出的是，它包含3,311个积极文本和7,293个消极文本，每个文本没有标签。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/kdd/DiaoQWSJW14">IMDB reviews IMDB评论</a></summary><blockquote><p align="justify">
The IMDB review is developed for binary sentiment classification of filmreviews with the same amount in each class. It can be separated into training and test groups onaverage, by 25,000 comments per group.
  
 IMDB评论专为电影评论的二元情感分类而开发，每个类别中的评论数量相同。 可以将其平均分为培训和测试组，每组25,000条评论。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/emnlp/TangQL15">Yelp reviews Yelp评论</a></summary><blockquote><p align="justify">
The Yelp review is summarized from the Yelp Dataset Challenges in 2013,2014, and 2015. This dataset has two categories. Yelp-2 of these were used for negative and positiveemotion classification tasks, including 560,000 training texts and 38,000 test texts. Yelp-5 is used todetect fine-grained affective labels with 650,000 training and 50,000 test texts in all classes.
  
 Yelp评论数据集总结自2013、2014和2015年的Yelp数据集挑战。此数据集有两个版本。 Yelp-2被用于消极和积极情绪分类任务，包括560,000个训练文本和38,000测试文本。 Yelp-5用于细粒度情感多分类任务，包含650,000个训练文本和50,000测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products">Amazon Reviews (AM) 亚马逊评论数据集</a></summary><blockquote><p align="justify">
The AM is a popular corpus formed by collecting Amazon websiteproduct reviews [190]. This dataset has two categories. The Amazon-2 with two classes includes 3,600,000 training sets and 400,000 testing sets. Amazon-5, with five classes, includes 3,000,000 and650,000 comments for training and testing.
  
 AM是通过收集亚马逊网站的产品评论而形成的流行语料库。 该数据集有两个不同版本。 具有两个类别的Amazon-2包括3,600,000个训练样本和400,000个测试样本。 Amazon-5具有五个类别，包括3,000,000个训练样本和650,000个测试样本。
</p></blockquote></details>


#### News Classification (NC) 新闻分类数据集
News content is one of the most crucial information sources which hasa critical influence on people. The NC system facilitates users to get vital knowledge in real-time.News classification applications mainly encompass: recognizing news topics and recommendingrelated news according to user interest. The news classification datasets include 20NG, AG, R8, R52,Sogou, and so on. Here we detail several of the primary datasets.

新闻内容是最关键的信息来源之一，对人们的生活具有重要的影响。 数控系统方便用户实时获取重要知识。 新闻分类应用主要包括：识别新闻主题并根据用户兴趣推荐相关新闻。 新闻分类数据集包括20NG，AG，R8，R52，Sogou等。 在这里，我们详细介绍了一些主要数据集。

<details/>
<summary/>
<a href="http://ana.cachopo.org/datasets-for-single-label-text-categorization">20 Newsgroups (20NG)</a></summary><blockquote><p align="justify">
 The 20NG is a newsgroup text dataset. It has 20 categories withthe same number of each category and includes 18,846 texts.
  
 20NG是新闻组文本数据集。 它有20个类别，每个类别样本数目相同，一共包含18,846篇文本。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html">AG News (AG)</a></summary><blockquote><p align="justify">
The AG News is a search engine for news from academia, choosingthe four largest classes. It uses the title and description fields of each news. AG contains 120,000texts for training and 7,600 texts for testing.
  
 AG新闻是搜索学术界新闻的搜索引擎，它选择了四个规模最大的类别。 它使用每个新闻的标题和描述字段。 AG包含用于训练的120,000个文本和用于测试的7,600个文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://www.cs.umb.edu/~smimarog/textmining/datasets/">R8 and R52</a></summary><blockquote><p align="justify">
R8 and R52 are two subsets which are the subset of Reuters. R8 has 8categories, divided into 2,189 test files and 5,485 training courses. R52 has 52 categories, split into6,532 training files and 2,568 test files.
  
 R8和R52是路透社新闻的两个子集。 R8有8个类别，分为2189个测试样本和5485个训练样本。 R52有52个类别，分为6,532个训练样本和2,568个测试样本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/conf/cncl/SunQXH19.bib">Sogou News (Sogou) 搜狗新闻</a></summary><blockquote><p align="justify">
The Sogou News combines two datasets, including SogouCA andSogouCS news sets. The label of each text is the domain names in the URL.
  
 搜狗新闻数据集包含搜狗CA新闻集和搜狗CS新闻集。 每个文本的标签是URL中的域名。
</p></blockquote></details>

#### Topic Labeling (TL) 话题标签
The topic analysis attempts to get the meaning of the text by defining thesophisticated text theme. The topic labeling is one of the essential components of the topic analysistechnique, intending to assign one or more subjects for each document to simplify the topic analysis.

话题分析旨在通过定义复杂的文本主题来获取文本的含义。 话题标记是话题分析技术的重要组成部分之一，旨在为每个文档分配一个或多个话题标签以简化话题分析。

<details/>
<summary/> <a href="https://dblp.org/rec/journals/semweb/LehmannIJJKMHMK15.bib">DBpedia</a></summary><blockquote><p align="justify">
The DBpedia is a large-scale multi-lingual knowledge base generated usingWikipedia’s most ordinarily used infoboxes. It publishes DBpedia each month, adding or deletingclasses and properties in every version. DBpedia’s most prevalent version has 14 classes and isdivided into 560,000 training data and 70,000 test data. 
  
 DBpedia是使用Wikipedia最常用的信息框生成的大规模多语言知识库。 每个月都会发布新版本的DBpedia，并在每个版本中添加或删除类和属性。 DBpedia最流行的版本有14个类别，包含560,000个训练数据和70,000个测试数据。
</p></blockquote></details>

<details/>
<summary/> <a href="http://davis.wpi.edu/xmdv/datasets/ohsumed.html">Ohsumed</a></summary><blockquote><p align="justify">
The Ohsumed belongs to the MEDLINE database. It includes 7,400 texts andhas 23 cardiovascular disease categories. All texts are medical abstracts and are labeled into one ormore classes.
  
 Ohsumed隶属于MEDLINE数据库。 它包括7,400个文本，并有23种心血管疾病类别。 所有文本均为医学摘要，并被标记为一个或多个类。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Yahoo answers (YahooA) 雅虎问答</a></summary><blockquote><p align="justify">
The YahooA is a topic labeling task with 10 classes. It includes140,000 training data and 5,000 test data. All text contains three elements, being question titles,question contexts, and best answers, respectively.
  
 YahooA是具有10个类的话题标记数据集。 它包括140,000个训练数据和5,000个测试数据。 所有文本均包含三个元素，分别是问题标题，问题上下文和最佳答案。
</p></blockquote></details>


#### Question Answering (QA) 问答
The QA task can be divided into two types: the extractive QA and thegenerative QA. The extractive QA gives multiple candidate answers for each question to choosewhich one is the right answer. Thus, the text classification models can be used for the extractiveQA task. The QA discussed in this paper is all extractive QA. The QA system can apply the textclassification model to recognize the correct answer and set others as candidates. The questionanswering datasets include SQuAD, MS MARCO, TREC-QA, WikiQA, and Quora [209]. Here wedetail several of the primary datasets.

问答任务可以分为两种：抽取式问答（extractiveQA）和生成式问答（extractiveQA）。 抽取式问答为每个问题提供了多个候选答案，以选择哪个是正确答案。 因此，文本分类模型可以用于抽取式问答任务。QA系统可以使用文本分类模型来识别正确答案，并将其他答案设置为候选答案。 问答数据集包括SQuAD，MS MARCO，TREC-QA，WikiQA和Quora [209]。 这里我们详细介绍了几个主要数据集。

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Stanford Question Answering Dataset (SQuAD） 斯坦福问答数据集</a></summary><blockquote><p align="justify">
The SQuAD is a set of question and answer pairs obtained from Wikipedia articles. The SQuAD has two categories. SQuAD1.1 contains 536 pairs of 107,785 Q&A items. SQuAD2.0 combines 100,000 questions in SQuAD1.1 with morethan 50,000 unanswerable questions that crowd workers face in a form similar to answerable questions.
  
SQuAD是由从Wikipedia文章获得的问题和答案对构成的数据集。 SQuAD有两个版本。 SQuAD1.1包含536对，107,785个问答项目。 SQuAD2.0将SQuAD1.1中的100,000个问题与超过50,000个无法回答的问题组合在一起。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">MS MARCO</a></summary><blockquote><p align="justify">
The MS MARCO contains questions and answers. The questions and part ofthe answers are sampled from actual web texts by the Bing search engine. Others are generative. Itis used for developing generative QA systems released by Microsoft.
  
MS MARCO包含问题和答案。 Bing搜索引擎从实际的网络文本中抽取了问题和部分答案。 其他则是生成的。该数据集用于开发Microsoft发布的生成质量保证系统。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">TREC-QA</a></summary><blockquote><p align="justify">
The TREC-QA includes 5,452 training texts and 500 testing texts. It has two versions. TREC-6 contains 6 categories, and TREC-50 has 50 categories.
  
TREC-QA包括5,452个训练文本和500个测试文本。 它有两个版本。 TREC-6包含6个类别，TREC-50包含50个类别。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">WikiQA</a></summary><blockquote><p align="justify">
The WikiQA dataset includes questions with no correct answer, which needs toevaluate the answer.
  
WikiQA数据集包含没有正确答案的问题，需要对答案进行评估。
</p></blockquote></details>



#### Natural Language Inference (NLI) 自然语言推理
NLI is used to predict whether the meaning of one text canbe deduced from another. Paraphrasing is a generalized form of NLI. It uses the task of measuringthe semantic similarity of sentence pairs to decide whether one sentence is the interpretation ofanother. The NLI datasets include SNLI, MNLI, SICK, STS, RTE, SciTail, MSRP, etc. Here we detailseveral of the primary datasets.

NLI用于预测一个文本的含义是否可以从另一个文本推论得出。 释义是NLI的一种广义形式。 它使用测量句子对语义相似性的任务来确定一个句子是否是另一句子的解释。 NLI数据集包括SNLI，MNLI，SICK，STS，RTE，SciTail，MSRP等。在这里，我们详细介绍了所有主要数据集。

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">The Stanford Natural Language Inference (SNLI)</a></summary><blockquote><p align="justify">
The SNLI is generally applied toNLI tasks. It contains 570,152 human-annotated sentence pairs, including training, development,and test sets, which are annotated with three categories: neutral, entailment, and contradiction.
  
  SNLI通常应用于NLI任务。 它包含570,152个人工注释的句子对，包括训练，发展和测试集，并用三类注释：中立，包含和矛盾。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Multi-Genre Natural Language Inference (MNLI)</a></summary><blockquote><p align="justify">
The Multi-NLI is an expansion of SNLI, embracing a broader scope of written and spoken text genres. It includes 433,000 sentencepairs annotated by textual entailment labels.
  
  Multi-NLI是SNLI的扩展，涵盖了更大范围的书面和口头文字类型。 它包括433,000个句子对，并带有文本是否蕴含的标签。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Sentences Involving Compositional Knowledge (SICK)</a></summary><blockquote><p align="justify">
The SICK contains almost10,000 English sentence pairs. It consists of neutral, entailment and contradictory labels.
  
SICK包含将近10,000个英语句子对。 它由中立，包含和矛盾的标签组成。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Microsoft Research Paraphrase (MSRP)</a></summary><blockquote><p align="justify">
The MSRP consists of sentence pairs, usuallyfor the text-similarity task. Each pair is annotated by a binary label to discriminate whether theyare paraphrases. It respectively includes 1,725 training and 4,076 test sets.
  
MSRP由句子对组成，通常用于文本相似性任务。 每对都用二进制标签注释，以区分它们是否由蕴含关系。 它包括1,725个训练样本和4,076个测试样本。
</p></blockquote></details>



#### Dialog Act Classification (DAC) 对话行为分类
A dialog act describes an utterance in a dialog based on semantic,pragmatic, and syntactic criteria. DAC labels a piece of a dialog according to its category of meaningand helps learn the speaker’s intentions. It is to give a label according to dialog. Here we detailseveral of the primary datasets, including DSTC 4, MRDA, and SwDA.

对话行为基于语义，语用和句法标准来描述对话中的话语。 DAC根据其含义类别标记一个对话框，并帮助理解讲话者的意图。它是根据对话框给标签。在这里，我们详细介绍了所有主要数据集，包括DSTC 4，MRDA和SwDA。

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Dialog State Tracking Challenge 4 (DSTC 4)</a></summary><blockquote><p align="justify">
The DSTC 4 is used for dialog act classi-fication. It has 89 training classes, 24,000 training texts, and 6,000 testing texts.
  
DSTC 4用于对话行为分类。 它有89个类别，24,000个训练文本和6,000个测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">ICSI Meeting Recorder Dialog Act (MRDA)</a></summary><blockquote><p align="justify">
The MRDA is used for dialog act classifi-cation. It has 5 training classes, 51,000 training texts, 11,000 testing texts, and 11,000 validation texts.
  
MRDA用于对话行为分类。 它有5个样本类别，51,000个训练文本，11,000个测试文本和11,000个验证文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Switchboard Dialog Act (SwDA)</a></summary><blockquote><p align="justify">
The SwDA is used for dialog act classification. It has43 training classes, 1,003,000 training texts, 19,000 testing texts and 112,000 validation texts.
  
SwDA用于对话行为分类。 它拥有43个训练类别，1,003,000个训练文本，19,000个测试文本和112,000个验证文本。
</p></blockquote></details>


#### Multi-label datasets 多标签数据集
In multi-label classification, an instance has multiple labels, and each la-bel can only take one of the multiple classes. There are many datasets based on multi-label textclassification. It includes Reuters, Education, Patent, RCV1, RCV1-2K, AmazonCat-13K, BlurbGen-reCollection, WOS-11967, AAPD, etc. Here we detail several of the main datasets.

在多标签分类中，一个实例具有多个标签，并且每个la-bel只能采用多个类之一。 有许多基于多标签文本分类的数据集。 它包括路透社，Education，Patent，RCV1，RCV1-2K，AmazonCat-13K，BlurbGen-reCollection，WOS-11967，AAPD等。这里我们详细介绍了一些主要数据集。

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Reuters news</a></summary><blockquote><p align="justify">
The Reuters is a popularly used dataset for text classification fromReuters financial news services. It has 90 training classes, 7,769 training texts, and 3,019 testingtexts, containing multiple labels and single labels. There are also some Reuters sub-sets of data,such as R8, BR52, RCV1, and RCV1-v2.
  
路透社新闻数据集是路透社金融新闻服务进行文本分类的常用数据集。 它具有90个训练类别，7,769个训练文本和3,019个测试文本，其中包含多个标签和单个标签。 它还有一些子数据集，例如R8，BR52，RCV1和RCV1-v2。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Patent Dataset</a></summary><blockquote><p align="justify">
The Patent Dataset is obtained from USPTO1, which is a patent system gratingU.S. patents containing textual details such title and abstract. It contains 100,000 US patents awardedin the real-world with multiple hierarchical categories.
  
专利数据集是从USPTO1获得的，USPTO1是美国的专利系统，包含文字详细信息（例如标题和摘要）的专利。 它包含在现实世界中授予的100,000种美国专利，具有多个层次类别。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Reuters Corpus Volume I (RCV1) and RCV1-2K</a></summary><blockquote><p align="justify">
The RCV1 is collected from Reuters News articles from 1996-1997, which is human-labeled with 103 categories. It consists of 23,149 training and 784,446 testing texts, respectively. The RCV1-2K dataset has the same features as the RCV1. However, the label set of RCV1-2K has been expanded with some new labels. It contains2456 labels.
  
RCV1是从1996-1997年的《路透社新闻》文章中收集的， 带有103个类别的人工标注标签。 它分别由23,149个训练和784,446个测试文本组成。 RCV1-2K数据集具有与RCV1相同的功能。 但是，RCV1-2K的标签集已经扩展了一些新标签。 它包含2456个标签。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Web of Science (WOS-11967)</a></summary><blockquote><p align="justify">
The WOS-11967 is crawled from the Web of Science,consisting of abstracts of published papers with two labels for each example. It is shallower, butsignificantly broader, with fewer classes in total.
  
WOS-11967是从Web of Science爬取的，它由已发表论文的摘要组成，每个示例带有两个标签。 该数据集样本数较少，但覆盖面明显更广泛，总共有较少的类。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Arxiv Academic Paper Dataset (AAPD)</a></summary><blockquote><p align="justify">
The AAPD is a large dataset in the computer science field for the multi-label text classification from website2. It has 55,840 papers, including the abstract and the corresponding subjects with 54 labels in total. The aim is to predict the corresponding subjects of each paper according to the abstract.
  
AAPD是计算机科学领域中的大型数据集，用于来自website2的多标签文本分类。 它拥有55,840篇论文，包括摘要和相应的主题，共有54个标签。目的是根据摘要预测每篇论文的主题。
</p></blockquote></details>


#### Others 其他
There are some datasets for other applications, such as Geonames toponyms, Twitter posts,and so on.

还有一些用于其他应用程序的数据集，比如Geonames toponyms、Twitter帖子等等。

# Evaluation Metrics
[:arrow_up:](#table-of-contents)

In terms of evaluating text classification models, accuracy and F1 score are the most used to assessthe text classification methods. Later, with the increasing difficulty of classification tasks or theexistence of some particular tasks, the evaluation metrics are improved. For example, evaluationmetrics such as P@K and Micro-F1 are used to evaluate multi-label text classification performance,and MRR is usually used to estimate the performance of QA tasks.

在评估文本分类模型方面，准确率和F1分数是评估文本分类方法最常用的指标。 随着分类任务难度的增加或某些特定任务的存在，评估指标也得到了改进。 例如P @ K和Micro-F1评估指标用于评估多标签文本分类性能，而MRR通常用于评估QA任务的性能。

#### Single-label metrics 单标签评价指标
Single-label text classification divides the text into one of the most likelycategories applied in NLP tasks such as QA, SA, and dialogue systems [9]. For single-label textclassification, one text belongs to just one catalog, making it possible not to consider the relationsamong labels. Here we introduce some evaluation metrics used for single-label text classificationtasks.

单标签文本分类将文本划分为NLP任务（如QA，SA和对话系统）中最相似的类别之一[9]。 对于单标签文本分类，一个文本仅属于一个目录，这使得不考虑标签之间的关系成为可能。 在这里，我们介绍一些用于单标签文本分类任务的评估指标。


<details/>
<summary/> <a>Accuracy and Error Rate</a></summary><blockquote><p align="justify">
Accuracy and Error Rate are the fundamental metrics for a text classification model. The Accuracy and Error Rate are respectively defined as
  
 准确性和错误率是文本分类模型的基本指标。 准确度和错误率分别定义为：
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/1.png)
</p></blockquote></details>

<details/>
<summary/> <a >Precision, Recall and F1</a></summary><blockquote><p align="justify">
These are vital metrics utilized for unbalanced test sets regardless ofthe standard type and error rate. For example, most of the test samples have a class label. F1 is theharmonic average of Precision and Recall. Accuracy, Recall, and F1 as defined
  
  无论标准类型和错误率如何，这些都是用于不平衡测试集的重要指标。 例如，大多数测试样本都具有类别标签。 F1是Precision和Recall的谐波平均值。 准确性，召回率和F1分数定义为：
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/2.png)
  
  The desired results will be obtained when the accuracy, F1 and recall value reach 1. On the contrary,when the values become 0, the worst result is obtained. For the multi-class classification problem,the precision and recall value of each class can be calculated separately, and then the performanceof the individual and whole can be analyzed.
  
  当准确率、F1和recall值达到1时，就可以得到预期的结果。相反，当值为0时，得到的结果最差。对于多类分类问题，可以分别计算各类的查准率和查全率，进而分析个体和整体的性能。
  </p></blockquote></details>

<details/>
<summary/> <a>Exact Match (EM)</a></summary><blockquote><p align="justify">
The EM is a metric for QA tasks measuring the prediction that matches all theground-truth answers precisely. It is the primary metric utilized on the SQuAD dataset.
  
EM是QA任务的度量标准，用于测量精确匹配所有正确答案的预测。 它是SQuAD数据集上使用的主要指标。
</p></blockquote></details>


<details/>
<summary/> <a >Mean Reciprocal Rank (MRR)</a></summary><blockquote><p align="justify">
The MRR is usually applied for assessing the performance of ranking algorithms on QA and Information Retrieval (IR) tasks.
  
  MRR通常用于评估在问答(QA)和信息检索(IR)任务中排序算法的性能。
</p></blockquote></details>

<details/>
<summary/> <a >Hamming-loss (HL)</a></summary><blockquote><p align="justify">
The HL assesses the score of misclassified instance-label pairs wherea related label is omitted or an unrelated is predicted.
  
  HL评估被错误分类的实例-标签对的得分，其中相关的标签被省略或不相关的标签被预测。
</p></blockquote></details>


#### Multi-label metrics 多标签评价指标
Compared with single-label text classification, multi-label text classifica-tion divides the text into multiple category labels, and the number of category labels is variable. These metrics are designed for single label text classification, which are not suitable for multi-label tasks. Thus, there are some metrics designed for multi-label text classification.

与单标签文本分类相比，多标签文本分类将文本分为多个类别标签，并且类别标签的数量是可变的。 然而上述的度量标准是为单标签文本分类设计的，不适用于多标签任务。 因此，存在一些为多标签文本分类而设计的度量标准。

<details/>
<summary/> <a >Micro−F1</a></summary><blockquote><p align="justify">
The Micro−F1 is a measure that considers the overall accuracy and recall of alllabels. The Micro−F1is defined as
  
  Micro-F1是一种考虑所有标签的整体精确率和召回率的措施。 Micro-F1定义为：
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/3.png)
    ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/4.png)
</p></blockquote></details>


<details/>
<summary/> <a >Macro−F1</a></summary><blockquote><p align="justify">
The Macro−F1 calculates the average F1 of all labels. Unlike Micro−F1, which setseven weight to every example, Macro−F1 sets the same weight to all labels in the average process. Formally, Macro−F1is defined as
  
  Marco-F1计算所有标签的平均F1分数。 与Micro-F1（每个示例都设置权重）不同，Macro-F1在平均过程中为所有标签设置相同的权重。 形式上，Macro-F1定义为
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/5.png)
    ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/6.png)
</p></blockquote></details>

In addition to the above evaluation metrics, there are some rank-based evaluation metrics forextreme multi-label classification tasks, including P@K and NDCG@K.

除了上述评估指标外，还有一些针对极端多标签分类任务的基于排序的评估指标，包括P @ K和NDCG @ K。

<details/>
<summary/> <a >Precision at Top K (P@K)</a></summary><blockquote><p align="justify">
The P@K is the precision at the top k. ForP@K, each text has a set of L ground truth labels Lt={l0,l1,l2...,lL−1}, in order of decreasing probability Pt=p0,p1,p2...,pQ−1.The precision at k is
  
  其中P@K为排名第k处的准确率。P@K，每个文本有一组L个全局真标签Lt={l0,l1,l2...,lL−1}, 为了减少概率Pt=p0,p1,p2...,pQ−1。第k处的准确率为
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/7.png)
</p></blockquote></details>

<details/>
<summary/> <a>Normalized Discounted Cummulated Gains (NDCG@K)</a></summary><blockquote><p align="justify">
The NDCG at k is
  
  排名第k处的NDCG值
  
   ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figures/9.png)
  
</p></blockquote></details>


# Future Research Challenges
[:arrow_up:](#table-of-contents)

文本分类作为高效的信息检索和挖掘技术，在文本数据的自动化管理中起着至关重要的作用。其中涉及到使用NLP、数据挖掘、机器学习和其他技术来实现自动主题分类或发掘新的不同的文本类型。文本分类将多种类型的文本作为输入，并且由预训练模型表示为可以计算的向量，然后将向量喂到DNN中进行训练，直到达到终止条件为止，最后，在下游任务验证训练模型的性能。现有的文本分类模型已经在实际应用中显现出了其可用性，但是仍有许多可改进的地方需要继续探索。

尽管一些新的文本分类模型不断刷新了大多数分类任务的准确率指标记录，但这并不能说明模型是否能像人类一样从语义层面“理解”文本。此外，随着噪声样本的出现，小的样本噪声可能导致决策置信度发生实质性变化，甚至逆转决策结果。因此，需要在实践中证明该模型的语义表示能力和鲁棒性。此外，由词向量表示的预训练语义表征模型往往可以提高下游NLP任务的性能。现有的上下文无关词向量迁移学习的研究还比较初步。因此，我们从数据，模型和性能三个角度总结出文本分类主要面临以下挑战：


#### 数据层面

对于文本分类任务，无论是浅层学习还是深度学习方法，数据对于模型性能都是必不可少的。研究的文本数据主要包括多篇章，短文本，跨语言，多标签，少样本文本。针对于这些数据的特质，现有的技术挑战如下：


<details/>
<summary/>
<a >Zero-shot/Few-shot learning</a> </summary><blockquote><p align="justify">
  用于文本分类的零样本或少样本学习旨在对没有或只有很少的相同标签类数据的文本进行分类。然而，当前模型过于依赖大量标记数据，它们的性能受零样本或少样本学习的影响很大。因此，一些工作着重于解决这些问题，其主要思想是通过学习各种语义知识来推断特征，例如学习类之间的关系和合并类描述。此外，潜在特征生成、元学习和动态记忆力机制也是有效的方法。尽管如此，由于少量未知类型的数据的限制以及已知和未知类别数据之间不同的数据分布，要达到与人类相当的学习能力还有很长的路要走。
</p></blockquote></details>

<details/>
<summary/>
<a >外部知识</a>  </summary><blockquote><p align="justify">
  众所周知，将更多有益的信息输入到DNN中，其性能会更好。因此，添加外部知识（知识库或知识图谱）是提高模型性能的有效方法。现有知识包括概念信息，常识知识，知识库信息，通用知识图谱等，这些知识增强了文本的语义表示。然而，由于投入规模的限制，如何为不同任务增加知识以及增加什么样的外部知识仍然是一个挑战。
</p></blockquote></details>

<details/>
<summary/>
<a >多标签文本分类任务</a>  </summary><blockquote><p align="justify">
  多标签文本分类需要充分考虑标签之间的语义关系，而模型的嵌入和编码是有损的压缩过程。因此，如何减少训练过程中层次语义的丢失以及如何保留丰富而复杂的文档语义信息仍然是一个亟待解决的问题。
</p></blockquote></details>

<details/>
<summary/>
<a >具有许多术语词汇的特殊领域</a> </summary><blockquote><p align="justify">
  特定领域的文本（例如金融和医学文本）包含许多特定的单词或领域专家才可理解的词汇，缩写等，这使得现有的预训练词向量难以使用。
</p></blockquote></details>


#### 模型层面

大多数现有的浅层和深度学习模型的结构可以用于文本分类，包括集成方法。 BERT学习了一种可用于微调许多下游NLP任务语言表征形式。主要方法是增加数据，提高计算能力以及设计训练程序以获得更好的结果。如何在数据与计算资源以及预测性能之间进行权衡值得研究。

#### 性能评估层面

浅层学习模型和深度学习模型可以在大多数文本分类任务中实现良好的性能，但是需要提高其结果的抗干扰能力。如何实现对深度模型的解释也是一个技术挑战。

<details/>
<summary/>
<a >模型的语义鲁棒性</a>  </summary><blockquote><p align="justify">
  近年来，研究人员设计了许多模型来增强文本分类模型的准确性。 但是，如果数据集中有一些对抗性样本，则模型的性能会大大降低。因此，如何提高模型的鲁棒性是当前研究的热点和挑战。
</p></blockquote></details>

<details/>
<summary/>
<a >模型的可解释性</a> </summary><blockquote><p align="justify">
 DNN在特征提取和语义挖掘方面具有独特的优势，并且已经出色地完成了文本分类任务。但是，深度学习是一个黑盒模型，训练过程难以重现，隐层的语义和输出可解释性很差。尽管它对模型进行了改进和优化，但是却缺乏明确的指导。此外，我们无法准确解释为什么该模型可以提高性能。
</p></blockquote></details>


# Tools and Repos
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

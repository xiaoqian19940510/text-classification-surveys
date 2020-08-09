# Text Classification papers

This repository contains resources for Natural Language Processing (NLP) with a focus on the task of Text Classification.

# Table of Contents

<details>

<summary><b>Expand Table of Contents</b></summary><blockquote><p align="justify">

- [Surveys](#Surveys)
- [Shallow Learning Models](#Shallow Learning Models)
- [Deep Learning models](#Deep Learning models)
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
<summary>1. <a href="https://dl.acm.org/doi/10.1145/321075.321084">Naïve Bayes (NB)</a> <a href="https://github.com/Gunjitbedi/Text-Classification">{Code}</a> </summary><blockquote><p align="justify">
</p></blockquote></details>


## Deep learning Models
[:arrow_up:](#table-of-contents)
### 2015 



<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P15-2060/">Event detection and domain adaptation with convolutional neural networks</a> by<i> Thien Huu Nguyen, Ralph Grishman </i>(<a href="https://github.com/ThanhChinhBK/event_detector">Github</a>)</summary><blockquote><p align="justify">
We study the event detection problem using convolutional neural networks (CNNs) that overcome the two fundamental limitations of the traditional feature-based approaches to this task: complicated feature engineering for rich feature sets and error propagation from the preceding stages which generate these features. The experimental results show that the CNNs outperform the best reported feature-based systems in the general setting as well as the domain adaptation setting without resorting to extensive external resources.
</p></blockquote></details>



## Data
[:arrow_up:](#table-of-contents)

* <a href="https://www-nlpir.nist.gov/related_projects/muc/muc_data/muc_data_index.html">MUC Data Sets</a>
* Automatic Content Extraction (ACE) 
	* <a href="https://www.ldc.upenn.edu/collaborations/past-projects/ace">Program</a>
	* <a href="https://catalog.ldc.upenn.edu/LDC2003T11">ACE-2 corpus</a>
	* <a href="https://catalog.ldc.upenn.edu/LDC2006T06">ACE 05 corpus</a>
* Light & Rich ERE
	* <a href="http://www.aclweb.org/old_anthology/W/W15/W15-0812.pdf">Paper</a> 
* <a href="http://www.newsreader-project.eu/results/data/the-ecb-corpus/">The ECB+ Corpus</a>
* <a href="http://www.newsreader-project.eu/results/data/wikinews/">The NewsReader MEANTIME corpus</a>
* <a href="https://tac.nist.gov//2015/KBP/Event/index.html">TAC KBP 2015 Event Track</a>
* <a href="https://tac.nist.gov/">Text Analysis Conference</a>
* <a href="https://framenet.icsi.berkeley.edu/fndrupal/">FrameNet</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2013T19">OntoNotes Release 5.0</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2008T23">NomBank v 1.0</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2004T14">Proposition Bank I</a>
* <a href="https://catalog.ldc.upenn.edu/LDC99T42">Treebank-3</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2014T12">Abstract Meaning Representation (AMR) Annotation Release 1.0</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2006T08">TimeBank 1.2</a>
* <a href="http://universal.elra.info/product_info.php?cPath=42_43&products_id=2333">AQUAINT TimeML</a>  ( <a href="https://github.com/cnorthwood/ternip/tree/master/sample_data/aquaint_timeml_1.0">data</a> )
* <a href="https://www.cs.york.ac.uk/semeval-2013/task1/index.html">TempEval-3</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2009T23">FactBank 1.0</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2011T08">Datasets for Generic Relation Extraction (reACE)</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2016T23">Richer Event Description</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2005T16">TDT4 Multilingual Text and Annotations</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2017T09">The EventStatus Corpus</a>
* <a href="https://catalog.ldc.upenn.edu/LDC2009T10">Language Understanding Annotation Corpus</a>
* <a href="http://nactem.ac.uk/MLEE/">Multi-Level Event Extraction </a>
* <a href="https://osf.io/enu2k/">SentiFM company-specific economic news event dataset (English)</a>
<details><summary><a href="https://www.aclweb.org/anthology/2020.lrec-1.763/">A French Corpus for Event Detection on Twitter</a> by<i> Béatrice Mazoyer, Julia Cagé, Nicolas Hervé, Céline Hudelot </i></summary><blockquote><p align="justify">
We present Event2018, a corpus annotated for event detection tasks, consisting of 38 million tweets in French (retweets excluded) including more than 130,000 tweets manually annotated by three annotators as related or unrelated to a given event. The 243 events were selected both from press articles and from subjects trending on Twitter during the annotation period (July to August 2018). In total, more than 95,000 tweets were annotated as related to one of the selected events. We also provide the titles and URLs of 15,500 news articles automatically detected as related to these events. In addition to this corpus, we detail the results of our event detection experiments on both this dataset and another publicly available dataset of tweets in English. We ran extensive tests with different types of text embeddings and a standard Topic Detection and Tracking algorithm, and detail our evaluation method. We show that tf-idf vectors allow the best performance for this task on both corpora. These results are intended to serve as a baseline for researchers wishing to test their own event detection systems on our corpus.
</p></blockquote></details>

<details><summary><a href="https://www.aclweb.org/anthology/2020.lrec-1.509/">A Dataset for Multi-lingual Epidemiological Event Extraction</a> by<i> Stephen Mutuvi, Antoine Doucet, Gaël Lejeune, Moses Odeo </i></summary><blockquote><p align="justify">
This paper proposes a corpus for the development and evaluation of tools and techniques for identifying emerging infectious disease threats in online news text. The corpus can not only be used for information extraction, but also for other natural language processing (NLP) tasks such as text classification. We make use of articles published on the Program for Monitoring Emerging Diseases (ProMED) platform, which provides current information about outbreaks of infectious diseases globally. Among the key pieces of information present in the articles is the uniform resource locator (URL) to the online news sources where the outbreaks were originally reported. We detail the procedure followed to build the dataset, which includes leveraging the source URLs to retrieve the news reports and subsequently pre-processing the retrieved documents. We also report on experimental results of event extraction on the dataset using the Data Analysis for Information Extraction in any Language(DAnIEL) system. DAnIEL is a multilingual news surveillance system that leverages unique attributes associated with news reporting to extract events: repetition and saliency. The system has wide geographical and language coverage, including low-resource languages. In addition, we compare different classification approaches in terms of their ability to differentiate between epidemic-related and unrelated news articles that constitute the corpus.
</p></blockquote></details>

<details><summary><a href="https://www.aclweb.org/anthology/2020.lrec-1.203/">An Event-comment Social Media Corpus for Implicit Emotion Analysis</a> by<i> Sophia Yat Mei Lee, Helena Yan Ping Lau </i></summary><blockquote><p align="justify">
The classification of implicit emotions in text has always been a great challenge to emotion processing. Even though the majority of emotion expressed implicitly, most previous attempts at emotions have focused on the examination of explicit emotions. The poor performance of existing emotion identification and classification models can partly be attributed to the disregard of implicit emotions. In view of this, this paper presents the development of a Chinese event-comment social media emotion corpus. The corpus deals with both explicit and implicit emotions with more emphasis being placed on the implicit ones. This paper specifically describes the data collection and annotation of the corpus. An annotation scheme has been proposed for the annotation of emotion-related information including the emotion type, the emotion cause, the emotion reaction, the use of rhetorical question, the opinion target (i.e. the semantic role in an event that triggers an emotion), etc. Corpus data shows that the annotated items are of great value to the identification of implicit emotions. We believe that the corpus will be a useful resource for both explicit and implicit emotion classification and detection as well as event classification.
</p></blockquote></details>

<details><summary><a href="https://www.aclweb.org/anthology/2020.lrec-1.174/">FloDusTA: Saudi Tweets Dataset for Flood, Dust Storm, and Traffic Accident Events</a> by<i> Btool Hamoui, Mourad Mars, Khaled Almotairi </i></summary><blockquote><p align="justify">
The rise of social media platforms makes it a valuable information source of recent events and users’ perspective towards them. Twitter has been one of the most important communication platforms in recent years. Event detection, one of the information extraction aspects, involves identifying specified types of events in the text. Detecting events from tweets can help to predict real-world events precisely. A serious challenge that faces Arabic event detection is the lack of Arabic datasets that can be exploited in detecting events. This paper will describe FloDusTA, which is a dataset of tweets that we have built for the purpose of developing an event detection system. The dataset contains tweets written in both Modern Standard Arabic and Saudi dialect. The process of building the dataset starting from tweets collection to annotation by human annotators will be present. The tweets are labeled with four labels: flood, dust storm, traffic accident, and non-event. The dataset was tested for classification and the result was strongly encouraging.
</p></blockquote></details>

<details><summary><a href="https://www.aclweb.org/anthology/2020.lrec-1.609/">Manovaad: A Novel Approach to Event Oriented Corpus Creation Capturing Subjectivity and Focus</a> by<i> Lalitha Kameswari, Radhika Mamidi </i></summary><blockquote><p align="justify">
In today’s era of globalisation, the increased outreach for every event across the world has been leading to conflicting opinions, arguments and disagreements, often reflected in print media and online social platforms. It is necessary to distinguish factual observations from personal judgements in news, as subjectivity in reporting can influence the audience’s perception of reality. Several studies conducted on the different styles of reporting in journalism are essential in understanding phenomena such as media bias and multiple interpretations of the same event. This domain finds applications in fields such as Media Studies, Discourse Analysis, Information Extraction, Sentiment Analysis, and Opinion Mining. We present an event corpus Manovaad-v1.0 consisting of 1035 news articles corresponding to 65 events from 3 levels of newspapers viz., Local, National, and International levels. Using this novel format, we correlate the trends in the degree of subjectivity with the geographical closeness of reporting using a Bi-RNN model. We also analyse the role of background and focus in event reporting and capture the focus shift patterns within a global discourse structure for an event. We do this across different levels of reporting and compare the results with the existing work on discourse processing.
</p></blockquote></details>

## Tools and Repos

<details>
<summary> <a href="http://www.newsreader-project.eu/results/software/">NewsReader project</a></summary><blockquote><p align="justify">
On this page, you can find the different software modules developed by the NewsReader project. The easiest setup is provided by the virtual machine package that contains the complete pipelines. For those interested in trying out different parts of the pipelines, all separate modules are listed below as well. Please note that the pipelines take NAF files as input, for which we have made available Java and Python libraries.

With each module, we specify who developed it. The quickest way to get help with a module is to contact that person. If a publication is associated with a module, it will be specified on the module’s page.
</p></blockquote></details>


<details>
<summary><a href="http://wit.istc.cnr.it/stlab-tools/fred/">FRED : Machine Reading for the Semantic Web </a></summary><blockquote><p align="justify">
FRED is a machine reader for the Semantic Web: it is able to parse natural language text in 48 different languages and transform it to linked data. It is implemented in Python and available as REST service and as a Python library suite [fredlib]. FRED background theories are: Combinatory Categorial Grammar [C&C], Discourse Representation Theory [DRT, Boxer], Frame Semantics [Fillmore 1976] and Ontology Design Patterns [Ontology Handbook]. FRED leverages Natural Language Processing components for performing Named Entity Resolution [Stanbol, TagMe], Coreference Resolution [CoreNLP], and Word Sense Disambiguation [Boxer, IMS]. All FRED graphs include textual annotations and represent textual segmentation, expressed by means of EARMARK and NIF.
</p></blockquote></details>

<details>
<summary><a href="https://github.com/jbjorne/TEES">Turku Event Extraction System 2.3</a></summary><blockquote><p align="justify">
Turku Event Extraction System (TEES) is a free and open source natural language processing system developed for the extraction of events and relations from  biomedical text. It is written mostly in Python, and should work in generic Unix/Linux environments.
</p></blockquote></details>

<details>
<summary><a href="https://github.com/nlpcl-lab/bert-event-extraction">bert-event-extraction</a></summary><blockquote><p align="justify">
Pytorch Solution of Event Extraction Task using BERT on ACE 2005 corpus
</p></blockquote></details>

<details>
<summary><a href="https://github.com/nlpcl-lab/ace2005-preprocessing">ACE2005 preprocessing</a></summary><blockquote><p align="justify">
This is a simple code for preprocessing ACE 2005 corpus for Event Extraction task.
</p></blockquote></details>

<details>
<summary><a href="https://github.com/Hanlard/Transformer-based-pretrained-model-for-event-extraction">Transformer-based-pretrained-model-for-event-extraction</a></summary><blockquote><p align="justify">
Pre-trained language models such as BERT / OpenAI-GPT2 / ALBERT / XLM / Roberta / XLNet / Ctrl / DistilBert / TransfoXL are used to perform event extraction tasks on the ace2005 dataset.
</p></blockquote></details>

<details>
<summary><a href="https://github.com/hltcoe/EventMiner">EventMiner</a></summary><blockquote><p align="justify">
EventMiner aims to serve, primarily, as an interface to various NLP analytics to extract event information from text. This project is setup with a REST frontend interface, which accepts text input, that is then further passed via a RabbitMQ messaging queue to various analytics as appropriate. The project is comprised of Docker containers, with orchestration handled by docker-compose. This, combined with RabbitMQ as the messaging layer, allows for clean definitions of interactions between services and minimal setup for the end user.
</p></blockquote></details>

<details>
<summary><a href="https://github.com/lx865712528/EMNLP2018-JMEE">Jointly Multiple Events Extraction</a></summary><blockquote><p align="justify">
This is the code of the Jointly Multiple Events Extraction (JMEE) in our EMNLP 2018 paper.
</p></blockquote></details>

<details>
<summary><a href="https://github.com/lx865712528/ACL2019-ODEE">Open Domain Event Extraction Using Neural Latent Variable Models</a></summary><blockquote><p align="justify">
This is the python3 code for the paper "Open Domain Event Extraction Using Neural Latent Variable Models" in ACL 2019.
</p></blockquote></details>

<details>
<summary><a href="https://github.com/ahsi/Multilingual_Event_Extraction">CMU Multilingual Event Extractor</a></summary><blockquote><p align="justify">
Python code to run ACE-style event extraction on English, Chinese, or Spanish texts 
</p></blockquote></details>

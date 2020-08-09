# Text Classification papers

This repository contains resources for Natural Language Processing (NLP) with a focus on the task of Text Classification.

# Table of Contents

<details>

<summary><b>Expand Table of Contents</b></summary><blockquote><p align="justify">

- [Pattern Matching](#pattern-matching)
- [Machine Learning](#machine-learning)
- [Deep Learning](#deep-learning)
- [Semi-supervised Learning](#semi-supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Event Coreference](#event-coreference)
- [Surveys](#surveys)
- [Others](#others)
- [Linguistics](#linguistics)
- [Data](#data)
- [Tools and Repos](#tools-and-repos)
</p></blockquote></details>

---


## Pattern matching


### 1993


<details>
<summary>1. <a href="https://aaai.org/Papers/AAAI/1993/AAAI93-121.pdf">Automatically Constructing a Dictionary for Information Extraction Tasks</a> by<i> Ellen Riloff
</i></summary><blockquote><p align="justify">
Knowledge-based natural language processing systems have achieved good success with certain tasks but they are often criticized because they depend on a domain-specific dictionary that requires a great deal of manual knowledge engineering. This knowledge engineering bottleneck makes knowledge-based NLP systems impractical for real-world applications because they cannot be easily scaled up orported to new domains. In response to this problem, we developed a system called AutoSlog that automatically builds a domain-specific dictionary of concepts for extracting information from text. Using AutoSlog. we constructed a dictionary for the domain of terrorist event descriptions in only 5 person-hours. We then compared the AutoSlog dictionary with a hand-crafted dictionary that was built by two highly skilled graduate students and required approximately 1500 person-hours of effort. We evaluated the two dictionaries using two blind test sets of 100 texts each. Overall, the AutoSlog dictionary achieved 98% of the performance of the hand-crafted dictionary. On the first test set, the Auto-Slog dictionary obtained 96.3% of the perfomlance of the hand-crafted dictionary. On the second test set, the overall scores were virtually indistinguishable with the AutoSlog dictionary achieving 99.7% of the performance of the handcrafted dictionary.
</p></blockquote></details>

### 1995


<details>
<summary>1. <a href="https://ieeexplore.ieee.org/document/469825">Acquisition of linguistic patterns for knowledge-based information extraction</a> by<i>  Jun-Tae Kim ; D.I. Moldovan 
</i></summary><blockquote><p align="justify">
The paper presents an automatic acquisition of linguistic patterns that can be used for knowledge based information extraction from texts. In knowledge based information extraction, linguistic patterns play a central role in the recognition and classification of input texts. Although the knowledge based approach has been proved effective for information extraction on limited domains, there are difficulties in construction of a large number of domain specific linguistic patterns. Manual creation of patterns is time consuming and error prone, even for a small application domain. To solve the scalability and the portability problem, an automatic acquisition of patterns must be provided. We present the PALKA (Parallel Automatic Linguistic Knowledge Acquisition) system that acquires linguistic patterns from a set of domain specific training texts and their desired outputs. A specialized representation of patterns called FP structures has been defined. Patterns are constructed in the form of FP structures from training texts, and the acquired patterns are tuned further through the generalization of semantic constraints. Inductive learning mechanism is applied in the generalization step. The PALKA system has been used to generate patterns for our information extraction system developed for the fourth Message Understanding Conference (MUC-4).
</p></blockquote></details>



<details>
<summary>2. <a href="https://www.aclweb.org/anthology/W95-0112/">Automatically Acquiring Conceptual Patterns without an Annotated Corpus</a> by<i> Ellen Riloff, Jay Shoen </i></summary><blockquote><p align="justify">
Previous work on automated dictionary construction for information extraction has relied on annotated text corpora. However, annotating a corpus is time-consuming and difficult. We propose that conceptual patterns for information extraction can be acquired automatically using only a preclassified training corpus and no text annotations. We describe a system called AutoSlog-TS, which is a variation of our previous AutoSlog system, that runs exhaustively on an untagged text corpus. Text classification experiments in the MUC-4 terrorism domain show that the AutoSlog-TS dictionary performs comparably to a hand-crafted dictionary, and actually achieves higher precision on one test set. For text classification, AutoSlog-TS requires no manual effort beyond the preclassified training corpus. Additional experiments suggest how a dictionary produced by AutoSlog-TS can be filtered automatically for information extraction tasks. Some manual intervention is still required in this case, but AutoSlog-TS significantly reduces the amount of effort required to create an appropriate training corpus.
</p></blockquote></details>

<details>
<summary>3. <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.597.3832&rep=rep1&type=pdf">Learning information extraction patterns from examples</a> by<i> Scott B. Huffman </i></summary><blockquote><p align="justify">
A growing population of users want to extract a growing variety of information from on-line texts. Unfortunately, current information extraction systems typically require experts to hand-build dictionaries of extraction patterns for each new type of information to be extracted. This paper presents a system that can learn dictionaries of extraction patterns directly from user-provided examples of texts and events to be extracted from them. The system, called LIEP, learns patterns that recognize relationships between key constituents based on local syntax. Sets of patterns learned by LIEP for a sample extraction task perform nearly at the level of a hand-built dictionary of patterns.
</p></blockquote></details>

### 1998 

<details>
<summary>1. <a href="https://www.semanticscholar.org/paper/Multistrategy-Learning-for-Information-Extraction-Freitag/29c99d263b5e05aae6bb96f004f025dcc9b5caae">Multistrategy Learning for Information Extraction</a> by<i> Dayne Freitag</i></summary><blockquote><p align="justify">
Information extraction IE is the problem of lling out pre de ned structured sum maries from text documents We are in terested in performing IE in non traditional domains where much of the text is often ungrammatical such as electronic bulletin board posts and Web pages We suggest that the best approach is one that takes into ac count many di erent kinds of information and argue for the suitability of a multistrat egy approach We describe learners for IE drawn from three separate machine learning paradigms rote memorization term space text classi cation and relational rule induc tion By building regression models mapping from learner con dence to probability of cor rectness and combining probabilities appro priately it is possible to improve extraction accuracy over that achieved by any individ ual learner We describe three di erent mul tistrategy approaches Experiments on two IE domains a collection of electronic seminar announcements from a university computer science department and a set of newswire ar ticles describing corporate acquisitions from the Reuters collection demonstrate the effectiveness of all three approaches
</p></blockquote></details>

### 1999 

<details>
<summary>1. <a href="https://www.researchgate.net/publication/221603776_Learning_Dictionaries_for_Information_Extraction_by_Multi-Level_Bootstrapping">Learning Dictionaries for Information Extraction by Multi-Level Bootstrapping</a> by<i> Ellen Riloff, Rosie Jones</i></summary><blockquote><p align="justify">
Information extraction systems usually require two dictionaries: a semantic lexicon and a dictionary of extraction patterns for the domain. We present a multilevel bootstrapping algorithm that generates both the semantic lexicon and extraction patterns simultaneously. As input, our technique requires only unannotated training texts and a handful of seed words for a category. We use a mutual bootstrapping technique to alternately select the best extraction pattern for the category and bootstrap its extractions into the semantic lexicon, which is the basis for selecting the next extraction pattern. To make this approach more robust, we add a second level of bootstrapping (metabootstrapping) that retains only the most reliable lexicon entries produced by mutual bootstrapping and then restarts the process. We evaluated this multilevel bootstrapping technique on a collection of corporate web pages and a corpus of terrorism news articles. The algorithm produced high-quality dictionaries for several semantic categories.
</p></blockquote></details>

### 2000 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/A00-1011/">REES: A Large-Scale Relation and Event Extraction System</a> by<i> Chinatsu Aone, Mila Ramos-Santacruz</i></summary><blockquote><p align="justify">
This paper reports on a large-scale, end-to-end relation and event extraction system. At present, the system extracts a total of 100 types of relations and events, which represents a much wider coverage than is typical of extraction systems. The system consists of three specialized pattem-based tagging modules, a high-precision co-reference resolution module, and a configurable template generation module. We report quantitative evaluation results, analyze the results in detail, and discuss future directions. 
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/C00-2136/">Automatic Acquisition of Domain Knowledge for Information Extraction</a> by<i> Roman Yangarber, Ralph Grishman, Pasi Tapanainen, Silja Huttunen</i></summary><blockquote><p align="justify">
In developing an Information Extraction (IE) system for a new class of events or relations, one of the major tasks is identifying the many ways in which these events or relations may be expressed in text. This has generally involved the manual analysis and, in some cases, the annotation of large quantities of text involving these events. This paper presents an alternative approach, based on an automatic discovery procedure, ExDisco, which identi es a set of relevant documents and a set of event patterns from un-annotated text, starting from a small set of seed patterns." We evaluate ExDisco by comparing the performance of discovered patterns against that of manually constructed systems on actual extraction tasks.
</p></blockquote></details>

### 2001 



<details>
<summary>1. <a href="https://www.semanticscholar.org/paper/Adaptive-Information-Extraction-from-Text-by-Rule-Ciravegna/436087083293ca8728fb96d2e05c011fff2c7751">Adaptive Information Extraction from Text by Rule Induction and Generalisation</a> by<i> Fabio Ciravegna</i></summary><blockquote><p align="justify">
(LP)2 is a covering algorithm for adaptive Information Extraction from text (IE). It induces symbolic rules that insert SGML tags into texts by learning from examples found in a user-defined tagged corpus. Training is performed in two steps: initially a set of tagging rules is learned; then additional rules are induced to correct mistakes and imprecision in tagging. Induction is performed by bottom-up generalization of examples in the training corpus. Shallow knowledge about Natural Language Processing (NLP) is used in the generalization process. The algorithm has a considerable success story. From a scientific point of view, experiments report excellent results with respect to the current state of the art on two publicly available corpora. From an application point of view, a successful industrial IE tool has been based on (LP)2. Real world applications have been developed and licenses have been released to external companies for building other applications. This paper presents (LP)2, experimental results and applications, and discusses the role of shallow NLP in rule induction. 
</p></blockquote></details>

### 2002 



<details>
<summary>1. <a href="https://link.springer.com/chapter/10.1007/3-540-36182-0_30">Event Pattern Discovery from the Stock Market Bulletin</a> by<i>      Fang Li, Huanye Sheng, Dongmo Zhang</i></summary><blockquote><p align="justify">
Electronic information grows rapidly as the Internet is widely used in our daily life. In order to identify the exact information for the user query, information extraction is widely researched and investigated. The template, which pertains to events or situations, and contains slots that denote who did what to whom, when, and where, is predefined by a template builder. Therefore, fixed templates are the main obstacles for the information extraction system out of the laboratory. In this paper, a method to automatically discover the event pattern in Chinese from stock market bulletin is introduced. It is based on the tagged corpus and the domain model. The pattern discovery process is independent of the domain model by introducing a link table. The table is the connection between text surface structure and semantic deep structure represented by a domain model. The method can be easily adapted to other domains by changing the link table.
</p></blockquote></details>

### 2003 


<details>
<summary>1. <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.9396&rep=rep1&type=pdf">A System for new event detection</a> by<i> Thorsten Brants, Francine Chen, Ayman  Farahat</i></summary><blockquote><p align="justify">
We present a new method and system for performing the New Event Detection task, i.e., in one or multiple streams of news stories, all stories on a previously unseen (new) event are marked. The method is based on an incremental TF-IDF model. Our extensions include: generation of source-specific models, similarity score normalization based on document-specific averages, similarity score normalization based on source-pair specific averages, term reweighting based on inverse event frequencies, and segmentation of the documents. We also report on extensions that did not improve results. The system performs very well on TDT3 and TDT4 test data and scored second in the TDT-2002 evaluation.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.researchgate.net/publication/220321036_Bottom-Up_Relational_Learning_of_Pattern_Matching_Rules_for_Information_Extraction">Bottom-Up Relational Learning of Pattern Matching Rules for Information Extraction.</a> by<i>     Mary Elaine Califf, Raymond J. Mooney</i></summary><blockquote><p align="justify">
Information extraction is a form of shallow text processing that locates a specified set of relevant items in a natural-language document. Systems for this task require significant domain-specific knowledge and are time-consuming and difficult to build by hand, making them a good application for machine learning. We present an algorithm, RAPIER, that uses pairs of sample documents and filled templates to induce pattern-match rules that directly extract fillers for the slots in the template. RAPIER is a bottom-up learning algorithm that incorporates techniques from several inductive logic programming systems. We have implemented the algorithm in a system that allows patterns to have constraints on the words, part-of-speech tags, and semantic classes present in the filler and the surrounding text. We present encouraging experimental results on two domains.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/P03-1029/">An Improved Extraction Pattern Representation Model for Automatic IE Pattern Acquisition</a> by<i> Kiyoshi Sudo, Satoshi Sekine, Ralph Grishman</i></summary><blockquote><p align="justify">
Several approaches have been described for the automatic unsupervised acquisition of patterns for information extraction. Each approach is based on a particular model for the patterns to be acquired, such as a predicate-argument structure or a dependency chain. The effect of these alternative models has not been previously studied. In this paper, we compare the prior models and introduce a new model, the Subtree model, based on arbitrary subtrees of dependency trees. We describe a discovery procedure for this model and demonstrate experimentally an improvement in recall using Subtree patterns.
</p></blockquote></details>



### 2005

<details>
<summary>1. <a href="https://www.aaai.org/Papers/Workshops/2006/WS-06-07/WS06-07-004.pdf">Automatic event and relation detection with seeds of varying complexity</a> by<i> Feiyu Xu, Hans Uszkoreit and Hong Li</i></summary><blockquote><p align="justify">
In this paper, we present an approach for automatically detecting events in natural language texts by learning patterns that signal the mentioning of such events.  We construe the relevant event types as relations and start with aset of seeds consisting of representative event instances thath appen to be known and also to be mentioned frequently in easily available training data.  Methods have been developed for the automatic identification of event extents andevent triggers.  We have learned patterns for a particular domain,  i.e.,  prize  award  events.  Currently  we  are systematically investigating the criteria for selecting the most effective patterns for the detection of events in sentences  and  paragraphs.  Although  the  systematic investigation is still under way, we can already report on first very promising results of the method for learning of patterns and for using these patterns in event detection.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.semanticscholar.org/paper/A-Semantic-Approach-to-IE-Pattern-Induction-Stevenson-Greenwood/f30b903047284e8a253b2da38530b99b6db13317">A Semantic Approach to IE Pattern Induction</a> by<i>     Mark Stevenson, Mark A. Greenwood</i></summary><blockquote><p align="justify">
This paper presents a novel algorithm for the acquisition of Information Extraction patterns. The approach makes the assumption that useful patterns will have similar meanings to those already identified as relevant. Patterns are compared using a variation of the standard vector space model in which information from an ontology is used to capture semantic similarity. Evaluation shows this algorithm performs well when compared with a previously reported document-centric approach. 
</p></blockquote></details>


### 2008

<details>
<summary>1. <a href="https://link.springer.com/chapter/10.1007/978-3-540-69858-6_21">Real-Time News Event Extraction for Global Crisis Monitoring</a> by<i> Hristo Tanev, Jakub Piskorski, Martin Atkinson</i></summary><blockquote><p align="justify">
This paper presents a real-time news event extraction system developed by the Joint Research Centre of the European Commission. It is capable of accurately and efficiently extracting violent and disaster events from online news without using much linguistic sophistication. In particular, in our linguistically relatively lightweight approach to event extraction, clustered news have been heavily exploited at various stages of processing. The paper describes the system’s architecture, news geo-tagging, automatic pattern learning, pattern specification language, information aggregation, the issues of integrating event information in a global crisis monitoring system and new experimental evaluation.
</p></blockquote></details>


### 2009 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W09-1405/">Biomedical event extraction without training data</a> by<i> Andreas Vlachos, Paula Buttery, Diarmuid Ó Séaghdha, Ted Briscoe </i></summary><blockquote><p align="justify">
We describe our system for the BioNLP 2009 event detection task. It is designed to be as domain-independent and unsupervised as possible. Nevertheless, the precisions achieved for single theme event classes range from 75% to 92%, while maintaining reasonable recall. The overall F-scores achieved were 36.44% and 30.80% on the development and the test sets respectively.
</p></blockquote></details> 

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/W09-1418/">Syntactic dependency based heuristics for biological event extraction</a> by<i> Halil Kilicoglu, Sabine Bergler</i></summary><blockquote><p align="justify">
We explore a rule-based methodology for the BioNLP'09 Shared Task on Event Extraction, using dependency parsing as the underlying principle for extracting and characterizing events. We approach the speculation and negation detection task with the same principle. Evaluation results demonstrate the utility of this syntax-based approach and point out some shortcomings that need to be addressed in future work.
</p></blockquote></details>


### 2010 

<details>
<summary>1. <a href="https://personal.eur.nl/frasincar/papers/IJWET2010/ijwet2010.pdf">Semi-Automatic Financial Events Discovery Based on Lexico-Semantic Patterns</a> by<i> Jethro Borsje, Frederik Hogenboom, Flavius Frasincar </i></summary><blockquote><p align="justify">
Due to the market sensitivity to emerging news, investors on financial markets need to continuously monitor financial events when deciding on buying and selling equities. We propose the use of lexico-semantic patterns for financial event extraction from RSS news feeds. These patterns use financial ontologies, leveraging the commonly used lexico-syntactic patterns to a higher abstraction level, thereby enabling lexico-semantic patterns to identify more and more precisely events than lexico-syntactic patterns from text. We have developed rules based on lexico-semantic patterns used to find events, and semantic actions that allow for updating the domain ontology with the effects of the discovered events. Both the lexico-semantic patterns and the semantic actions make use of the triple paradigm that fosters their easy construction and understanding by the user. Based on precision, recall, and F1 measures, we show the effectiveness of the proposed approach.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/C10-1077/">Filtered Ranking for Bootstrapping in Event Extraction</a> by<i> Shasha Liao, Ralph Grishman</i></summary><blockquote><p align="justify">
Several researchers have proposed semi-supervised learning methods for adapting event extraction systems to new event types. This paper investigates two kinds of bootstrapping methods used for event extraction: the document-centric and similarity-centric approaches, and proposes a filtered ranking method that combines the advantages of the two. We use a range of extraction tasks to compare the generality of this method to previous work. We analyze the results using two evaluation metrics and observe the effect of different training corpora. Experiments show that our new ranking method not only achieves higher performance on different evaluation metrics, but also is more stable across different bootstrapping corpora.
</p></blockquote></details>


### 2011

<details>
<summary>1. <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8640.2011.00401.x">Effective bio-event extraction using trigger words and syntactic dependencies</a> by<i> Halil Kilicoglu, Sabine Bergler
</i></summary><blockquote><p align="justify">
The scientific literature is the main source for comprehensive, up-to-date biological knowledge. Automatic extraction of this knowledge facilitates core biological tasks, such as database curation and knowledge discovery. We present here a linguistically inspired, rule-based and syntax-driven methodology for biological event extraction. We rely on a dictionary of trigger words to detect and characterize event expressions and syntactic dependency based heuristics to extract their event arguments. We refine and extend our prior work to recognize speculated and negated events. We show that heuristics based on syntactic dependencies, used to identify event arguments, extend naturally to also identify speculation and negation scope. In the BioNLP’09 Shared Task on Event Extraction, our system placed third in the Core Event Extraction Task (F-score of 0.4462), and first in the Speculation and Negation Task (F-score of 0.4252). Of particular interest is the extraction of complex regulatory events, where it scored second place. Our system significantly outperformed other participating systems in detecting speculation and negation. These results demonstrate the utility of a syntax-driven approach. In this article, we also report on our more recent work on supervised learning of event trigger expressions and discuss event annotation issues, based on our corpus analysis.
</p></blockquote></details>


### 2012 

<details>
<summary>1. <a href="https://link.springer.com/chapter/10.1007%2F978-3-642-33185-5_10">Ontology-Based Information and Event Extraction for Business Intelligence</a> by<i> Ernest Arendarenko, Tuomo Kakkonen</i></summary><blockquote><p align="justify">
We would like to introduce BEECON, an information and event extraction system for business intelligence. This is the first ontology-based system for business documents analysis that is able to detect 41 different types of business events from unstructured sources of information. The described system is intended to enhance business intelligence efficiency by automatically extracting relevant content such as business entities and events. In order to achieve it, we use natural language processing techniques, pattern recognition algorithms and hand-written detection rules. In our test set consisting of 190 documents with 550 events, the system achieved 95% precision and 67% recall in detecting all supported business event types from newspaper texts.
</p></blockquote></details>

<details>
<summary>2. <a href="https://ieeexplore.ieee.org/document/6299414">A Real -- Time News Event Extraction Framework for Vietnamese</a> by<i>  Mai-Vu Tran , Minh-Hoang Nguyen , Sy-Quan Nguyen , Minh-Tien Nguyen, Xuan-Hieu Phan</i></summary><blockquote><p align="justify">
Event Extraction is a complex and interesting topic in Information Extraction that includes event extraction methods from free text or web data. The result of event extraction systems can be used in several fields such as risk analysis systems, online monitoring systems or decide support tools. In this paper, we introduce a method that combines lexico -- semantic and machine learning to extract event from Vietnamese news. Furthermore, we concentrate to describe event online monitoring system named VnLoc based on the method that was proposed above to extract event in Vietnamese language. Besides, in experiment phase, we have evaluated this method based on precision, recall and F1 measure. At this time of experiment, we on investigated on three types of event: FIRE, CRIME and TRANSPORT ACCIDENT.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/E12-1029/">Bootstrapped Training of Event Extraction Classifiers</a> by<i>  Ruihong Huang, Ellen Riloff</i></summary><blockquote><p align="justify">
Most event extraction systems are trained with supervised learning and rely on a collection of annotated documents. Due to the domain-specificity of this task, event extraction systems must be retrained with new annotated data for each domain. In this paper, we propose a bootstrapping solution for event role filler extraction that requires minimal human supervision. We aim to rapidly train a state-of-the-art event extraction system using a small set of "seed nouns" for each event role, a collection of relevant (in-domain) and irrelevant (out-of-domain) texts, and a semantic dictionary. The experimental results show that the bootstrapped system outperforms previous weakly supervised event extraction systems on the MUC-4 data set, and achieves performance levels comparable to supervised training with 700 manually annotated documents.
</p></blockquote></details>

### 2015  

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P15-4022/">A Domain-independent Rule-based Framework for Event Extraction</a> by<i> Marco A. Valenzuela-Escárcega, Gus Hahn-Powell, Mihai Surdeanu, Thomas Hicks</i></summary><blockquote><p align="justify">
We describe the design, development, and API of ODIN (Open Domain INformer), a domain- independent, rule-based event extraction (EE) framework. The proposed EE approach is: simple (most events are captured with simple lexico-syntactic patterns), powerful (the language can capture complex constructs, such as events taking other events as arguments, and regular expressions over syntactic graphs), robust (to recover from syntactic parsing errors, syntactic patterns can be freely mixed with surface, token-based patterns), and fast (the runtime environment processes 110 sentences/second in a real-world domain with a grammar of over 200 rules). We used this framework to develop a grammar for the bio-chemical domain, which approached human performance. Our EE framework is accompanied by a web-based user interface for the rapid development of event grammars and visualization of matches. The ODIN framework and the domain-specific grammars are available as open-source code.
</p></blockquote></details> 

<details>
<summary>2. <a href="https://dl.acm.org/citation.cfm?doid=3008658.2994600">Minimally Supervised Chinese Event Extraction from Multiple Views</a> by<i> Peifeng Li , Guodong Zhou, Qiaoming Zhu</i></summary><blockquote><p align="justify">
Although several semi-supervised learning models have been proposed for English event extraction, there are few successful stories in Chinese due to its special characteristics. In this article, we propose a novel minimally supervised model for Chinese event extraction from multiple views. Besides the traditional pattern similarity view (PSV), a semantic relationship view (SRV) is introduced to capture the relevant event mentions from relevant documents. Moreover, a morphological structure view (MSV) is incorporated to both infer more positive patterns and help filter negative patterns via morphological structure similarity. An evaluation of the ACE 2005 Chinese corpus shows that our minimally supervised model significantly outperforms several strong baselines.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/R15-1010/">Improving Event Detection with Active Learning</a> by<i> Kai Cao, Xiang Li, Miao Fan, Ralph Grishman</i></summary><blockquote><p align="justify">
Event Detection (ED), one aspect of Information Extraction, involves identifying instances of specified types of events in text. Much of the research on ED has been based on the specifications of the 2005 ACE [Automatic Content Extraction] event task 1 , and the associated annotated corpus. However, as the event instances in the ACE corpus are not evenly distributed, some frequent expressions involving ACE events do not appear in the training data, adversely affecting performance. In this paper, we demonstrate the effectiveness of a Pattern Expansion technique to import frequent patterns extracted from external corpora to boost ED performance. The experimental results show that our pattern-based system with the expanded patterns can achieve 70.4% (with 1.6% absolute improvement) F-measure over the baseline, an advance over current state-of-the-art systems.
</p></blockquote></details>

### 2018 

<details>
<summary>1. <a href="https://pdfs.semanticscholar.org/dc04/f814fb210edf62a6237c52eb88ac98a5b732.pdf">Including new patterns to improve event extraction systems</a> by<i> Kai Cao,Xiang Li,Weicheng Ma,Ralph Grishman</i></summary><blockquote><p align="justify">
Event Extraction (EE) is a challenging Information Extraction task which aims to discover event triggers of specific types along with their arguments. Most recent research on Event Extraction relies on pattern-based or feature-based approaches, trained on annotated corpora, to recognize combi- nations of event triggers, arguments, and other contextual in- formation. However, as the event instances in the ACE corpus are not evenly distributed, some frequent expressions involving ACE event triggers do not appear in the training data, adversely affecting the performance. In this paper, we demon- strate the effectiveness of systematically importing expert-level patterns from TABARI to boost EE performance. The experimental results demonstrate that our pattern-based sys- tem with the expanded patterns can achieve 69.8% (with 1.9% absolute improvement) F-measure over the baseline, an advance over current state-of-the-art systems.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.researchgate.net/publication/329735770_Rule_based_Event_Extraction_System_from_Newswires_and_Social_Media_Text_in_Indian_Languages_EventXtract-IL_for_English_and_Hindi_Data">Rule Based Event Extraction System from Newswires and Social Media Text in Indian Languages (EventXtract-IL) for English and Hindi data</a> by<i> Anita Saroj, Rajesh kumar Munodtiya, and Sukomal Pal </i></summary><blockquote><p align="justify">
Due to today’s information overload, the user is particularly finding it difficult to access the right information through the World Wide Web. The situation becomes worse when this information is in multiple languages. In this paper we present a model for information extraction. Our model mainly works on the concept of speech tagging and named entity recognization. We represent each word with the POS tag and the entity identified for that term. We assume that the event exists in the first line of the document. If we do not find it in the first line, then we take the help of emotion analysis. If it has negative polarity, then it is associated with an unexpected event which has negative meaning. We use NLTK for emotion analysis.
</p></blockquote></details>



## Machine learning 
[:arrow_up:](#table-of-contents)
### 2006 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W06-0901/">The stages of event extraction</a> by<i> David Ahn</i></summary><blockquote><p align="justify">
Event detection and recognition is a complex task consisting of multiple sub-tasks of varying difficulty. In this paper, we present a simple, modular approach to event extraction that allows us to experiment with a variety of machine learning methods for these sub-tasks, as well as to evaluate the impact on performance these sub-tasks have on the overall task. 
</p></blockquote></details>

### 2007 

<details>
<summary>1. <a href="https://link.springer.com/chapter/10.1007/978-3-540-72035-5_22">Extracting Violent Events From On-Line News for Ontology Population</a> by<i> Jakub Piskorski, Hristo Tanev, Pinar Oezden Wennerberg</i></summary><blockquote><p align="justify">
This paper presents nexus, an event extraction system, developed at the Joint Research Center of the European Commission utilized for populating violent incident knowledge bases. It automatically extracts security-related facts from on-line news articles. In particular, the paper focuses on a novel bootstrapping algorithm for weakly supervised acquisition of extraction patterns from clustered news, cluster-level information fusion and pattern specification language. Finally, a preliminary evaluation of nexus on real-world data is given which revealed acceptable precision and a strong application potential.
</p></blockquote></details>

### 2008

<details>
<summary>1. <a href="https://www.researchgate.net/publication/255646580_Research_on_Chinese_Event_Extraction">Research on Chinese Event Extraction</a> by<i> Yanyan Zhao, Bing Qin, Wanxiang Che, Ting Liu</i></summary><blockquote><p align="justify">
Event Extraction is an important research point in the area of Information Extraction. This paper makes an intensive study of the two stages of Chinese event extraction, namely event type recognition and event argument recognition. A novel method combining event trigger expansion and a binary classifier is presented in the step of event type recognition while in the step of argument recognition, one with multi-class classification based on maximum entropy is introduced. The above methods solved the data unbalanced problem in training model and the data sparseness problem brought by the small set of training data effectively, and finally our event extraction system achieved a better performance.
</p></blockquote></details> 

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P08-1030/">Refining Event Extraction through Cross-Document Inference</a> by<i> Heng Ji, Ralph Grishman</i></summary><blockquote><p align="justify">
We apply the hypothesis of "One Sense Per Discourse" (Yarowsky, 1995) to information extraction (IE), and extend the scope of "discourse" from one single document to a cluster of topically-related documents. We employ a similar approach to propagate consistent event arguments across sentences and documents. Combining global evidence from related documents with local decisions, we design a simple scheme to conduct cross-document inference for improving the ACE event extraction task 1 . Without using any additional labeled data this new approach obtained 7.6% higher F-Measure in trigger labeling and 6% higher F-Measure in argument labeling over a state-of-the-art IE system which extracts events independently for each sentence.
</p></blockquote></details>

### 2009

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W09-1402/">Extracting Complex Biological Events with Rich Graph-Based Feature Sets</a> by<i> Jari Björne, Juho Heimonen, Filip Ginter, Antti Airola, Tapio Pahikkala, Tapio Salakoski</i></summary><blockquote><p align="justify">
We describe a system for extracting complex events among genes and proteins from biomedical literature, developed in context of the BioNLP’09 Shared Task on Event Extraction. For each event, the system extracts its text trigger, class, and arguments. In contrast to the approaches prevailing prior to the shared task, events can be arguments of other events, resulting in a nested structure that better captures the underlying biological statements. We divide the task into independent steps which we approach as machine learning problems. We define a wide array of features and in particular make extensive use of dependency parse graphs. A rule‐based postprocessing step is used to refine the output in accordance with the restrictions of the extraction task. In the shared task evaluation, the system achieved an F‐score of 51.95% on the primary task, the best performance among the participants. Currently, with modifications and improvements described in this article, the system achieves 52.86% F‐score on Task 1, the primary task, improving on its original performance. In addition, we extend the system also to Tasks 2 and 3, gaining F‐scores of 51.28% and 50.18%, respectively. The system thus addresses the BioNLP’09 Shared Task in its entirety and achieves the best performance on all three subtasks.
</p></blockquote></details>

<details>
<summary>2. <a href="https://pdfs.semanticscholar.org/b6ce/13412a9cadb6c57f1349ad389affbdea2321.pdf">Language Specific Issue and Feature Exploration in  Chinese Event Extraction</a> by<i> Zheng Chen, Heng Ji</i></summary><blockquote><p align="justify">
In this paper, we present a Chinese event extraction system. We point out a language spe- cific issue in Chinese trigger labeling, and then commit to discussing the contributions of lexical, syntactic and semantic features applied in trigger labeling and argument labeling. As a result, we achieved competitive performance, specifically, F-measure of 59.9 in trigger labeling and F-measure of 43.8 in argument labeling.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/W09-1406/">A Markov Logic Approach to Bio-Molecular Event Extraction</a> by<i> Sebastian Riedel, Hong-Woo Chun, Toshihisa Takagi, Jun’ichi Tsujii </i></summary><blockquote><p align="justify">
In this paper we describe our entry to the BioNLP 2009 Shared Task regarding biomolecular event extraction. Our work can be described by three design decisions: (1) instead of building a pipeline using local classifier technology, we design and learn a joint probabilistic model over events in a sentence; (2) instead of developing specific inference and learning algorithms for our joint model, we apply Markov Logic, a general purpose Statistical Relation Learning language, for this task; (3) we represent events as relational structures over the tokens of a sentence, as opposed to structures that explicitly mention abstract event entities. Our results are competitive: we achieve the 4th best scores for task 1 (in close range to the 3rd place) and the best results for task 2 with a 13 percent point margin.
</p></blockquote></details>

### 2010 

<details>
<summary>1. <a href="https://www.worldscientific.com/doi/abs/10.1142/S0219720010004586">Event Extraction with Complex Event Classification Using Rich Features</a> by<i> MAKOTO MIWA, RUNE SÆTRE, JIN-DONG KIM and JUN'ICHI TSUJII </i></summary><blockquote><p align="justify">
Biomedical Natural Language Processing (BioNLP) attempts to capture biomedical phenomena from texts by extracting relations between biomedical entities (i.e. proteins andgenes).  Traditionally,  only binary relations  have  been  extracted  from large numbers of published papers. Recently, more complex relations (biomolecular events) have also been extracted. Such events may include several entities or other relations. To evaluate the performance of the text mining systems, several shared task challenges have been arranged for the BioNLP community. With a common and consistent task setting, theBioNLP’09 shared task evaluated complex biomolecular events such as binding and regulation. Finding these events automatically is important in order to improve biomedical event extraction systems. In the present paper, we propose an automatic event extraction system, which contains a model for complex events, by solving a classification problem with rich features. The main contributions of the present paper are: (1) the proposal of an effective bio-event detection method using machine learning, (2) provision of a high-performance event extraction system, and (3) the execution of a quantitative error analysis. The proposed complex (binding and regulation) event detector outperforms the best system from the BioNLP’09 shared task challenge.
</p></blockquote></details>

<details>
<summary>2. <a href="https://academic.oup.com/bioinformatics/article/26/12/i382/282442">Complex event extraction at PubMed scale</a> by<i> Björne J, Ginter F, Pyysalo S, Tsujii J, Salakoski T.</i></summary><blockquote><p align="justify">
There has recently been a notable shift in biomedical information extraction (IE) from relation models toward the more expressive event model, facilitated by the maturation of basic tools for biomedical text analysis and the availability of manually annotated resources. The event model allows detailed representation of complex natural language statements and can support a number of advanced text mining applications ranging from semantic search to pathway extraction. A recent collaborative evaluation demonstrated the potential of event extraction systems, yet there have so far been no studies of the generalization ability of the systems nor the feasibility of large-scale extraction. This study considers event-based IE at PubMed scale. We introduce a system combining publicly available, state-of-the-art methods for domain parsing, named entity recognition and event extraction, and test the system on a representative 1% sample of all PubMed citations. We present the first evaluation of the generalization performance of event extraction systems to this scale and show that despite its computational complexity, event extraction from the entire PubMed is feasible. We further illustrate the value of the extraction approach through a number of analyses of the extracted information. The event detection system and extracted data are open source licensed and available at http://bionlp.utu.fi/.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/P10-1081/">Using Document Level Cross-Event Inference to Improve Event Extraction</a> by<i> Shasha Liao, Ralph Grishman</i></summary><blockquote><p align="justify">
Event extraction is a particularly challenging type of information extraction (IE). Most current event extraction systems rely on local information at the phrase or sentence level. However, this local context may be insufficient to resolve ambiguities in identifying particular types of events; information from a wider scope can serve to resolve some of these ambiguities. In this paper, we use document level information to improve the performance of ACE event extraction. In contrast to previous work, we do not limit ourselves to information about events of the same type, but rather use information about other types of events to make predictions or resolve ambiguities regarding a given event. We learn such relationships from the training corpus and use them to help predict the occurrence of events and event arguments in a text. Experiments show that we can get 9.0% (absolute) gain in trigger (event) classification, and more than 8% gain for argument (role) classification in ACE event extraction.
</p></blockquote></details>

### 2011

<details>
<summary>1. <a href="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-S2-S4">Word sense disambiguation for event trigger word detection in biomedicine</a> by<i> David Martinez , Timothy Baldwin </i></summary><blockquote><p align="justify">
This paper describes a method for detecting event trigger words in biomedical text based on a word sense disambiguation (WSD) approach. We first investigate the applicability of existing WSD techniques to trigger word disambiguation in the BioNLP 2009 shared task data, and find that we are able to outperform a traditional CRF-based approach for certain word types. On the basis of this finding, we combine the WSD approach with the CRF, and obtain significant improvements over the standalone CRF, gaining particularly in recall.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P11-1113/">Using Cross-Entity Inference to Improve Event Extraction</a> by<i> Yu Hong, Jianfeng Zhang, Bin Ma, Jianmin Yao, Guodong Zhou, Qiaoming Zhu</i></summary><blockquote><p align="justify">
Event extraction is the task of detecting certain specified types of events that are mentioned in the source language data. The state-of-the-art research on the task is transductive inference (e.g. cross-event inference). In this paper, we propose a new method of event extraction by well using cross-entity inference. In contrast to previous inference methods, we regard entity-type consistency as key feature to predict event mentions. We adopt this inference method to improve the traditional sentence-level event extraction system. Experiments show that we can get 8.6% gain in trigger (event) identification, and more than 11.8% gain for argument (role) classification in ACE event extraction.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/R11-1002/">Acquiring Topic Features to improve Event Extraction: in Pre-selected and Balanced Collections</a> by<i> Shasha Liao, Ralph Grishman</i></summary><blockquote><p align="justify">
Event extraction is a particularly challenging type of information extraction (IE) that may require inferences from the whole article. However, most current event extraction systems rely on local information at the phrase or sentence level, and do not consider the article as a whole, thus limiting extraction performance. Moreover, most annotated corpora are artificially enriched to include enough positive samples of the events of interest; event identification on a more balanced collection, such as unfiltered newswire, may perform much worse. In this paper, we investigate the use of unsupervised topic models to extract topic features to improve event extraction both on test data similar to training data, and on more balanced collections. We compare this unsupervised approach to a supervised multi-label text classifier, and show that unsupervised topic modeling can get better results for both collections, and especially for a more balanced collection. We show that the unsupervised topic model can improve trigger, argument and role labeling by 3.5%, 6.9% and 6% respectively on a pre-selected corpus, and by 16.8%, 12.5% and 12.7% on a balanced corpus.
</p></blockquote></details>

<details>
<summary>4. <a href="https://www.aclweb.org/anthology/D11-1001/">Fast and Robust Joint Models for Biomedical Event Extraction</a> by<i> Sebastian Riedel, Andrew McCallum</i></summary><blockquote><p align="justify">
Extracting biomedical events from literature has attracted much recent attention. The best-performing systems so far have been pipelines of simple subtask-specific local classifiers. A natural drawback of such approaches are cascading errors introduced in early stages of the pipeline. We present three joint models of increasing complexity designed to overcome this problem. The first model performs joint trigger and argument extraction, and lends itself to a simple, efficient and exact inference algorithm. The second model captures correlations between events, while the third model ensures consistency between arguments of the same event. Inference in these models is kept tractable through dual decomposition. The first two models outperform the previous best joint approaches and are very competitive with respect to the current state-of-the-art. The third model yields the best results reported so far on the BioNLP 2009 shared task, the BioNLP 2011 Genia task and the BioNLP 2011 Infectious Diseases task.
</p></blockquote></details>

<details>
<summary>5. <a href="https://jbiomedsem.biomedcentral.com/articles/10.1186/2041-1480-2-S5-S6">Coreference based event-argument relation extraction on biomedical text</a> by<i> Katsumasa Yoshikawa, Sebastian Riedel, Tsutomu Hirao, Masayuki Asahara & Yuji Matsumoto</i></summary><blockquote><p align="justify">
This paper presents a new approach to exploit coreference information for extracting event-argument (E-A) relations from biomedical documents. This approach has two advantages: (1) it can extract a large number of valuable E-A relations based on the concept of salience in discourse; (2) it enables us to identify E-A relations over sentence boundaries (cross-links) using transitivity of coreference relations. We propose two coreference-based models: a pipeline based on Support Vector Machine (SVM) classifiers, and a joint Markov Logic Network (MLN). We show the effectiveness of these models on a biomedical event corpus. Both models outperform the systems that do not use coreference information. When the two proposed models are compared to each other, joint MLN outperforms pipeline SVM with gold coreference information.
</p></blockquote></details> 

<details>
<summary>6. <a href="https://www.aclweb.org/anthology/W11-1805/">Biomedical Event Extraction from Abstracts and Full Papers using Search-based Structured Prediction</a> by<i> Andreas Vlachos, Mark Craven</i></summary><blockquote><p align="justify">
Biomedical event extraction has attracted substantial attention as it can assist researchers in understanding the plethora of interactions among genes that are described in publications in molecular biology. While most recent work has focused on abstracts, the BioNLP 2011 shared task evaluated the submitted systems on both abstracts and full papers. In this article, we describe our submission to the shared task which decomposes event extraction into a set of classification tasks that can be learned either independently or jointly using the search-based structured prediction framework. Our intention is to explore how these two learning paradigms compare in the context of the shared task. We report that models learned using search-based structured prediction exceed the accuracy of independently learned classifiers by 8.3 points in F-score, with the gains being more pronounced on the more complex Regulation events (13.23 points). Furthermore, we show how the trade-off between recall and precision can be adjusted in both learning paradigms and that search-based structured prediction achieves better recall at all precision points. Finally, we report on experiments with a simple domain-adaptation method, resulting in the second-best performance achieved by a single system. We demonstrate that joint inference using the search-based structured prediction framework can achieve better performance than independently learned classifiers, thus demonstrating the potential of this learning paradigm for event extraction and other similarly complex information-extraction tasks.
</p></blockquote></details> 

<details>
<summary>7. <a href="https://www.sciencedirect.com/science/article/pii/S0933365711001060?via%3Dihub">Biomedical events extraction using the hidden vector state model</a> by<i> Deyu Zhou, Yulan He</i></summary><blockquote><p align="justify">
Biomedical events extraction concerns about events describing changes on the state of bio-molecules from literature. Comparing to the protein-protein interactions (PPIs) extraction task which often only involves the extraction of binary relations between two proteins, biomedical events extraction is much harder since it needs to deal with complex events consisting of embedded or hierarchical relations among proteins, events, and their textual triggers. In this paper, we propose an information extraction system based on the hidden vector state (HVS) model, called HVS-BioEvent, for biomedical events extraction, and investigate its capability in extracting complex events. HVS has been previously employed for extracting PPIs. In HVS-BioEvent, we propose an automated way to generate abstract annotations for HVS training and further propose novel machine learning approaches for event trigger words identification, and for biomedical events extraction from the HVS parse results. Our proposed system achieves an F-score of 49.57% on the corpus used in the BioNLP'09 shared task, which is only 2.38% lower than the best performing system by UTurku in the BioNLP'09 shared task. Nevertheless, HVS-BioEvent outperforms UTurku's system on complex events extraction with 36.57% vs. 30.52% being achieved for extracting regulation events, and 40.61% vs. 38.99% for negative regulation events. The results suggest that the HVS model with the hierarchical hidden state structure is indeed more suitable for complex event extraction since it could naturally model embedded structural context in sentences.
</p></blockquote></details>

<details>
<summary>8. <a href="https://www.aclweb.org/anthology/P11-1163/">Event Extraction as Dependency Parsing</a> by<i> David McClosky, Mihai Surdeanu, Christopher Manning</i></summary><blockquote><p align="justify">


Nested event structures are a common occurrence in both open domain and domain specific extraction tasks, e.g., a "crime" event can cause a "investigation" event, which can lead to an "arrest" event. However, most current approaches address event extraction with highly local models that extract each event and argument independently. We propose a simple approach for the extraction of such structures by taking the tree of event-argument relations and using it directly as the representation in a reranking dependency parser. This provides a simple framework that captures global properties of both nested and flat event structures. We explore a rich feature space that models both the events to be parsed and context from the original supporting text. Our approach obtains competitive results in the extraction of biomedical events from the BioNLP'09 shared task with a F1 score of 53.5% in development and 48.6% in testing.

</p></blockquote></details> 

<details>
<summary>9. <a href="https://www.aclweb.org/anthology/W11-1807/">Robust Biomedical Event Extraction with Dual Decomposition and Minimal Domain Adaptation</a> by<i> Sebastian Riedel, Andrew McCallum</i></summary><blockquote><p align="justify">
We present a joint model for biomedical event extraction and apply it to four tracks of the BioNLP 2011 Shared Task. Our model decomposes into three sub-models that concern (a) event triggers and outgoing arguments, (b) event triggers and incoming arguments and (c) protein-protein bindings. For efficient decoding we employ dual decomposition. Our results are very competitive: With minimal adaptation of our model we come in second for two of the tasks---right behind a version of the system presented here that includes predictions of the Stanford event extractor as features. We also show that for the Infectious Diseases task using data from the Genia track is a very effective way to improve accuracy.
</p></blockquote></details>

<details>
<summary>10. <a href="https://www.aaai.org/Papers/AAAI/2002/AAAI02-118.pdf">A maximum entropy approach to information extraction from semi-structured and free text</a> by<i> Hai Leong Chieu, Hwee Tou Ng</i></summary><blockquote><p align="justify">
In this paper, we present a classification-based approach towards single-slot as well as multi-slot information extraction (IE). For single-slot IE, we worked on the domain of Seminar Announcements, where each document contains information on only one seminar. For multi-slot IE, we worked on the domain of Management Succession. For this domain, we restrict ourselves to extracting information sentence by sentence, in the same way as (Soderland 1999). Each sentence can contain information on several management succession events. By using a classification approach based on a maximum entropy framework, our system achieves higher accuracy than the best previously published results in both domains.
</p></blockquote></details>

### 2012 
 

<details>
<summary>1. <a href="https://academic.oup.com/bioinformatics/article/28/13/1759/234417">Boosting automatic event extraction from the literature using domain adaptation and coreference resolution</a> by<i> Makoto Miwa, Paul Thompson, Sophia Ananiadou</i></summary><blockquote><p align="justify">
In recent years, several biomedical event extraction (EE) systems have been developed. However, the nature of the annotated training corpora, as well as the training process itself, can limit the performance levels of the trained EE systems. In particular, most event-annotated corpora do not deal adequately with coreference. This impacts on the trained systems' ability to recognize biomedical entities, thus affecting their performance in extracting events accurately. Additionally, the fact that most EE systems are trained on a single annotated corpus further restricts their coverage. We have enhanced our existing EE system, EventMine, in two ways. First, we developed a new coreference resolution (CR) system and integrated it with EventMine. The standalone performance of our CR system in resolving anaphoric references to proteins is considerably higher than the best ranked system in the COREF subtask of the BioNLP'11 Shared Task. Secondly, the improved EventMine incorporates domain adaptation (DA) methods, which extend EE coverage by allowing several different annotated corpora to be used during training. Combined with a novel set of methods to increase the generality and efficiency of EventMine, the integration of both CR and DA have resulted in significant improvements in EE, ranging between 0.5% and 3.4% F-Score. The enhanced EventMine outperforms the highest ranked systems from the BioNLP'09 shared task, and from the GENIA and Infectious Diseases subtasks of the BioNLP'11 shared task. The improved version of EventMine, incorporating the CR system and DA methods, is available at: http://www.nactem.ac.uk/EventMine/.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/D12-1092/">Employing Compositional Semantics and Discourse Consistency in Chinese Event Extraction</a> by<i> Peifeng Li, Guodong Zhou, Qiaoming Zhu, Libin Hou </i></summary><blockquote><p align="justify">
Current Chinese event extraction systems suffer much from two problems in trigger identification: unknown triggers and word segmentation errors to known triggers. To resolve these problems, this paper proposes two novel inference mechanisms to explore special characteristics in Chinese via compositional semantics inside Chinese triggers and discourse consistency between Chinese trigger mentions. Evaluation on the ACE 2005 Chinese corpus justifies the effectiveness of our approach over a strong baseline.
</p></blockquote></details> 

<details>
<summary>3. <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPaper/5113">Modeling Textual Cohesion for Event Extraction</a> by<i> Ruihong Huang, Ellen Riloff</i></summary><blockquote><p align="justify">
Event extraction systems typically locate the role fillers for an event by analyzing sentences in isolation and identifying each role filler independently of the others. We argue that more accurate event extraction requires a view of the larger context to decide whether an entity is related to a relevant event. We propose a bottom-up approach to event extraction that initially identifies candidate role fillers independently and then uses that information as well as discourse properties to model textual cohesion. The novel component of the architecture is a sequentially structured sentence classifier that identifies event-related story contexts. The sentence classifier uses lexical associations and discourse relations across sentences, as well as domain-specific distributions of candidate role fillers within and across sentences. This approach yields state-of-the-art performance on the MUC-4 data set, achieving substantially higher precision than previous systems.
</p></blockquote></details>

<details>
<summary>4. <a href="https://www.aclweb.org/anthology/C12-1100/">Joint Modeling of Trigger Identification and Event Type Determination in Chinese Event Extraction</a> by<i> Peifeng Li, Qiaoming Zhu, Hongjun Diao, Guodong Zhou</i></summary><blockquote><p align="justify">
Currently, Chinese event extraction systems suffer much from the low quality of annotated event corpora and the high ratio of pseudo trigger mentions to true ones. To resolve these two issues, this paper proposes a joint model of trigger identification and event type determination. Besides, several trigger filtering schemas are introduced to filter out those pseudo trigger mentions as many as possible. Evaluation on the ACE 2005 Chinese corpus justifies the effectiveness of our approach over a strong baseline.
</p></blockquote></details>

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/C12-1033/">Joint Modeling for Chinese Event Extraction with Rich Linguistic Features</a> by<i> Chen Chen, Vincent Ng</i></summary><blockquote><p align="justify">
Compared to the amount of research that has been done on English event extraction, there exists relatively little work on Chinese event extraction. We seek to push the frontiers of supervised Chinese event extraction research by proposing two extension to Li et al.'s (2012) state-of-the-art event extraction system. First, we employ a joint modeling approach to event extraction, aiming to address the error propagation problem inherent in Li et al.'s pipeline system architecture. Second, we investigate a variety of rich knowledge sources for Chinese event extraction that encode knowledge ranging from the character level to the discourse level. Experimental results on the ACE 2005 dataset show that our joint-modeling, knowledge-rich approach significantly outperforms Li et al.'s approach.
</p></blockquote></details>

<details>
<summary>6. <a href="https://www.aclweb.org/anthology/C12-1033/">Multi-Event Extraction Guided by Global Constraints</a> by<i> Roi Reichart, Regina Barzilay</i></summary><blockquote><p align="justify">
This  paper  addresses  the  extraction of  eventrecords from documents that describe multi-ple  events.   Specifically,  we  aim  to  identify the fields of information contained in a document and aggregate together those fields that describe  the  same  event.   To  exploit  the  inherent  connections  between  field  extraction and event identification, we propose to model them  jointly.   Our  model  is  novel  in  that  it integrates information from separate sequential  models,  using  global  potentials  that  encourage  the  extracted  event  records  to  have desired  properties. While  the  model  contains high-order potentials,  efficient approximate  inference  can  be  performed with  dual decomposition. We experiment with two datasets  that  consist  of  newspaper  articles  describing multiple terrorism events,  and show that our model substantially outperforms tra-ditional pipeline models.
</p></blockquote></details>

### 2013 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W13-2017/">A Hybrid approach for biomedical event extraction</a> by<i> Xuan Quang Pham, Minh Quang Le, Bao Quoc Ho </i></summary><blockquote><p align="justify">
In this paper we propose a system which uses hybrid methods that combine both rule-based and machine learning (ML)-based approaches to solve GENIA Event Extraction of BioNLP Shared Task 2013. We apply UIMA 1 Framework to support coding. There are three main stages in model: Pre-processing, trigger detection and biomedical event detection. We use dictionary and support vector machine classifier to detect event triggers. Event detection is applied on syntactic patterns which are combined with features extracted for classification.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P13-1145/">Argument Inference from Relevant Event Mentions in Chinese Argument Extraction</a> by<i> Peifeng Li, Qiaoming Zhu, Guodong Zhou</i></summary><blockquote><p align="justify">
As a paratactic language, sentence-level argument extraction in Chinese suffers much from the frequent occurrence of ellipsis with regard to inter-sentence arguments. To resolve such problem, this paper proposes a novel global argument inference model to explore specific relationships, such as Coreference, Sequence and Parallel, among relevant event mentions to recover those intersentence arguments in the sentence, discourse and document layers which represent the cohesion of an event or a topic. Evaluation on the ACE 2005 Chinese corpus justifies the effectiveness of our global argument inference model over a state-of-the-art baseline.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.ijcai.org/Proceedings/13/Papers/313.pdf">Joint Modeling of Argument Identification and Role Determination in Chinese Event Extraction with Discourse-Level Information</a> by<i> Peifeng Li, Qiaoming Zhu and Guodong Zhou</i></summary><blockquote><p align="justify">
Argument extraction is a challenging task in event extraction. However, most of previous studies focused on intra-sentence information and failed to extract inter-sentence arguments. This paper proposes a discourse-level joint model of argument identification and role determination to infer those inter-sentence arguments in a discourse. Moreover, to better represent the relationship among relevant event mentions and the relationship between an event mention and its arguments in a discourse, this paper introduces various kinds of corpus-based and discourse-based constraints in the joint model, either automatically learned or linguistically motivated. Evaluation on the ACE 2005 Chinese corpus justifies the effectiveness of our joint model over a strong baseline in Chinese argument extraction, in particular argument identification.
</p></blockquote></details> 

<details>
<summary>4. <a href="https://www.aclweb.org/anthology/P13-1008/">Joint Event Extraction via Structured Prediction with Global Features</a> by<i> Qi Li, Heng Ji, Liang Huang</i></summary><blockquote><p align="justify">
Traditional approaches to the task of ACE event extraction usually rely on sequential pipelines with multiple stages, which suffer from error propagation since event triggers and arguments are predicted in isolation by independent local classifiers. By contrast, we propose a joint framework based on structured prediction which extracts triggers and arguments together so that the local predictions can be mutually improved. In addition, we propose to incorporate global features which explicitly capture the dependencies of multiple triggers and arguments. Experimental results show that our joint approach with local features outperforms the pipelined baseline, and adding global features further improves the performance significantly. Our approach advances state-ofthe-art sentence-level event extraction, and even outperforms previous argument labeling methods which use external knowledge from other sentences and documents.
</p></blockquote></details>

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/W13-2006/">Biomedical Event Extraction by Multi-class Classification of Pairs of Text Entities</a> by<i> Xiao Liu, Antoine Bordes, Yves Grandvalet</i></summary><blockquote><p align="justify">
Biomedical event extraction from articles as become a popular research topic driven by important applications, such as the automatic update of dedicated knowledge base. Most existing approaches are either pipeline models of specific classifiers, usually subject to cascading errors, or joint structured models, more efficient but also more costly and more involved to train. We propose here a system based on a pairwise model that transforms event extraction into a simple multi-class problem of classifying pairs of text entities. Such pairs are recursively provided to the classifier, so as to extract events involving other events as arguments. Our model is more direct than the usual pipeline approaches, and speeds up inference compared to joint models. We report here the best results reported so far on the BioNLP 2011 and 2013 Genia tasks.
</p></blockquote></details>


### 2014

<details>
<summary>1. <a href="https://www.hindawi.com/journals/bmri/2014/205239/">A Novel Feature Selection Strategy for Enhanced Biomedical Event Extraction Using the Turku System</a> by<i> Jingbo Xia, Alex Chengyu Fang, and Xing Zhang</i></summary><blockquote><p align="justify">
Feature selection is of paramount importance for text-mining classifiers with high-dimensional features. The Turku Event Extraction System (TEES) is the best performing tool in the GENIA BioNLP 2009/2011 shared tasks, which relies heavily on high-dimensional features. This paper describes research which, based on an implementation of an accumulated effect evaluation (AEE) algorithm applying the greedy search strategy, analyses the contribution of every single feature class in TEES with a view to identify important features and modify the feature set accordingly. With an updated feature set, a new system is acquired with enhanced performance which achieves an increased -score of 53.27% up from 51.21% for Task 1 under strict evaluation criteria and 57.24% according to the approximate span and recursive criterion.
</p></blockquote></details>



<details>
<summary>2. <a href="https://www.aclweb.org/anthology/D14-1090/">Relieving the Computational Bottleneck: Joint Inference for Event Extraction with High-Dimensional Features</a> by<i> Deepak Venugopal, Chen Chen, Vibhav Gogate, Vincent Ng</i></summary><blockquote><p align="justify">
Several state-of-the-art event extraction systems employ models based on Support Vector Machines (SVMs) in a pipeline architecture, which fails to exploit the joint dependencies that typically exist among events and arguments. While there have been attempts to overcome this limitation using Markov  Logic  Networks  (MLNs),  it  remains challenging to perform joint inference in MLNs when the model encodes many high-dimensional sophisticated features such as those essential for event extraction.  In this paper, we propose a new model for event extraction that combines the power of MLNs and SVMs, dwarfing their limitations.  The key idea is to reliably learn and process high-dimensional features using SVMs;  encode the outputof SVMs as low-dimensional, soft formulas in MLNs; and use the superior joint inferencing power of MLNs to enforce joint consistency constraints over the soft formulas.  We evaluate our approach for the task  of  extracting  biomedical  events  onthe BioNLP 2013, 2011 and 2009 Geniashared task datasets. Our approach yields the best F1 score to date on the BioNLP’13 (53.61) and BioNLP’11 (58.07) datasets and the second-best F1 score to date on theBioNLP’09 dataset (58.16).
</p></blockquote></details>

<details>
<summary>3. <a href="http://aclweb.org/anthology/P14-5007">Real-Time Detection, Tracking, and Monitoring of Automatically Discovered Events in Social Media</a> by<i> Osborne, Miles and Moran, Sean and McCreadie, Richard and Von Lunen, Alexander and Sykora, Martin and Cano, Elizabeth and Ireson, Neil and Macdonald, Craig and Ounis, Iadh and He, Yulan and Jackson, Tom and Ciravegna, Fabio and O'Brien, Ann </i></summary><blockquote><p align="justify">
We introduce ReDites, a system for realtime event detection, tracking, monitoring and visualisation. It is designed to assist Information Analysts in understanding and exploring complex events as they unfold in the world. Events are automatically detected from the Twitter stream. Then those that are categorised as being security-relevant are tracked, geolocated, summarised and visualised for the end-user. Furthermore, the system tracks changes in emotions over events, signalling possible ﬂashpoints or abatement. We demonstrate the capabilities of ReDites using an extended use case from the September 2013 Westgate shooting incident. Through an evaluation of system latencies, we also show that enriched events are made available for users to explore within seconds of that event occurring.
</p></blockquote></details>


<details>
<summary>3. <a href="https://www.cs.cmu.edu/~hovy/papers/14LREC-event-coref.pdf">A Simple Bayesian Modelling Approach to Event Extraction from Twitter</a> by<i> Zhengzhong Liu, Jun Araki, Eduard Hovy, Teruko Mitamura</i></summary><blockquote><p align="justify">
Event coreference is an important task for full text analysis. However, previous work uses a variety of approaches, sources and evaluation,making the literature confusing and the results incommensurate. We provide a description of the differences to facilitate future research. Second,  we  present  a  supervised  method  for  event  coreference  resolution  that  uses  a  rich  feature  set  and  propagates  information alternatively between events and their arguments, adapting appropriately for each type of argument.
</p></blockquote></details>

<details>
<summary>4. <a href="http://aclweb.org/anthology/P14-2114">Supervised Within-Document Event Coreference using Information Propagation</a> by<i> Deyu Zhou, Liangyu Chen, Yulan He</i></summary><blockquote><p align="justify">
With the proliferation of social media sites, social streams have proven to contain the most up-to-date information on current events. Therefore, it is crucial to extract events from the social streams such as tweets. However, it is not straightforward to adapt the existing event extraction systems since texts in social media are fragmented and noisy. In this paper we propose a simple and yet effective Bayesian model, called Latent Event Model (LEM), to extract structured representation of events from social media. LEM is fully unsupervised and does not require annotated data for training. We evaluate LEM on a Twitter corpus. Experimental results show that the proposed model achieves 83% in F-measure, and outperforms the state-of-the-art baseline by over 7%.
</p></blockquote></details>


<details>
<summary>5. <a href="http://aclweb.org/anthology/P14-2136">Bilingual Event Extraction: a Case Study on Trigger Type Determination</a> by<i> Zhu, Zhu and Li, Shoushan and Zhou, Guodong and Xia, Rui </i></summary><blockquote><p align="justify">
Event extraction generally suffers from the data sparseness problem. In this paper, we address this problem by utilizing the labeled data from two different languages. As a preliminary study, we mainly focus on the subtask of trigger type determination in event extraction. To make the training data in different languages help each other, we propose a uniform text representation with bilingual features to represent the samples and handle the difficulty of locating the triggers in the translated text from both monolingual and bilingual perspectives. Empirical studies demonstrate the effectiveness of the proposed approach to bilingual classification on trigger type determination.
</p></blockquote></details>



### 2015 



<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P15-3005/">Disease event detection based on deep modality analysis</a> by<i> Yoshiaki Kitagawa, Mamoru Komachi, Eiji Aramaki, Naoaki Okazaki, Hiroshi Ishikawa</i></summary><blockquote><p align="justify">
Social media has attracted attention because of its potential for extraction of information of various types. For example, information collected from Twitter enables us to build useful applications such as predicting an epidemic of influenza. However, using text information from social media poses challenges for event detection because of the unreliable nature of user-generated texts, which often include counter-factual statements. Consequently, this study proposes the use of modality features to improve disease event detection from Twitter messages, or "tweets". Experimental results demonstrate that the combination of a modality dictionary and a modality analyzer improves the F1-score by 3.5 points.
</p></blockquote></details>

 

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/S15-1018/">Event Extraction as Frame-Semantic Parsing</a> by<i> Alex Judea, Michael Strube</i></summary><blockquote><p align="justify">
Based on the hypothesis that frame-semantic parsing  and  event  extraction  are structurally identical tasks, we retrain SEMAFOR, a state-of-the-art  frame-semantic  parsing  system  to predict event triggers and arguments.  We describe how we change SEMAFOR to be better suited for the new task and show that it performs comparable to one of the best systems in event extraction. We also describe a bias in one of its models and propose a feature factorization which is better suited for this model.
</p></blockquote></details>



<details>
<summary>3. <a href="https://ieeexplore.ieee.org/document/7244210">Extracting Biomedical Event with Dual Decomposition Integrating Word Embeddings</a> by<i>  Lishuang Li ; Shanshan Liu ; Meiyue Qin ; Yiwen Wang ; Degen Huang</i></summary><blockquote><p align="justify">
Extracting biomedical event from literatures has attracted much attention recently. By now, most of the state-of-the-art systems have been based on pipelines which suffer from cascading errors, and the words encoded by one-hot are unable to represent the semantic information. Joint inference with dual decomposition and novel word embeddings are adopted to address the two problems, respectively, in this work. Word embeddings are learnt from large scale unlabeled texts and integrated as an unsupervised feature into other rich features based on dependency parse graphs to detect triggers and arguments. The proposed system consists of four components: trigger detector, argument detector, jointly inference with dual decomposition, and rule-based semantic post-processing, and outperforms the state-of-the-art systems. On the development set of BioNLP'09, the F-score is 59.77 percent on the primary task, which is 0.96 percent higher than the best system. On the test set of BioNLP'11, the F-score is 56.09 and 0.89 percent higher than the best published result that do not adopt additional techniques. On the test set of BioNLP'13, the F-score reaches 53.19 percent which is 2.22 percent higher than the best result.
</p></blockquote></details>

 

<details>
<summary>4. <a href="https://www.aclweb.org/anthology/D15-1247/">Joint event trigger identification and event coreference resolution with structured perceptron</a> by<i> Jun Araki, Teruko Mitamura</i></summary><blockquote><p align="justify">
Events and their coreference offer useful semantic and discourse resources. We show that the semantic and discourse aspects of events interact with each other. However, traditional approaches addressed event extraction and event coreference resolution either separately or sequentially, which limits their interactions. This paper proposes a document-level structured learning model that simultaneously identifies event triggers and resolves event coreference. We demonstrate that the joint model outperforms a pipelined model by 6.9 BLANC F1 and 1.8 CoNLL F1 points in event coreference resolution using a corpus in the biology domain.
</p></blockquote></details>

<details>
<summary>5. <a href="http://aclweb.org/anthology/P15-1056">Bring you to the past: Automatic Generation of Topically Relevant Event Chronicles</a> by<i> Ge, Tao and Pei, Wenzhe and Ji, Heng and Li, Sujian and Chang, Baobao and Sui, Zhifang </i></summary><blockquote><p align="justify">
An event chronicle provides people with an easy and fast access to learn the past. In this paper, we propose the ﬁrst novel approach to automatically generate a topically relevant event chronicle during a certain period given a reference chronicle during another period. Our approach consists of two core components – a timeaware hierarchical Bayesian model for event detection, and a learning-to-rank model to select the salient events to construct the ﬁnal chronicle. Experimental results demonstrate our approach is promising to tackle this new problem.
</p></blockquote></details>

<details>
<summary>6. <a href="http://aclweb.org/anthology/P15-3005">Disease Event Detection based on Deep Modality Analysis</a> by<i> Kitagawa, Yoshiaki and Komachi, Mamoru and Aramaki, Eiji and Okazaki, Naoaki and Ishikawa, Hiroshi </i></summary><blockquote><p align="justify">
Social media has attracted attention because of its potential for extraction of information of various types. For example, information collected from Twitter enables us to build useful applications such as predicting an epidemic of inﬂuenza. However, using text information from social media poses challenges for event detection because of the unreliable nature of user-generated texts, which often include counter-factual statements.
</p></blockquote></details>

<details>
<summary>7. <a href="http://aclweb.org/anthology/P15-2061">Seed-Based Event Trigger Labeling: How far can event descriptions get us?</a> by<i> Bronstein, Ofer and Dagan, Ido and Li, Qi and Ji, Heng and Frank, Anette </i></summary><blockquote><p align="justify">
The task of event trigger labeling is typically addressed in the standard supervised setting: triggers for each target event type are annotated as training data, based on annotation guidelines. We propose an alternative approach, which takes the example trigger terms mentioned in the guidelines as seeds, and then applies an eventindependent similarity-based classiﬁer for trigger labeling. This way we can skip manual annotation for new event types, while requiring only minimal annotated training data for few example events at system setup. Our method is evaluated on the ACE-2005 dataset, achieving 5.7\% F1 improvement over a state-of-the-art supervised system which uses the full training data.
</p></blockquote></details>

### 2016 

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W16-6308/">Biomolecular Event Extraction using a Stacked Generalization based Classifier</a> by<i> Amit Majumder, Asif Ekbal, Sudip Kumar Naskar</i></summary><blockquote><p align="justify">
In this paper we propose a stacked generalization (or stacking) model for event extraction in bio-medical text. Event extraction deals with the process of extracting detailed biological phenomenon, which is more challenging compared to the traditional binary relation extraction such as protein-protein interaction. The overall process consists of mainly three steps: event trigger detection, argument extraction by edge detection and finding correct combination of arguments. In stacking, we use Linear Support Vector Classification (Linear SVC), Logistic Regression (LR) and Stochastic Gradient Descent (SGD) as base-level learning algorithms. As meta-level learner we use Linear SVC. In edge detection step, we find out the arguments of triggers detected in trigger detection step using a SVM classifier. To find correct combination of arguments, we use rules generated by studying the properties of bio-molecular event expressions, and form an event expression consisting of event trigger, its class and arguments. The output of trigger detection is fed to edge detection for argument extraction. Experiments on benchmark datasets of BioNLP2011 show the recall, precision and Fscore of 48.96%, 66.46% and 56.38%, respectively. Comparisons with the existing systems show that our proposed model attains state-of-the-art performance.
</p></blockquote></details>



<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P16-1116/">RBPB: Regularization-Based Pattern Balancing Method for Event Extraction</a> by<i> Lei Sha, Jing Liu, Chin-Yew Lin, Sujian Li, Baobao Chang, Zhifang Sui </i></summary><blockquote><p align="justify">
Event extraction is a particularly challenging information extraction task, which intends to identify and classify event triggers and arguments from raw text. In recent works, when determining event types (trigger classification), most of the works are either pattern-only or feature-only. However, although patterns cannot cover all representations of an event, it is still a very important feature. In addition, when identifying and classifying arguments, previous works consider each candidate argument separately while ignoring the relationship between arguments. This paper proposes a Regularization-Based Pattern Balancing Method (RBPB). Inspired by the progress in representation learning, we use trigger embedding, sentence-level embedding and pattern features together as our features for trigger classification so that the effect of patterns and other useful features can be balanced. In addition, RBPB uses a regularization method to take advantage of the relationship between arguments. Experiments show that we achieve results better than current state-of-art equivalents.
</p></blockquote></details>

 

<details>
<summary>3. <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/11990">A Probabilistic Soft Logic Based Approach to Exploiting Latent and Global Information in Event Classification</a> by<i> Shulin Liu, Kang Liu, Shizhu He, Jun Zhao</i></summary><blockquote><p align="justify">
Global information such as event-event association, and latent local information such as fine-grained entity types, are crucial to event classification. However, existing methods typically focus on sophisticated local features such as part-of-speech tags, either fully or partially ignoring the aforementioned information. By contrast, this paper focuses on fully employing them for event classification. We notice that it is difficult to encode some global information such as event-event association for previous methods. To resolve this problem, we propose a feasible approach which encodes global information in the form of logic using Probabilistic Soft Logic model. Experimental results show that, our proposed approach advances state-of-the-art methods, and achieves the best F1 score to date on the ACE data set.
</p></blockquote></details>



<details>
<summary>4. <a href="https://www.aclweb.org/anthology/C16-1215/">Incremental Global Event Extraction</a> by<i> Alex Judea, Michael Strube</i></summary><blockquote><p align="justify">
Event extraction is a difficult information extraction task. Li et al. (2014) explore the benefits of modeling event extraction and two related tasks, entity mention and relation extraction, jointly. This joint system achieves state-of-the-art performance in all tasks. However, as a system operating only at the sentence level, it misses valuable information from other parts of the document. In this paper, we present an incremental easy-first approach to make the global context of the entire document available to the intra-sentential, state-of-the-art event extractor. We show that our method robustly increases performance on two datasets, namely ACE 2005 and TAC 2015.
</p></blockquote></details>

 

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/N16-1033/">Joint Extraction of Events and Entities within a Document Context</a> by<i> Bishan Yang, Tom M. Mitchell</i></summary><blockquote><p align="justify">
Events and entities are closely related; entities are often actors or participants in events and events without entities are uncommon. The interpretation of events and entities is highly contextually dependent. Existing work in information extraction typically models events separately from entities, and performs inference at the sentence level, ignoring the rest of the document. In this paper, we propose a novel approach that models the dependencies among variables of events, entities, and their relations, and performs joint inference of these variables across a document. The goal is to enable access to document-level contextual information and facilitate context-aware predictions. We demonstrate that our approach substantially outperforms the state-of-the-art methods for event extraction as well as a strong baseline for entity extraction.
</p></blockquote></details>

### 2020 

<details>
<summary>1. <a href="https://www.tandfonline.com/doi/full/10.1080/24751839.2020.1763007">Event detection based on open information extraction and ontology</a> by<i> Sihem Sahnoun, Samir Elloumi, Sadok Ben Yahia</i></summary><blockquote><p align="justify">
Most of the information is available in the form of unstructured textual documents due to the growth of information sources (the Web for example). In this respect, to extract a set of events from texts written in natural language in the management change event, we have been introduced an open information extraction (OIE) system. For instance, in the management change event, a PERSON might be either the new coming person to the company or the leaving one. As a result, the Adaptive CRF approach (A-CRF) has shown good performance results. However, it requires a lot of expert intervention during the construction of classifiers, which is time consuming. To palpate such a downside, we introduce an approach that reduces the expert intervention during the relation extraction. Also, the named entity recognition and the reasoning, which are automatic and based on techniques of adaptation and correspondence, were implemented. Carried out experiments show the encouraging results of the main approaches of the literature.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/2020.lrec-1.258/">Event Extraction from Unstructured Amharic Text</a> by<i> ephrem tadesse, Rosa Tsegaye, Kuulaa Qaqqabaa</i></summary><blockquote><p align="justify">
In information extraction, event extraction is one of the types that extract the specific knowledge of certain incidents from texts. Event extraction has been done on different languages text but not on one of the Semitic language, Amharic. In this study, we present a system that extracts an event from unstructured Amharic text. The system has designed by the integration of supervised machine learning and rule-based approaches. We call this system a hybrid system. The system uses the supervised machine learning to detect events from the text and the handcrafted and the rule-based rules to extract the event from the text. For the event extraction, we have been using event arguments. Event arguments identify event triggering words or phrases that clearly express the occurrence of the event. The event argument attributes can be verbs, nouns, sometimes adjectives (such as ̃rg/wedding) and time as well. The hybrid system has compared with the standalone rule-based method that is well known for event extraction. The study has shown that the hybrid system has outperformed the standalone rule-based method.
</p></blockquote></details>

## Deep learning 
[:arrow_up:](#table-of-contents)
### 2015 



<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P15-2060/">Event detection and domain adaptation with convolutional neural networks</a> by<i> Thien Huu Nguyen, Ralph Grishman </i>(<a href="https://github.com/ThanhChinhBK/event_detector">Github</a>)</summary><blockquote><p align="justify">
We study the event detection problem using convolutional neural networks (CNNs) that overcome the two fundamental limitations of the traditional feature-based approaches to this task: complicated feature engineering for rich feature sets and error propagation from the preceding stages which generate these features. The experimental results show that the CNNs outperform the best reported feature-based systems in the general setting as well as the domain adaptation setting without resorting to extensive external resources.
</p></blockquote></details>



<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P15-1017/">Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks</a> by<i>  Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, Jun Zhao</i> (<a href="https://github.com/zhangluoyang/cnn-for-auto-event-extract">Github</a>)</summary><blockquote><p align="justify">
Traditional approaches to the task of ACE event extraction primarily rely on elaborately designed features and complicated natural language processing (NLP) tools. These traditional approaches lack generalization, take a large amount of human effort and are prone to error propagation and data sparsity problems. This paper proposes a novel event-extraction method, which aims to automatically extract lexical-level and sentence-level features without using complicated NLP tools. We introduce a word-representation model to capture meaningful semantic regularities for words and adopt a framework based on a convolutional neural network (CNN) to capture sentence-level clues. However, CNN can only capture the most important information in a sentence and may miss valuable facts when considering multiple-event sentences. We propose a dynamic multi-pooling convolutional neural network (DMCNN), which uses a dynamic multi-pooling layer according to event triggers and arguments, to reserve more crucial information. The experimental results show that our approach significantly outperforms other state-of-the-art methods.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.semanticscholar.org/paper/Event-Nugget-Detection%2C-Classification-and-using-Reimers-Gurevych/1b5cf83ea210e1793526c915e132d21e53f6726f">Event Nugget Detection, Classification and Coreference Resolution using Deep Neural Networks and eXtreme Grandient Boosting</a> by<i> Nils Reimers, Iryna Gurevych </i> (<a href="https://github.com/UKPLab/tac2015-event-detection">Github</a>)</summary><blockquote><p align="justify">
Traditional approaches to the task of ACE event extraction primarily rely on elaborately designed features and complicated natural language processing (NLP) tools. These traditional approaches lack generalization, take a large amount of human effort and are prone to error propagation and data sparsity problems. This paper proposes a novel event-extraction method, which aims to automatically extract lexical-level and sentence-level features without using complicated NLP tools. We introduce a word-representation model to capture meaningful semantic regularities for words and adopt a framework based on a convolutional neural network (CNN) to capture sentence-level clues. However, CNN can only capture the most important information in a sentence and may miss valuable facts when considering multiple-event sentences. We propose a dynamic multi-pooling convolutional neural network (DMCNN), which uses a dynamic multi-pooling layer according to event triggers and arguments, to reserve more crucial information. The experimental results show that our approach significantly outperforms other state-of-the-art methods.
</p></blockquote></details>

### 2016 

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/D16-1085/">Modeling Skip-Grams for Event Detection with Convolutional Neural Networks</a> by<i> Thien Huu Nguyen, Ralph Grishman</i></summary><blockquote><p align="justify">
Convolutional neural networks (CNN) have achieved the top performance for event detection due to their capacity to induce the underlying structures of the k-grams in the sentences. However, the current CNN-based event detectors only model the consecutive k-grams and ignore the non-consecutive kgrams that might involve important structures for event detection. In this work, we propose to improve the current CNN models for ED by introducing the non-consecutive convolution. Our systematic evaluation on both the general setting and the domain adaptation setting demonstrates the effectiveness of the non-consecutive CNN model, leading to the significant performance improvement over the current state-of-the-art systems.
</p></blockquote></details>



<details>
<summary>2. <a href="https://link.springer.com/chapter/10.1007%2F978-3-319-50496-4_27">Joint Event Extraction Based on Skip-Window Convolutional Neural Networks</a> by<i> Zhengkuan Zhang, Weiran Xu, Qianqian Chen</i></summary><blockquote><p align="justify">
Traditional approaches to the task of ACE event extraction are either the joint model with elaborately designed features which may lead to generalization and data-sparsity problems, or the word-embedding model based on a two-stage, multi-class classification architecture, which suffers from error propagation since event triggers and arguments are predicted in isolation. This paper proposes a novel event-extraction method that not only extracts triggers and arguments simultaneously, but also adopts a framework based on convolutional neural networks (CNNs) to extract features automatically. However, CNNs can only capture sentence-level features, so we propose the skip-window convolution neural networks (S-CNNs) to extract global structured features, which effectively capture the global dependencies of every token in the sentence. The experimental results show that our approach outperforms other state-of-the-art methods.
</p></blockquote></details>



<details>
<summary>3. <a href="https://www.aclweb.org/anthology/N16-1034/">Joint Event Extraction via Recurrent Neural Networks</a> by<i> Thien Huu Nguyen, Kyunghyun Cho, Ralph Grishman </i>(<a href="https://github.com/anoperson/jointEE-NN">Github</a>)</summary><blockquote><p align="justify">
Event extraction is a particularly challenging problem in information extraction. The stateof-the-art models for this problem have either applied convolutional neural networks in a pipelined framework (Chen et al., 2015) or followed the joint architecture via structured prediction with rich local and global features (Li et al., 2013). The former is able to learn hidden feature representations automatically from data based on the continuous and generalized representations of words. The latter, on the other hand, is capable of mitigating the error propagation problem of the pipelined approach and exploiting the inter-dependencies between event triggers and argument roles via discrete structures. In this work, we propose to do event extraction in a joint framework with bidirectional recurrent neural networks, thereby benefiting from the advantages of the two models as well as addressing issues inherent in the existing approaches. We systematically investigate different memory features for the joint model and demonstrate that the proposed model achieves the state-of-the-art performance on the ACE 2005 dataset.
</p></blockquote></details>



<details>
<summary>4. <a href="https://www.aclweb.org/anthology/P16-2060/">Event Nugget Detection with Forward-Backward Recurrent Neural Networks</a> by<i> Reza Ghaeini, Xiaoli Fern, Liang Huang, Prasad Tadepalli</i></summary><blockquote><p align="justify">
Traditional event detection methods heavily rely on manually engineered rich features. Recent deep learning approaches alleviate this problem by automatic feature engineering. But such efforts, like tradition methods, have so far only focused on single-token event mentions, whereas in practice events can also be a phrase. We instead use forward-backward recurrent neural networks (FBRNNs) to detect events that can be either words or phrases. To the best our knowledge, this is one of the first efforts to handle multi-word events and also the first attempt to use RNNs for event detection. Experimental results demonstrate that FBRNN is competitive with the state-of-the-art methods on the ACE 2005 and the Rich ERE 2015 event detection tasks.
</p></blockquote></details>

 

<details>
<summary>5. <a href="https://link.springer.com/chapter/10.1007/978-3-319-47674-2_17">Event Extraction via Bidirectional Long Short-Term Memory Tensor Neural Network</a> by<i>     Yubo Chen, Shulin Liu, Shizhu He, Kang LiuJun Zhao</i></summary><blockquote><p align="justify">
Traditional approaches to the task of ACE event extraction usually rely on complicated natural language processing (NLP) tools and elaborately designed features. Which suffer from error propagation of the existing tools and take a large amount of human effort. And nearly all of approaches extract each argument of an event separately without considering the interaction between candidate arguments. By contrast, we propose a novel event-extraction method, which aims to automatically extract valuable clues without using complicated NLP tools and predict all arguments of an event simultaneously. In our model, we exploit a context-aware word representation model based on Long Short-Term Memory Networks (LSTM) to capture the semantics of words from plain texts. In addition, we propose a tensor layer to explore the interaction between candidate arguments and predict all arguments simultaneously. The experimental results show that our approach significantly outperforms other state-of-the-art methods.
</p></blockquote></details>



<details>
<summary>6. <a href="https://tac.nist.gov/publications/2016/participant.papers/TAC2016.wip.proceedings.pdf">WIP Event Detection System at TAC KBP 2016 Event Nugget Track</a> by<i> Ying  Zeng, Bingfeng Luo, Yansong Feng, Dongyan Zhao</i></summary><blockquote><p align="justify">
Event detection aims to extract events with specific types from unstructured data, which is the crucial and challenging task in event related applications, such as event coreference resolution and event argument extraction. In this paper, we propose an event detection system that combines traditional feature-based methods and novel neural network (NN) models. Experiments show that our ensemble approaches can achieve promising performance in the Event Nugget Detection task. 
</p></blockquote></details>



<details>
<summary>7. <a href="https://link.springer.com/chapter/10.1007%2F978-3-319-50496-4_23">A Convolution BiLSTM Neural Network Model for Chinese Event Extraction</a> by<i>     Ying Zeng, Honghui Yang, Yansong Feng, Zheng Wang, Dongyan Zhao</i></summary><blockquote><p align="justify">
Chinese event extraction is a challenging task in information extraction. Previous approaches highly depend on sophisticated feature engineering and complicated natural language processing (NLP) tools. In this paper, we first come up with the language specific issue in Chinese event extraction, and then propose a convolution bidirectional LSTM neural network that combines LSTM and CNN to capture both sentence-level and lexical information without any hand-craft features. Experiments on ACE 2005 dataset show that our approaches can achieve competitive performances in both trigger labeling and argument role labeling.
</p></blockquote></details>



<details>
<summary>8. <a href="https://www.aclweb.org/anthology/P16-2011/">A Language-Independent Neural Network for Event Detection</a> by<i> Xiaocheng Feng, Lifu Huang, Duyu Tang, Heng Ji, Bing Qin, Ting Liu</i></summary><blockquote><p align="justify">
Event detection remains a challenge because of the difficulty of encoding the word semantics in various contexts. Previous approaches have heavily depended on language-specific knowledge and preexisting natural language processing tools. However, not all languages have such resources and tools available compared with English language. A more promising approach is to automatically learn effective features from data, without relying on language-specific resources. In this study, we develop a language-independent neural network to capture both sequence and chunk information from specific contexts and use them to train an event detector for multiple languages without any manually encoded features. Experiments show that our approach can achieve robust, efficient and accurate results for various languages. In the ACE 2005 English event detection task, our approach achieved a 73.4% F-score with an average of 3.0% absolute improvement compared with state-of-the-art. Additionally, our experimental results are competitive for Chinese and Spanish.
</p></blockquote></details>

### 2017 

 

<details>
<summary>1. <a href="https://link.springer.com/chapter/10.1007%2F978-3-319-69005-6_11">Improving Event Detection via Information Sharing among Related Event Types</a> by<i> Shulin Liu, Yubo Chen, Kang Liu, Jun Zhao, Zhunchen Luo, Wei Luo</i></summary><blockquote><p align="justify">
Event detection suffers from data sparseness and label imbalance problem due to the expensive cost of manual annotations of events. To address this problem, we propose a novel approach that allows for information sharing among related event types. Specifically, we employ a fully connected three-layer artificial neural network as our basic model and propose a type-group regularization term to achieve the goal of information sharing. We conduct experiments with different configurations of type groups, and the experimental results show that information sharing among related event types remarkably improves the detecting performance. Compared with state-of-the-art methods, our proposed approach achieves a better F-1 score on the widely used ACE 2005 event evaluation dataset.
</p></blockquote></details>

 

<details>
<summary>2. <a href="http://oro.open.ac.uk/49639/">On semantics and deep learning for event detection in crisis situations</a> by<i> Burel Grégoire; Saif Hassan; Fernandez Miriam and Alani Harith</i></summary><blockquote><p align="justify">
In this paper, we introduce Dual-CNN, a semantically-enhanced deep learning model to target the problem of event detection in crisis situations from social media data. A layer of semantics is added to a traditional Convolutional Neural Network (CNN) model to capture the contextual information that is generally scarce in short, ill-formed social media messages. Our results show that our methods are able to successfully identify the existence of events, and event types (hurricane, floods, etc.) accurately (> 79% F-measure), but the performance of the model significantly drops (61% F-measure) when identifying fine-grained event-related information (affected individuals, damaged infrastructures, etc.). These results are competitive with more traditional Machine Learning models, such as SVM.
</p></blockquote></details>



<details>
<summary>3. <a href="https://www.aclweb.org/anthology/I17-1036/">Exploiting Document Level Information to Improve Event Detection via Recurrent Neural Networks</a> by<i> Shaoyang Duan, Ruifang He, Wenli Zhao</i></summary><blockquote><p align="justify">
This paper tackles the task of event detection, which involves identifying and categorizing events. The previous work mainly exist two problems: (1) the traditional feature-based methods apply cross-sentence information, yet need taking a large amount of human effort to design complicated feature sets and inference rules; (2) the representation-based methods though overcome the problem of manually extracting features, while just depend on local sentence representation. Considering local sentence context is insufficient to resolve ambiguities in identifying particular event types, therefore, we propose a novel document level Recurrent Neural Networks (DLRNN) model, which can automatically extract cross-sentence clues to improve sentence level event detection without designing complex reasoning rules. Experiment results show that our approach outperforms other state-of-the-art methods on ACE 2005 dataset without external knowledge base.
</p></blockquote></details>



<details>
<summary>4. <a href="https://www.aclweb.org/anthology/W17-2315/">Biomedical Event Extraction using Abstract Meaning Representation</a> by<i>  Sudha Rao, Daniel Marcu, Kevin Knight, Hal Daumé III</i></summary><blockquote><p align="justify">
We propose a novel, Abstract Meaning Representation (AMR) based approach to identifying molecular events/interactions in biomedical text. Our key contributions are: (1) an empirical validation of our hypothesis that an event is a subgraph of the AMR graph, (2) a neural network-based model that identifies such an event subgraph given an AMR, and (3) a distant supervision based approach to gather additional training data. We evaluate our approach on the 2013 Genia Event Extraction dataset and show promising results.
</p></blockquote></details>



<details>
<summary>5. <a href="https://www.aclweb.org/anthology/P17-1164/">Exploiting Argument Information to Improve Event Detection via Supervised Attention Mechanisms</a> by<i> Shulin Liu, Yubo Chen, Kang Liu, Jun Zhao</i></summary><blockquote><p align="justify">
This paper tackles the task of event detection (ED), which involves identifying and categorizing events. We argue that arguments provide significant clues to this task, but they are either completely ignored or exploited in an indirect manner in existing detection approaches. In this work, we propose to exploit argument information explicitly for ED via supervised attention mechanisms. In specific, we systematically investigate the proposed model under the supervision of different attention strategies. Experimental results show that our approach advances state-of-the-arts and achieves the best F1 score on ACE 2005 dataset.
</p></blockquote></details>


<details>
<summary>6. <a href="https://www.aclweb.org/anthology/D17-1163/">Identifying civilians killed by police with distantly supervised entity-event extraction</a> by<i> Katherine Keith, Abram Handler, Michael Pinkham, Cara Magliozzi, Joshua McDuffie, Brendan O’Connor</i> (<a href="https://github.com/slanglab/policefatalities_emnlp2017">Github</a>)</summary><blockquote><p align="justify">
We propose a new, socially-impactful task for natural language processing: from a news corpus, extract names of persons who have been killed by police. We present a newly collected police fatality corpus, which we release publicly, and present a model to solve this problem that uses EM-based distant supervision with logistic regression and convolutional neural network classifiers. Our model outperforms two off-the-shelf event extractor systems, and it can suggest candidate victim names in some cases faster than one of the major manually-collected police fatality databases.
</p></blockquote></details>

<details>
<summary>7. <a href="http://aclweb.org/anthology/P17-2046">English Event Detection With Translated Language Features</a> by<i> Wei, Sam and Korostil, Igor and Nothman, Joel and Hachey, Ben </i></summary><blockquote><p align="justify">
We propose novel radical features from automatic translation for event extraction. Event detection is a complex language processing task for which it is expensive to collect training data, making generalisation challenging. We derive meaningful subword features from automatic translations into target language. Results suggest this method is particularly useful when using languages with writing systems that facilitate easy decomposition into subword features, e.g., logograms and Cangjie. The best result combines logogram features from Chinese and Japanese with syllable features from Korean, providing an additional 3.0 points f-score when added to state-of-the-art generalisation features on the TAC KBP 2015 Event Nugget task.
</p></blockquote></details>

<details>
<summary>8. <a href="http://dl.acm.org/citation.cfm?doid=3123266.3123294">Improving Event Extraction via Multimodal Integration</a> by<i> Zhang, Tongtao and Whitehead, Spencer and Zhang, Hanwang and Li, Hongzhi and Ellis, Joseph and Huang, Lifu and Liu, Wei and Ji, Heng and Chang, Shih-Fu </i></summary><blockquote><p align="justify">
In this paper, we focus on improving Event Extraction (EE) by incorporating visual knowledge with words and phrases from text documents. We rst discover visual pa erns from large-scale textimage pairs in a weakly-supervised manner and then propose a multimodal event extraction algorithm where the event extractor is jointly trained with textual features and visual pa erns. Extensive experimental results on benchmark data sets demonstrate that the (a) proposed multimodal EE method can achieve signi cantly be er performance on event extraction: absolute 7.1\% F-score gain on event trigger labeling and 8.5\% F-score gain on event argument labeling.
</p></blockquote></details>


### 2018 

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/D18-1517/">Similar but not the Same: Word Sense Disambiguation Improves Event Detection via Neural Representation Matching</a> by<i> Weiyi Lu, Thien Huu Nguyen</i></summary><blockquote><p align="justify">
Event detection (ED) and word sense disambiguation (WSD) are two similar tasks in that they both involve identifying the classes (i.e. event types or word senses) of some word in a given sentence. It is thus possible to extract the knowledge hidden in the data for WSD, and utilize it to improve the performance on ED. In this work, we propose a method to transfer the knowledge learned on WSD to ED by matching the neural representations learned for the two tasks. Our experiments on two widely used datasets for ED demonstrate the effectiveness of the proposed method.
</p></blockquote></details>



<details>
<summary>2. <a href="https://www.aclweb.org/anthology/D18-1127/">Exploiting Contextual Information via Dynamic Memory Network for Event Detection</a> by<i> Shaobo Liu, Rui Cheng, Xiaoming Yu, Xueqi Cheng </i>(<a href="https://github.com/AveryLiu/TD-DMN">Github</a>)</summary><blockquote><p align="justify">
The task of event detection involves identifying and categorizing event triggers. Contextual information has been shown effective on the task. However, existing methods which utilize contextual information only process the context once. We argue that the context can be better exploited by processing the context multiple times, allowing the model to perform complex reasoning and to generate better context representation, thus improving the overall performance. Meanwhile, dynamic memory network (DMN) has demonstrated promising capability in capturing contextual information and has been applied successfully to various tasks. In light of the multi-hop mechanism of the DMN to model the context, we propose the trigger detection dynamic memory network (TD-DMN) to tackle the event detection problem. We performed a five-fold cross-validation on the ACE-2005 dataset and experimental results show that the multi-hop mechanism does improve the performance and the proposed model achieves best F1 score compared to the state-of-the-art methods.
</p></blockquote></details>

 

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/P18-1145/">Nugget Proposal Networks for Chinese Event Detection</a> by<i> Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun</i> (<a href="https://github.com/sanmusunrise/NPNs">Github</a>)</summary><blockquote><p align="justify">
Neural network based models commonly regard event detection as a word-wise classification task, which suffer from the mismatch problem between words and event triggers, especially in languages without natural word delimiters such as Chinese. In this paper, we propose Nugget Proposal Networks (NPNs), which can solve the word-trigger mismatch problem by directly proposing entire trigger nuggets centered at each character regardless of word boundaries. Specifically, NPNs perform event detection in a character-wise paradigm, where a hybrid representation for each character is first learned to capture both structural and semantic information from both characters and words. Then based on learned representations, trigger nuggets are proposed and categorized by exploiting character compositional structures of Chinese event triggers. Experiments on both ACE2005 and TAC KBP 2017 datasets show that NPNs significantly outperform the state-of-the-art methods.
</p></blockquote></details>



<details>
<summary>4. <a href="https://ieeexplore.ieee.org/document/8453008">Extracting Biomedical Events with Parallel Multi-Pooling Convolutional Neural Networks</a> by<i>  Lishuang Li ; Yang Liu ; Meiyue Qin</i></summary><blockquote><p align="justify">
Biomedical event extraction is important for medical research and disease prevention, which has attracted much attention in recent years. Traditionally, most of the state-of-the-art systems have been based on shallow machine learning methods, which require many complex, hand-designed features. In addition, the words encoded by one-hot are unable to represent semantic information. Therefore, we utilize dependency-based embeddings to represent words semantically and syntactically. Then, we propose a parallel multi-pooling convolutional neural network (PMCNN) model to capture the compositional semantic features of sentences. Furthermore, we employ a rectified linear unit, which creates sparse representations with true zeros, and which is adapted to the biomedical event extraction, as a nonlinear function in PMCNN architecture. The experimental results from MLEE dataset show that our approach achieves an F1 score of 80.27% in trigger identification and an F1 score of 59.65% in biomedical event extraction, which performs better than other state-of-the-art methods.
</p></blockquote></details>



<details>
<summary>5. <a href="https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16222">Jointly Extracting Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based Argument Interaction</a> by<i> Lei Sha, Feng Qian, Baobao Chang, Zhifang Sui</i></summary><blockquote><p align="justify">
Event extraction plays an important role in natural language processing (NLP) applications including question answering and information retrieval. Traditional event extraction relies heavily on lexical and syntactic features, which require intensive human engineering and may not generalize to different datasets. Deep neural networks, on the other hand, are able to automatically learn underlying features, but existing networks do not make full use of syntactic relations. In this paper, we propose a novel dependency bridge recurrent neural network (dbRNN) for event extraction. We build our model upon a recurrent neural network, but enhance it with dependency bridges, which carry syntactically related information when modeling each word. We illustrates that simultaneously applying tree structure and sequence structure in RNN brings much better performance than only uses sequential RNN. In addition, we use a tensor layer to simultaneously capture the various types of latent interaction between candidate arguments as well as identify/classify all arguments of an event. Experiments show that our approach achieves competitive results compared with previous work.
</p></blockquote></details>



<details>
<summary>6. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01012-6_20">Learning Target-Dependent Sentence Representations for Chinese Event Detection</a> by<i> Wenbo Zhang, Xiao Ding, Ting Liu </i></summary><blockquote><p align="justify">
Chinese event detection is a particularly challenging task in information extraction. Previous work mainly consider the sequential representation of sentences. However, long-range dependencies between words in the sentences may hurt the performance of these approaches. We believe that syntactic representations can provide an effective mechanism to directly link words to their informative context in the sentences. In this paper, we propose a novel event detection model based on dependency trees. In particular, we propose transforming dependency trees to target-dependent trees where leaf nodes are words and internal nodes are dependency relations, to distinguish the target words. Experimental results on the ACE 2005 corpus show that our approach significantly outperforms state-of-the-art baseline methods.
</p></blockquote></details>



<details>
<summary>7. <a href="https://arxiv.org/abs/1812.00195">One for All: Neural Joint Modeling of Entities and Events</a> by<i> Trung Minh Nguyen, Thien Huu Nguyen</i></summary><blockquote><p align="justify">
The previous work for event extraction has mainly focused on the predictions for event triggers and argument roles, treating entity mentions as being provided by human annotators. This is unrealistic as entity mentions are usually predicted by some existing toolkits whose errors might be propagated to the event trigger and argument role recognition. Few of the recent work has addressed this problem by jointly predicting entity mentions, event triggers and arguments. However, such work is limited to using discrete engineering features to represent contextual information for the individual tasks and their interactions. In this work, we propose a novel model to jointly perform predictions for entity mentions, event triggers and arguments based on the shared hidden representations from deep learning. The experiments demonstrate the benefits of the proposed method, leading to the state-of-the-art performance for event extraction.
</p></blockquote></details>



<details>
<summary>8. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0950705119300097?via%3Dihub">Empower event detection with bi-directional neural language model</a> by<i> Yunyan Zhang , Guangluan Xu , Yang Wang, Xiao Liang, Lei Wang, Tinglei Huang</i></summary><blockquote><p align="justify">
Event detection is an essential and challenging task in Information Extraction (IE). Recent advances in neural networks make it possible to build reliable models without complicated feature engineering. However, data scarcity hinders their further performance. Moreover, training data has been underused since majority of labels in datasets are not event triggers and contribute very little to the training process. In this paper, we propose a novel multi-task learning framework to extract more general patterns from raw data and make better use of the training data. Specifically, we present two paradigms to incorporate neural language model into event detection model on both word and character levels: (1) we use the features extracted by language model as an additional input to event detection model. (2) We use a hard parameter sharing approach between language model and event detection model. The extensive experiments demonstrate the benefits of the proposed multi-task learning framework for event detection. Compared to the previous methods, our method does not rely on any additional supervision but still beats the majority of them and achieves a competitive performance on the ACE 2005 benchmark.
</p></blockquote></details>



<details>
<summary>9. <a href="https://arxiv.org/abs/1809.09078">Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation</a> by<i> Xiao Liu, Zhunchen Luo, Heyan Huang</i> (<a href="https://github.com/lx865712528/EMNLP2018-JMEE">Github</a>)</summary><blockquote><p align="justify">
Event extraction is of practical utility in natural language processing. In the real world, it is a common phenomenon that multiple events existing in the same sentence, where extracting them are more difficult than extracting a single event. Previous works on modeling the associations between events by sequential modeling methods suffer a lot from the low efficiency in capturing very long-range dependencies. In this paper, we propose a novel Jointly Multiple Events Extraction (JMEE) framework to jointly extract multiple event triggers and arguments by introducing syntactic shortcut arcs to enhance information flow and attention-based graph convolution networks to model graph information. The experiment results demonstrate that our proposed framework achieves competitive results compared with state-of-the-art methods.
</p></blockquote></details>

 

<details>
<summary>10. <a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16329">Graph Convolutional Networks With Argument-Aware Pooling for Event Detection</a> by<i> Thien Huu Nguyen, Ralph Grishman</i></summary><blockquote><p align="justify">
The current neural network models for event detection have only considered the sequential representation of sentences. Syntactic representations have not been explored in this area although they provide an effective mechanism to directly link words to their informative context for event detection in the sentences. In this work, we investigate a convolutional neural network based on dependency trees to perform event detection. We propose a novel pooling method that relies on entity mentions to aggregate the convolution vectors. The extensive experiments demonstrate the benefits of the dependency based convolutional neural networks and the entity mentionbased pooling method for event detection. We achieve the state-of-the-art performance on widely used datasets with both perfect and predicted entity mentions.
</p></blockquote></details>



<details>
<summary>11. <a href="https://link.springer.com/chapter/10.1007%2F978-3-030-04221-9_23">Chinese Event Recognition via Ensemble Model</a> by<i> Wei Liu, Zhenyu Yang, Zongtian Liu</i></summary><blockquote><p align="justify">
Event recognition is one of the most fundamental and critical field in information extraction. In this paper, Event recognition task can be divided into two sub-problems containing candidate event triggers identification and the classification of candidate event trigger words. Firstly, we use trigger vocabulary generated by trigger expansion to identify candidate event trigger, and then input sequences are generated according to the following three features: word embedding, POS (part of speech) and DP (dependency parsing). Finally multiclass classifier based on joint neural networks is introduced in the step of candidate trigger classification. The experiments in CEC (Chinese Emergency Corpus) have shown the superiority of our proposal model with a maximum F-measure of 80.55%.
</p></blockquote></details>



<details>
<summary>12. <a href="http://ceur-ws.org/Vol-2266/T5-2.pdf">A neural network based Event extraction system for Indian languages</a> by<i> Alapan Kuila, Sarath chandra Bussa, Sudeshna Sarkar</i></summary><blockquote><p align="justify">
In this paper we have described a neural network based approach for Event extraction(EE) task which aims to discover different types of events along with the event arguments form the text documents written in Indian languages like Hindi, Tamil and English as part of our participation in the task on Event Extraction from Newswires and Social Media Text in Indian Languages at Forum for Information Retrieval Evaluation (FIRE) in 2018. A neural netork model which is a combination of Convolution neural network(CNN) and Recurrent neural network(RNN) is employed for the Event identification task. In addition to event detection, the system also extracts the event arguments which contain the information related to the events(i.e. when[Time], where[Place], Reason, Casualty, After-effect etc.). Our proposed Event Extraction model achieves f-score of 39.71, 37.42 and 39.91 on Hindi, Tamil and English dataset respectively which shows the overall performance of Event identification and argument extraction task in these three language domain.
</p></blockquote></details>



<details>
<summary>13. <a href="https://iopscience.iop.org/article/10.1088/1742-6596/978/1/012078">5W1H Information Extraction with CNN-Bidirectional LSTM</a> by<i> A Nurdin1, N U Maulidevi</i></summary><blockquote><p align="justify">
In this work, information about who, did what, when, where, why, and how on Indonesian news articles were extracted by combining Convolutional Neural Network and Bidirectional Long Short-Term Memory. Convolutional Neural Network can learn semantically meaningful representations of sentences. Bidirectional LSTM can analyze the relations among words in the sequence. We also use word embedding word2vec for word representation. By combining these algorithms, we obtained F-measure 0.808. Our experiments show that CNN-BLSTM outperforms other shallow methods, namely IBk, C4.5, and Naïve Bayes with the F-measure 0.655, 0.645, and 0.595, respectively.
</p></blockquote></details>

<details>
<summary>14. <a href="https://www.aclweb.org/anthology/P18-1048/">Self-regulation: Employing a Generative Adversarial Network to Improve Event Detection</a> by<i> Yu Hong, Wenxuan Zhou, Jingli Zhang, Guodong Zhou, Qiaoming Zhu</i> (<a href="https://github.com/JoeZhouWenxuan/Self-regulation-Employing-a-Generative-Adversarial-Network-to-Improve-Event-Detection">Github</a>)</summary><blockquote><p align="justify">
Due to the ability of encoding and mapping semantic information into a high-dimensional latent feature space, neural networks have been successfully used for detecting events to a certain extent. However, such a feature space can be easily contaminated by spurious features inherent in event detection. In this paper, we propose a self-regulated learning approach by utilizing a generative adversarial network to generate spurious features. On the basis, we employ a recurrent network to eliminate the fakes. Detailed experiments on the ACE 2005 and TAC-KBP 2015 corpora show that our proposed method is highly effective and adaptable.
</p></blockquote></details>



<details>
<summary>15. <a href="http://tcci.ccf.org.cn/conference/2018/papers/51.pdf">Event Detection via Recurrent Neural Networkand Argument Prediction</a> by<i> Wentao Wu, Xiaoxu Zhu, Jiaming Tao, and Peifeng Li</i></summary><blockquote><p align="justify">
This paper tackles the task of event detection, which involves identifying and categorizing the events. Currently event detection remains a challenging task due to the difficulty at encoding the event semantics in complicate contexts. The core semantics of an event may derive from its trigger and arguments. However, most of previous studies failed to capture the argument semantics in event detection. To address this issue, this paper first provides a rule-based method to predict candidate arguments on the event types of possibilities, and then proposes a recurrent neural network model RNN-ARG with the attention mechanism for event detection to capture meaningful semantic regularities form these predicted candidate arguments. The experimental results on the ACE 2005 English corpus show that our approach achieves competitive results compared with previous work.
</p></blockquote></details>



<details>
<summary>16. <a href="https://arxiv.org/abs/1808.08504">Event Detection with Neural Networks: A Rigorous Empirical Evaluation</a> by<i> J. Walker Orr, Prasad Tadepalli, Xiaoli Fern</i></summary><blockquote><p align="justify">
Detecting events and classifying them into predefined types is an important step in knowledge extraction from natural language texts. While the neural network models have generally led the state-of-the-art, the differences in performance between different architectures have not been rigorously studied. In this paper we present a novel GRU-based model that combines syntactic information along with temporal structure through an attention mechanism. We show that it is competitive with other neural network architectures through empirical evaluations under different random initializations and training-validation-test splits of ACE2005 dataset. 
</p></blockquote></details>

 

<details>
<summary>17. <a href="https://link.springer.com/chapter/10.1007/978-3-319-99495-6_15">Using Entity Relation to Improve Event Detection via Attention Mechanism</a> by<i>     Jingli Zhang, Wenxuan Zhou, Yu Hong, Jianmin Yao, Min Zhang</i></summary><blockquote><p align="justify">
Identifying event instance in texts plays a critical role in the field of Information Extraction (IE). The currently proposed methods that employ neural networks have successfully solve the problem to some extent, by encoding a series of linguistic features, such as lexicon, part-of-speech and entity. However, so far, the entity relation hasn’t yet been taken into consideration. In this paper, we propose a novel event extraction method to exploit relation information for event detection (ED), due to the potential relevance between entity relation and event type. Methodologically, we combine relation and those widely used features in an attention-based network with Bidirectional Long Short-term Memory (Bi-LSTM) units. In particular, we systematically investigate the effect of relation representation between entities. In addition, we also use different attention strategies in the model. Experimental results show that our approach outperforms other state-of-the-art methods
</p></blockquote></details>

<details>
<summary>18. <a href="https://link.springer.com/chapter/10.1007/978-3-030-05090-0_17">Event Extraction with Deep Contextualized Word Representation and Multi-attention Layer</a> by<i>  Ruixue Ding, Zhoujun Li</i></summary><blockquote><p align="justify">
One common application of text mining is event extraction. The purpose of an event extraction task is to identify event triggers of a certain event type in the text and to find related arguments. In recent years, the technology to automatically extract events from text has drawn researchers’ attention. However, the existing works including feature based systems and neural network base models don’t capture the contextual information well. Besides, it is still difficult to extract deep semantic relations when finding related arguments for events. To address these issues, we propose a novel model for event extraction using multi-attention layers and deep contextualized word representation. Furthermore, we put forward an attention function suitable for event extraction tasks. Experimental results show that our model outperforms the state-of-the-art models on ACE2005.
</p></blockquote></details> 

<details>
<summary>19. <a href="https://www.mdpi.com/1999-5903/10/10/95">Chinese Event Extraction Based on Attention and Semantic Features: A Bidirectional Circular Neural Network</a> by<i> Yue Wu; Junyi Zhang</i></summary><blockquote><p align="justify">
Chinese event extraction uses word embedding to capture similarity, but suffers when handling previously unseen or rare words. From the test, we know that characters may provide some information that we cannot obtain in words, so we propose a novel architecture for combining word representations: character–word embedding based on attention and semantic features. By using an attention mechanism, our method is able to dynamically decide how much information to use from word or character level embedding. With the semantic feature, we can obtain some more information about a word from the sentence. We evaluate different methods on the CEC Corpus, and this method is found to improve performance.
</p></blockquote></details>



<details>
<summary>20. <a href="https://www.aclweb.org/anthology/P18-2066/">Document Embedding Enhanced Event Detection with Hierarchical and Supervised Attention</a> by<i> Yue Zhao, Xiaolong Jin, Yuanzhuo Wang, Xueqi Cheng</i></summary><blockquote><p align="justify">
Document-level information is very important for event detection even at sentence level. In this paper, we propose a novel Document Embedding Enhanced Bi-RNN model, called DEEB-RNN, to detect events in sentences. This model first learns event detection oriented embeddings of documents through a hierarchical and supervised attention based RNN, which pays word-level attention to event triggers and sentence-level attention to those sentences containing events. It then uses the learned document embedding to enhance another bidirectional RNN model to identify event triggers and their types in sentences. Through experiments on the ACE-2005 dataset, we demonstrate the effectiveness and merits of the proposed DEEB-RNN model via comparison with state-of-the-art methods.
</p></blockquote></details>


<details>
<summary>21. <a href="https://www.aclweb.org/anthology/D18-1158/">Collective Event Detection via a Hierarchical and Bias Tagging Networks with Gated Multi-level Attention Mechanisms</a> by<i> Yubo Chen, Hang Yang, Kang Liu, Jun Zhao, Yantao Jia</i> (<a href="https://github.com/yubochen/NBTNGMA4ED">Github</a>)</summary><blockquote><p align="justify">
Traditional approaches to the task of ACE event detection primarily regard multiple events in one sentence as independent ones and recognize them separately by using sentence-level information. However, events in one sentence are usually interdependent and sentence-level information is often insufficient to resolve ambiguities for some types of events. This paper proposes a novel framework dubbed as Hierarchical and Bias Tagging Networks with Gated Multi-level Attention Mechanisms (HBTNGMA) to solve the two problems simultaneously. Firstly, we propose a hierachical and bias tagging networks to detect multiple events in one sentence collectively. Then, we devise a gated multi-level attention to automatically extract and dynamically fuse the sentence-level and document-level information. The experimental results on the widely used ACE 2005 dataset show that our approach significantly outperforms other state-of-the-art methods.
</p></blockquote></details>
 

<details>
<summary>22. <a href="http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Liu_aaai2018.pdf">Event Detection via Gated Multilingual Attention Mechanism</a> by<i> Jian Liu, Yubo Chen1, Kang Liu, Jun Zhao</i></summary><blockquote><p align="justify">
Identifying event instance in text plays a critical role in building NLP applications such as Information Extraction (IE) system. However, most existing methods for this task focus only on monolingual clues of a specific language and ignore the massive information provided by other languages. Data scarcity and monolingual ambiguity hinder the performance of these monolingual approaches. In this paper, we propose a novel multilingual approach — dubbed as Gated MultiLingual Attention (GMLATT) framework — to address the two issues simultaneously. In specific, to alleviate data scarcity problem, we exploit the consistent information in multilingual data via context attention mechanism. Which takes advantage of the consistent evidence in multilingual data other than learning only from monolingual data. To deal with monolingual ambiguity problem, we propose gated cross-lingual attention to exploit the complement information conveyed by multilingual data, which is helpful for the disambiguation. The cross-lingual attention gate serves as a sentinel modelling the confidence of the clues provided by other languages and controls the information integration of various languages. We have conducted extensive experiments on the ACE 2005 benchmark. Experimental results show that our approach significantly outperforms state-of-the-art methods.
</p></blockquote></details>



<details>
<summary>23. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01012-6_21">Prior Knowledge Integrated with Self-attention for Event Detection</a> by<i> Yan Li, Chenliang Li, Weiran Xu, Junliang Li</i></summary><blockquote><p align="justify">
Recently, end-to-end models based on recurrent neural networks (RNN) have gained great success in event detection. However these methods cannot deal with long-distance dependency and internal structure information well. They are also hard to be controlled in process of learning since lacking of prior knowledge integration. In this paper, we present an effective framework for event detection which aims to address these problems. Our model based on self-attention can ignore the distance between any two words to obtain their relationship and leverage internal event argument information to improve event detection. In order to control the process of learning, we first collect keywords from corpus and then use a prior knowledge integration network to encode keywords to a prior knowledge representation. Experimental results demonstrate that our model has significant improvement of 3.9 F1 over the previous state-of-the-art on ACE 2005 dataset.
</p></blockquote></details>


### 2019

 

<details>
<summary>1. <a href="https://arxiv.org/abs/1906.06003">Cost-sensitive Regularization for Label Confusion-aware Event Detection</a> by<i> Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun</i> (<a href="https://github.com/sanmusunrise/CSR">Github</a>)</summary><blockquote><p align="justify">
In supervised event detection, most of the mislabeling occurs between a small number of confusing type pairs, including trigger-NIL pairs and sibling sub-types of the same coarse type. To address this label confusion problem, this paper proposes cost-sensitive regularization, which can force the training procedure to concentrate more on optimizing confusing type pairs. Specifically, we introduce a cost-weighted term into the training loss, which penalizes more on mislabeling between confusing label pairs. Furthermore, we also propose two estimators which can effectively measure such label confusion based on instance-level or population-level statistics. Experiments on TAC-KBP 2017 datasets demonstrate that the proposed method can significantly improve the performances of different models in both English and Chinese event detection.
</p></blockquote></details>



<details>
<summary>2. <a href="https://link.springer.com/chapter/10.1007/978-3-030-15712-8_51">Exploiting a More Global Context for Event Detection Through Bootstrapping</a> by<i> Dorian Kodelja, Romaric Besançon, Olivier Ferret</i></summary><blockquote><p align="justify">
Over the last few years, neural models for event extraction have obtained interesting results. However, their application is generally limited to sentences, which can be an insufficient scope for disambiguating some occurrences of events. In this article, we propose to integrate into a convolutional neural network the representation of contexts beyond the sentence level. This representation is built following a bootstrapping approach by exploiting an intra-sentential convolutional model. Within the evaluation framework of TAC 2017, we show that our global model significantly outperforms the intra-sentential model while the two models are competitive with the results obtained by TAC 2017 participants.
</p></blockquote></details>



<details>
<summary>3. <a href="https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btz607/5544930?redirectedFrom=fulltext">Context awareness and embedding for biomedical event extraction</a> by<i> Shankai Yan, Ka-Chun Wong</i></summary><blockquote><p align="justify">
Motivation: Biomedical event detection is fundamental for information extraction in molecular biology and biomedical research. The detected events form the central basis for comprehensive biomedical knowledge fusion, facilitating the digestion of massive information influx from literature. Limited by the feature context, the existing event detection models are mostly applicable for a single task. A general and scalable computational model is desiderated for biomedical knowledge management. Results: We consider and propose a bottom-up detection framework to identify the events from recognized arguments. To capture the relations between the arguments, we trained a bi-directional Long Short-Term Memory (LSTM) network to model their context embedding. Leveraging the compositional attributes, we further derived the candidate samples for training event classifiers. We built our models on the datasets from BioNLP Shared Task for evaluations. Our method achieved the average F-scores of 0.81 and 0.92 on BioNLPST-BGI and BioNLPST-BB datasets respectively. Comparing with 7 state-of-the-art methods, our method nearly doubled the existing F-score performance (0.92 vs 0.56) on the BioNLPST-BB dataset. Case studies were conducted to reveal the underlying reasons. Availability: https://github.com/cskyan/evntextrc
</p></blockquote></details>

 

<details>
<summary>4. <a href="https://www.aclweb.org/anthology/N19-1145/">Biomedical Event Extraction based on Knowledge-driven Tree-LSTM</a> by<i> Diya Li, Lifu Huang, Heng Ji, Jiawei Han</i></summary><blockquote><p align="justify">
Event extraction for the biomedical domain is more challenging than that in the general news domain since it requires broader acquisition of domain-specific knowledge and deeper understanding of complex contexts. To better encode contextual information and external background knowledge, we propose a novel knowledge base (KB)-driven tree-structured long short-term memory networks (Tree-LSTM) framework, incorporating two new types of features: (1) dependency structures to capture wide contexts; (2) entity properties (types and category descriptions) from external ontologies via entity linking. We evaluate our approach on the BioNLP shared task with Genia dataset and achieve a new state-of-the-art result. In addition, both quantitative and qualitative studies demonstrate the advancement of the Tree-LSTM and the external knowledge representation for biomedical event extraction.
</p></blockquote></details>

<details>
<summary>5. <a href="https://aaai.org/ojs/index.php/AAAI/article/view/4649">Exploiting the Ground-Truth: An Adversarial Imitation Based Knowledge Distillation Approach for Event Detection</a> by<i>  Jian Liu,  Yubo Chen,  Kang Liu </i></summary><blockquote><p align="justify">
The ambiguity in language expressions poses a great challenge for event detection. To disambiguate event types, current approaches rely on external NLP toolkits to build knowledge representations. Unfortunately, these approaches work in a pipeline paradigm and suffer from error propagation problem. In this paper, we propose an adversarial imitation based knowledge distillation approach, for the first time, to tackle the challenge of acquiring knowledge from rawsentences for event detection. In our approach, a teacher module is first devised to learn the knowledge representations from the ground-truth annotations. Then, we set up a student module that only takes the raw-sentences as the input. The student module is taught to imitate the behavior of the teacher under the guidance of an adversarial discriminator. By this way, the process of knowledge distillation from rawsentence has been implicitly integrated into the feature encoding stage of the student module. To the end, the enhanced student is used for event detection, which processes raw texts and requires no extra toolkits, naturally eliminating the error propagation problem faced by pipeline approaches. We conduct extensive experiments on the ACE 2005 datasets, and the experimental results justify the effectiveness of our approach.
</p></blockquote></details>
 

<details>
<summary>6. <a href="http://nlp.cs.rpi.edu/paper/imitation2019.pdf">Joint Entity and Event Extraction with Generative Adversarial Imitation Learning</a> by<i> Tongtao Zhang, Heng Ji, Avirup Sil</i></summary><blockquote><p align="justify">
We propose a new framework for entity and event extraction based on generative adversarial imitation learning-an inverse reinforcement learning method using a generative adversarial network (GAN). We assume that instances and labels yield to various extents of difficulty and the gains and penalties (rewards) are expected to be diverse. We utilize discriminators to estimate proper rewards according to the difference between the labels committed by the ground-truth (expert) and the extractor (agent). Our experiments demonstrate that the proposed framework outperforms state-of-the-art methods.
</p></blockquote></details>

 

<details>
<summary>7. <a href="https://dl.acm.org/citation.cfm?doid=3308558.3313659">Event Detection using Hierarchical Multi-Aspect Attention</a> by<i> Sneha Mehta, Mohammad Raihanul Islam, Huzefa Rangwala, Naren Ramakrishnan</i> (<a href="https://github.com/sumehta/FBMA">Github</a>)</summary><blockquote><p align="justify">
Classical event encoding and extraction methods rely on fixed dictionaries of keywords and templates or require ground truth labels for phrase/sentences. This hinders widespread application of information encoding approaches to large-scale free form (unstructured) text available on the web. Event encoding can be viewed as a hierarchical task where the coarser level task is event detection, i.e., identification of documents containing a specific event, and where the fine-grained task is one of event encoding, i.e., identifying key phrases, key sentences. Hierarchical models with attention seem like a natural choice for this problem, given their ability to differentially attend to more or less important features when constructing document representations. In this work we present a novel factorized bilinear multi-aspect attention mechanism (FBMA) that attends to different aspects of text while constructing its representation. We find that our approach outperforms state-of-the-art baselines for detecting civil unrest, military action, and non-state actor events from corpora in two different languages.
</p></blockquote></details>

<details>
<summary>8. <a href="https://www.ijcai.org/proceedings/2019/753">Extracting Entities and Events as a Single Task Using a Transition-Based Neural Model</a> by<i> Junchi Zhang, Yanxia Qin, Yue Zhang, Mengchi Liu, Donghong Ji</i></summary><blockquote><p align="justify">
The task of event extraction contains subtasks including detections for entity mentions, event triggers and argument roles. Traditional methods solve them as a pipeline, which does not make use of task correlation for their mutual benefits. There have been recent efforts towards building a joint model for all tasks. However, due to technical challenges, there has not been work predicting the joint output structure as a single task. We build a first model to this end using a neural transition-based framework, incrementally predicting complex joint structures in a state-transition process. Results on standard benchmarks show the benefits of the joint model, which gives the best result in the literature. 
</p></blockquote></details>



<details>
<summary>9. <a href="https://link.springer.com/chapter/10.1007%2F978-3-030-32381-3_22">Leveraging Multi-head Attention Mechanism to Improve Event Detection</a> by<i>     Meihan Tong, Bin Xu, Lei Hou, Juanzi Li, Shuai Wang</i></summary><blockquote><p align="justify">
Event detection (ED) task aims to automatically identify trigger words from unstructured text. In recent years, neural models with attention mechanism have achieved great success on this task. However, existing attention methods tend to focus on meaningless context words and ignore the semantically rich words, which weakens their ability to recognize trigger words. In this paper, we propose MANN, a multi-head attention mechanism model enhanced by argument knowledge to address the above issues. The multi-head mechanism gives MANN the ability to detect a variety of information in a sentence while argument knowledge acts as a supervisor to further improve the quality of attention. Experimental results show that our approach is significantly superior to existing attention-based models.
</p></blockquote></details>



<details>
<summary>10. <a href="https://ialp2019.com/files/papers/IALP2019_092.pdf">Using Mention Segmentation to Improve Event Detection with Multi-head Attention</a> by<i> Jiali Chen, Yu Hong, Jingli Zhang, and Jianmin Yao</i></summary><blockquote><p align="justify">
Sentence-level event detection (ED) is a task ofdetecting words that describe specific types of events, in-cluding the subtasks of trigger word identification and eventtype classification. Previous work straight forwardly inputs asentence into neural classification models and analyzes deepsemantics of words in the sentence one by one. Relying on the semantics, probabilities of event classes can be predicted foreach word, including the carefully defined ACE event classesand a ”N/A” class(i.e., non-trigger word). The models achieve remarkable successes nowadays. However, our findings show that a natural sentence may posses more than one trigger word and thus entail different types of events. In particular,the closely related information of each event only lies in a unique sentence segment but has nothing to do with other segments. In order to reduce negative influences from noises in other segments, we propose to perform semantics learning for event detection only in the scope of segment instead of the whole sentence. Accordingly, we develop a novel ED method which integrates sentence segmentation into the neural event classification architecture. Bidirectional Long Short-Term Memory (Bi-LSTM) with multi-head attention is used as the classification model. Sentence segmentation is boiled down to a sequence labeling problem, where BERT is used. We combine embeddings, and use them as the input of the neural classification model. The experimental results show that the performance of our method reaches 76.8% and 74.2% F1-scores for trigger identification and event type classification, which outperforms the state-of-the-art
</p></blockquote></details>



<details>
<summary>11. <a href="https://www.aclweb.org/anthology/P19-1429/">Distilling Discrimination and Generalization Knowledge for Event Detection via Delta-Representation Learning</a> by<i> Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun</i></summary><blockquote><p align="justify">
Event detection systems rely on discrimination knowledge to distinguish ambiguous trigger words and generalization knowledge to detect unseen/sparse trigger words. Current neural event detection approaches focus on trigger-centric representations, which work well on distilling discrimination knowledge, but poorly on learning generalization knowledge. To address this problem, this paper proposes a Delta-learning approach to distill discrimination and generalization knowledge by effectively decoupling, incrementally learning and adaptively fusing event representation. Experiments show that our method significantly outperforms previous approaches on unseen/sparse trigger words, and achieves state-of-the-art performance on both ACE2005 and KBP2017 datasets.
</p></blockquote></details>



<details>
<summary>12. <a href="https://www.aclweb.org/anthology/P19-1471/">Detecting Subevents using Discourse and Narrative Features</a> by<i> Mohammed Aldawsari, Mark Finlayson</i></summary><blockquote><p align="justify">
Recognizing the internal structure of events is a challenging language processing task of great importance for text understanding. We present a supervised model for automatically identifying when one event is a subevent of another. Building on prior work, we introduce several novel features, in particular discourse and narrative features, that significantly improve upon prior state-of-the-art performance. Error analysis further demonstrates the utility of these features. We evaluate our model on the only two annotated corpora with event hierarchies: HiEve and the Intelligence Community corpus. No prior system has been evaluated on both corpora. Our model outperforms previous systems on both corpora, achieving 0.74 BLANC F1 on the Intelligence Community corpus and 0.70 F1 on the HiEve corpus, respectively a 15 and 5 percentage point improvement over previous models.
</p></blockquote></details>



<details>
<summary>13. <a href="https://www.aclweb.org/anthology/P19-1521/">Cost-sensitive Regularization for Label Confusion-aware Event Detection</a> by<i> Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun</i></summary><blockquote><p align="justify">
In supervised event detection, most of the mislabeling occurs between a small number of confusing type pairs, including trigger-NIL pairs and sibling sub-types of the same coarse type. To address this label confusion problem, this paper proposes cost-sensitive regularization, which can force the training procedure to concentrate more on optimizing confusing type pairs. Specifically, we introduce a cost-weighted term into the training loss, which penalizes more on mislabeling between confusing label pairs. Furthermore, we also propose two estimators which can effectively measure such label confusion based on instance-level or population-level statistics. Experiments on TAC-KBP 2017 datasets demonstrate that the proposed method can significantly improve the performances of different models in both English and Chinese event detection.
</p></blockquote></details>



<details>
<summary>14. <a href="https://www.aclweb.org/anthology/P19-1522/">Exploring Pre-trained Language Models for Event Extraction and Generation</a> by<i> Sen Yang, Dawei Feng, Linbo Qiao, Zhigang Kan, Dongsheng Li</i></summary><blockquote><p align="justify">
Traditional approaches to the task of ACE event extraction usually depend on manually annotated data, which is often laborious to create and limited in size. Therefore, in addition to the difficulty of event extraction itself, insufficient training data hinders the learning process as well. To promote event extraction, we first propose an event extraction model to overcome the roles overlap problem by separating the argument prediction in terms of roles. Moreover, to address the problem of insufficient training data, we propose a method to automatically generate labeled data by editing prototypes and screen out generated samples by ranking the quality. Experiments on the ACE2005 dataset demonstrate that our extraction model can surpass most existing extraction methods. Besides, incorporating our generation method exhibits further significant improvement. It obtains new state-of-the-art results on the event extraction task, including pushing the F1 score of trigger classification to 81.1%, and the F1 score of argument classification to 58.9%.
</p></blockquote></details>



<details>
<summary>15. <a href="https://www.aclweb.org/anthology/D19-1027/">Open Event Extraction from Online Text using a Generative Adversarial Network</a> by<i> Rui Wang, Deyu ZHOU, Yulan He</i></summary><blockquote><p align="justify">
To extract the structured representations of open-domain events, Bayesian graphical models have made some progress. However, these approaches typically assume that all words in a document are generated from a single event. While this may be true for short text such as tweets, such an assumption does not generally hold for long text such as news articles. Moreover, Bayesian graphical models often rely on Gibbs sampling for parameter inference which may take long time to converge. To address these limitations, we propose an event extraction model based on Generative Adversarial Nets, called Adversarial-neural Event Model (AEM). AEM models an event with a Dirichlet prior and uses a generator network to capture the patterns underlying latent events. A discriminator is used to distinguish documents reconstructed from the latent events and the original documents. A byproduct of the discriminator is that the features generated by the learned discriminator network allow the visualization of the extracted events. Our model has been evaluated on two Twitter datasets and a news article dataset. Experimental results show that our model outperforms the baseline approaches on all the datasets, with more significant improvements observed on the news article dataset where an increase of 15\% is observed in F-measure. 
</p></blockquote></details>



<details>
<summary>16. <a href="https://www.aclweb.org/anthology/D19-1030/">Cross-lingual Structure Transfer for Relation and Event Extraction</a> by<i> Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss</i></summary><blockquote><p align="justify">
The identification of complex semantic structures such as events and entity relations, already a challenging Information Extraction task, is doubly difficult from sources written in under-resourced and under-annotated languages. We investigate the suitability of cross-lingual structure transfer techniques for these tasks. We exploit relation and event-relevant language-universal features, leveraging both symbolic (including part-of-speech and dependency path) and distributional (including type representation and contextualized representation) information. By representing all entity mentions, event triggers, and contexts into this complex and structured multilingual common space, using graph convolutional networks, we can train a relation or event extractor from source language annotations and apply it to the target language. Extensive experiments on cross-lingual relation and event transfer among English, Chinese, and Arabic demonstrate that our approach achieves performance comparable to state-of-the-art supervised models trained on up to 3,000 manually annotated mentions: up to 62.6% F-score for Relation Extraction, and 63.1% F-score for Event Argument Role Labeling. The event argument role labeling model transferred from English to Chinese achieves similar performance as the model trained from Chinese. We thus find that language-universal symbolic and distributional representations are complementary for cross-lingual structure transfer.
</p></blockquote></details>



<details>
<summary>17. <a href="https://www.aclweb.org/anthology/D19-1033/">Event Detection with Trigger-Aware Lattice Neural Network</a> by<i> Ning Ding, Ziran Li, Zhiyuan Liu, Haitao Zheng, Zibo Lin</i> (<a href="https://github.com/thunlp/TLNN">Github</a>)</summary><blockquote><p align="justify">
Event detection (ED) aims to locate trigger words in raw text and then classify them into correct event types. In this task, neural net- work based models became mainstream in recent years. However, two problems arise when it comes to languages without natural delimiters, such as Chinese. First, word-based models severely suffer from the problem of word trigger mismatch, limiting the performance of the methods. In addition, even if trigger words could be accurately located, the ambiguity of polysemy of triggers could still affect the trigger classification stage. To address the two issues simultaneously, we propose the Trigger-aware Lattice Neural Net- work (TLNN). (1) The framework dynamically incorporates word and character information so that the trigger-word mismatch issue can be avoided. (2) Moreover, for polysemous characters and words, we model all senses of them with the help of an external linguistic knowledge base, so as to alleviate the problem of ambiguous triggers. Experiments on two benchmark datasets show that our model could effectively tackle the two issues and outperforms previous state-of-the-art methods significantly, giving the best results. The source code of this paper can be obtained from https://github.com/thunlp/TLNN.
</p></blockquote></details>



<details>
<summary>18. <a href="https://www.aclweb.org/anthology/D19-1041/">Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction</a> by<i> Rujun Han, Qiang Ning, Nanyun Peng</i> (<a href="https://github.com/rujunhan/EMNLP-2019">Github</a>)</summary><blockquote><p align="justify">
We propose a joint event and temporal relation extraction model with shared representation learning and structured prediction. The proposed method has two advantages over existing work. First, it improves event representation by allowing the event and relation modules to share the same contextualized embeddings and neural representation learner. Second, it avoids error propagation in the conventional pipeline systems by leveraging structured inference and learning methods to assign both the event labels and the temporal relation labels jointly. Experiments show that the proposed method can improve both event extraction and temporal relation extraction over state-of-the-art systems, with the end-to-end F1 improved by 10% and 6.8% on two benchmark datasets respectively. 
</p></blockquote></details>



<details>
<summary>19. <a href="https://www.aclweb.org/anthology/D19-1584/">HMEAE: Hierarchical Modular Event Argument Extraction</a> by<i> Xiaozhi Wang, Ziqi Wang, Xu Han, Zhiyuan Liu, Juanzi Li, Peng Li, Maosong Sun, Jie Zhou, Xiang Ren</i> (<a href="https://github.com/thunlp/HMEAE">Github</a>)</summary><blockquote><p align="justify">
Existing event extraction methods classify each argument role independently, ignoring the conceptual correlations between different argument roles. In this paper, we propose a Hierarchical Modular Event Argument Extraction (HMEAE) model, to provide effective inductive bias from the concept hierarchy of event argument roles. Specifically, we design a neural module network for each basic unit of the concept hierarchy, and then hierarchically compose relevant unit modules with logical operations into a role-oriented modular network to classify a specific argument role. As many argument roles share the same high-level unit module, their correlation can be utilized to extract specific event arguments better. Experiments on real-world datasets show that HMEAE can effectively leverage useful knowledge from the concept hierarchy and significantly outperform the state-of-the-art baselines. The source code can be obtained from https://github.com/thunlp/HMEAE.
</p></blockquote></details>



<details>
<summary>20. <a href="https://www.aclweb.org/anthology/D19-1585/">Entity, Relation, and Event Extraction with Contextualized Span Representations</a> by<i> David Wadden, Ulme Wennberg, Yi Luan, Hannaneh Hajishirzi</i> (<a href="https://github.com/dwadden/dygiepp">Github</a>)</summary><blockquote><p align="justify">
We examine the capabilities of a unified, multi-task framework for three information extraction tasks: named entity recognition, relation extraction, and event extraction. Our framework (called DyGIE++) accomplishes all tasks by enumerating, refining, and scoring text spans designed to capture local (within-sentence) and global (cross-sentence) context. Our framework achieves state-of-the-art results across all tasks, on four datasets from a variety of domains. We perform experiments comparing different techniques to construct span representations. Contextualized embeddings like BERT perform well at capturing relationships among entities in the same or adjacent sentences, while dynamic span graph updates model long-range cross-sentence relationships. For instance, propagating span representations via predicted coreference links can enable the model to disambiguate challenging entity mentions. Our code is publicly available at this https URL and can be easily adapted for new tasks or datasets. 
</p></blockquote></details>

<details>
<summary>21. <a href="https://www.aclweb.org/anthology/D19-1582/">Event Detection with Multi-Order Graph Convolution and Aggregated Attention</a> by<i> Haoran Yan, Xiaolong Jin, Xiangbin Meng, Jiafeng Guo, Xueqi Cheng</i> (<a href="https://github.com/ll0iecas/MOGANED">Github TensorFlow Unofficial</a>, <a href="https://github.com/wzq016/MOGANED-Implementation">Github Pytorch Unofficial</a>)</summary><blockquote><p align="justify">
Syntactic relations are broadly used in many NLP tasks. For event detection, syntactic relation representations based on dependency tree can better capture the interrelations between candidate trigger words and related entities than sentence representations. But, existing studies only use first-order syntactic relations (i.e., the arcs) in dependency trees to identify trigger words. For this reason, this paper proposes a new method for event detection, which uses a dependency tree based graph convolution network with aggregative attention to explicitly model and aggregate multi-order syntactic representations in sentences. Experimental comparison with state-of-the-art baselines shows the superiority of the proposed method.
</p></blockquote></details>*

<details>
<summary>22. <a href="https://ieeexplore.ieee.org/document/8852355">GADGET: Using Gated GRU for Biomedical Event Trigger Detection</a> by<i>  Cheng Zeng ; Yi Zhang ; Heng-Yang Lu ; Chong-Jun Wang </i></summary><blockquote><p align="justify">
Biomedical event extraction plays an important role in the field of biomedical text mining, and the event trigger detection is the first step in the pipeline process of event extraction. Event trigger can clearly indicates the occurrence of related events. There have been many machine learning based methods applied to this area already. However, most previous work have omitted two crucial points: (1) Class Difference: They simply regard non-trigger as same level class label. (2) Information Isolation: Most methods only utilize token level information. In this paper, we propose a novel model based on gate mechanism, which identifies trigger and non-trigger words in the first stage. At the same time, we also introduce additional fusion layer in order to incorporate sentence level information for event trigger detection. Experimental results on the Multi Level Event Extraction (MLEE) corpus achieve superior performance than other state-of-the-art models. We have also performed ablation study to show the effectiveness of proposed model components.
</p></blockquote></details>

<details>
<summary>23. <a href="https://arxiv.org/abs/1910.11621">Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection</a> by<i>  Shumin Deng, Ningyu Zhang, Jiaojian Kang, Yichi Zhang, Wei Zhang, Huajun Chen </i></summary><blockquote><p align="justify">
Event detection (ED), a sub-task of event extraction, involves identifying triggers and categorizing event mentions. Existing methods primarily rely upon supervised learning and require large-scale labeled event datasets which are unfortunately not readily available in many real-life applications. In this paper, we consider and reformulate the ED task with limited labeled data as a Few-Shot Learning problem. We propose a Dynamic-Memory-Based Prototypical Network (DMB-PN), which exploits Dynamic Memory Network (DMN) to not only learn better prototypes for event types, but also produce more robust sentence encodings for event mentions. Differing from vanilla prototypical networks simply computing event prototypes by averaging, which only consume event mentions once, our model is more robust and is capable of distilling contextual information from event mentions for multiple times due to the multi-hop mechanism of DMNs. The experiments show that DMB-PN not only deals with sample scarcity better than a series of baseline models but also performs more robustly when the variety of event types is relatively large and the instance quantity is extremely small. 
</p></blockquote></details>

## 2020

<details>
<summary>1. <a href="https://arxiv.org/abs/2002.10757">Event Detection with Relation-Aware Graph Convolutional Neural Networks</a> by<i> Shiyao Cui, Bowen Yu, Tingwen Liu, Zhenyu Zhang, Xuebin Wang, Jinqiao Shi</i></summary><blockquote><p align="justify">
Event detection (ED), a key subtask of information extraction, aims to recognize instances of specific types of events in text. Recently, graph convolutional networks (GCNs) over dependency trees have been widely used to capture syntactic structure information and get convincing performances in event detection. However, these works ignore the syntactic relation labels on the tree, which convey rich and useful linguistic knowledge for event detection. In this paper, we investigate a novel architecture named Relation-Aware GCN (RA-GCN), which efficiently exploits syntactic relation labels and models the relation between words specifically. We first propose a relation-aware aggregation module to produce expressive word representation by aggregating syntactically connected words through specific relation. Furthermore, a context-aware relation update module is designed to explicitly update the relation representation between words, and these two modules work in the mutual promotion way. Experimental results on the ACE2005 dataset show that our model achieves a new state-of-the-art performance for event detection. 
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/2020.lrec-1.273">A Platform for Event Extraction in Hindi</a> by<i> Sovan Kumar Sahoo, Saumajit Saha, Asif Ekbal, Pushpak Bhattacharyya</i></summary><blockquote><p align="justify">
Event Extraction is an important task in the widespread field of Natural Language Processing (NLP). Though this task is adequately addressed in English with sufficient resources, we are unaware of any benchmark setup in Indian languages. Hindi is one of the most widely spoken languages in the world. In this paper, we present an Event Extraction framework for Hindi language by creating an annotated resource for benchmarking, and then developing deep learning based models to set as the baselines. We crawl more than seventeen hundred disaster related Hindi news articles from the various news sources. We also develop deep learning based models for Event Trigger Detection and Classification, Argument Detection and Classification and Event-Argument Linking.
</p></blockquote></details>

<details>
<summary>3. <a href="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3376-2">Biomedical event extraction with a novel combination strategy based on hybrid deep neural networks</a> by<i> Lvxing Zhu, Haoran Zheng</i></summary><blockquote><p align="justify">
Background

Biomedical event extraction is a fundamental and in-demand technology that has attracted substantial interest from many researchers. Previous works have heavily relied on manual designed features and external NLP packages in which the feature engineering is large and complex. Additionally, most of the existing works use the pipeline process that breaks down a task into simple sub-tasks but ignores the interaction between them. To overcome these limitations, we propose a novel event combination strategy based on hybrid deep neural networks to settle the task in a joint end-to-end manner.

Results

We adapted our method to several annotated corpora of biomedical event extraction tasks. Our method achieved state-of-the-art performance with noticeable overall F1 score improvement compared to that of existing methods for all of these corpora.

Conclusions

The experimental results demonstrated that our method is effective for biomedical event extraction. The combination strategy can reconstruct complex events from the output of deep neural networks, while the deep neural networks effectively capture the feature representation from the raw text. The biomedical event extraction implementation is available online at http://www.predictor.xin/event_extraction.
</p></blockquote></details>

<details>
<summary>4. <a href="https://www.aclweb.org/anthology/2020.lrec-1.244/">Cross-Domain Evaluation of Edge Detection for Biomedical Event Extraction</a> by<i> Alan Ramponi, Barbara Plank, Rosario Lombardo</i></summary><blockquote><p align="justify">
Biomedical event extraction is a crucial task in order to automatically extract information from the increasingly growing body of biomedical literature. Despite advances in the methods in recent years, most event extraction systems are still evaluated in-domain and on complete event structures only. This makes it hard to determine the performance of intermediate stages of the task, such as edge detection, across different corpora. Motivated by these limitations, we present the first cross-domain study of edge detection for biomedical event extraction. We analyze differences between five existing gold standard corpora, create a standardized benchmark corpus, and provide a strong baseline model for edge detection. Experiments show a large drop in performance when the baseline is applied on out-of-domain data, confirming the need for domain adaptation methods for the task. To encourage research efforts in this direction, we make both the data and the baseline available to the research community: https://www.cosbi.eu/cfx/9985.
</p></blockquote></details>

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/2020.lrec-1.243/">Cross-lingual Structure Transfer for Zero-resource Event Extraction</a> by<i> Di Lu, Ananya Subburathinam, Heng Ji, Jonathan May, Shih-Fu Chang, Avi Sil, Clare Voss</i></summary><blockquote><p align="justify">
Most of the current cross-lingual transfer learning methods for Information Extraction (IE) have been only applied to name tagging. To tackle more complex tasks such as event extraction we need to transfer graph structures (event trigger linked to multiple arguments with various roles) across languages. We develop a novel share-and-transfer framework to reach this goal with three steps: (1) Convert each sentence in any language to language-universal graph structures; in this paper we explore two approaches based on universal dependency parses and complete graphs, respectively. (2) Represent each node in the graph structure with a cross-lingual word embedding so that all sentences in multiple languages can be represented with one shared semantic space. (3) Using this common semantic space, train event extractors from English training data and apply them to languages that do not have any event annotations. Experimental results on three languages (Spanish, Russian and Ukrainian) without any annotations show this framework achieves comparable performance to a state-of-the-art supervised model trained from more than 1,500 manually annotated event mentions.
</p></blockquote></details>

<details>
<summary>6. <a href="https://arxiv.org/abs/2004.13625">Event Extraction by Answering (Almost) Natural Questions</a> by<i> Xinya Du, Claire Cardie</i></summary><blockquote><p align="justify">
The problem of event extraction requires detecting the event trigger and extracting its corresponding arguments. Existing work in event argument extraction typically relies heavily on entity recognition as a preprocessing/concurrent step, causing the well-known problem of error propagation. To avoid this issue, we introduce a new paradigm for event extraction by formulating it as a question answering (QA) task, which extracts the event arguments in an end-to-end manner. Empirical results demonstrate that our framework outperforms prior methods substantially; in addition, it is capable of extracting event arguments for roles not seen at training time (zero-shot learning setting). 
</p></blockquote></details>

<details>
<summary>7. <a href="https://www.aclweb.org/anthology/2020.lrec-1.216/">Towards Few-Shot Event Mention Retrieval: An Evaluation Framework and A Siamese Network Approach</a> by<i> Bonan Min, Yee Seng Chan, Lingjun Zhao</i></summary><blockquote><p align="justify">
Automatically analyzing events in a large amount of text is crucial for situation awareness and decision making. Previous approaches treat event extraction as “one size fits all” with an ontology defined a priori. The resulted extraction models are built just for extracting those types in the ontology. These approaches cannot be easily adapted to new event types nor new domains of interest. To accommodate personalized event-centric information needs, this paper introduces the few-shot Event Mention Retrieval (EMR) task: given a user-supplied query consisting of a handful of event mentions, return relevant event mentions found in a corpus. This formulation enables “query by example”, which drastically lowers the bar of specifying event-centric information needs. The retrieval setting also enables fuzzy search. We present an evaluation framework leveraging existing event datasets such as ACE. We also develop a Siamese Network approach, and show that it performs better than ad-hoc retrieval models in the few-shot EMR setting.
</p></blockquote></details>


## Semi-supervised learning 
[:arrow_up:](#table-of-contents)
### 2009



<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W09-2209/">Can One Language Bootstrap the Other: A Case Study on Event Extraction</a> by<i> Zheng Chen, Heng Ji</i></summary><blockquote><p align="justify">
This paper proposes a new bootstrapping framework using cross-lingual information projection. We demonstrate that this framework is particularly effective for a challenging NLP task which is situated at the end of a pipeline and thus suffers from the errors propagated from up- stream processing and has low-performance baseline. Using Chinese event extraction as a case study and bitexts as a new source of information, we present three bootstrapping techniques. We first conclude that the standard mono-lingual bootstrapping approach is not so effective. Then we exploit a second approach that potentially benefits from the extra information captured by an English event extraction system and projected into Chinese. Such a cross-lingual scheme produces significant performance gain. Finally we show that the combination of mono-lingual and cross-lingual information in bootstrapping can further enhance the performance. Ultimately this new framework obtained 10.1% relative improvement in trigger labeling (F-measure) and 9.5% relative improvement in argument-labeling.
</p></blockquote></details>

### 2011

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P11-2045/">Can Document Selection Help Semi-supervised Learning? A Case Study On Event Extraction</a> by<i> Shasha Liao, Ralph Grishman</i></summary><blockquote><p align="justify">
Annotating training data for event extraction is tedious and labor-intensive. Most current event extraction tasks rely on hundreds of annotated documents, but this is often not enough. In this paper, we present a novel self-training strategy, which uses Information Retrieval (IR) to collect a cluster of related documents as the resource for bootstrapping. Also, based on the particular characteristics of this corpus, global inference is applied to provide more confident and informative data selection. We compare this approach to self-training on a normal newswire corpus and show that IR can provide a better corpus for bootstrapping and that global inference can further improve instance selection. We obtain gains of 1.7% in trigger labeling and 2.3% in role labeling through IR and an additional 1.1% in trigger labeling and 1.3% in role labeling by applying global inference.
</p></blockquote></details>


 

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/I11-1080/">Using Prediction from Sentential Scope to Build a Pseudo Co-Testing Learner for Event Extraction</a> by<i> Shasha Liao, Ralph Grishman</i></summary><blockquote><p align="justify">
Event extraction involves the identification of instances of a type of event, along with their attributes and participants. Developing a training corpus by annotating events in text is very labor intensive, and so selecting informative instances to annotate can save a great deal of manual work. We present an active learning (AL) strategy, pseudo co-testing, based on one view from a classifier aiming to solve the original problem of event extraction, and another view from a classifier aiming to solve a coarser granularity task. As the second classifier can provide more graded matching from a wider scope, we can build a set of pseudocontention-points which are very informative, and can speed up the AL process. Moreover, we incorporate multiple selection criteria into the pseudo cotesting, seeking training examples that are informative, representative, and varied. Experiments show that pseudo co-testing can reduce annotation labor by 81%; incorporating multiple selection criteria reduces the labor by a further 7%.
</p></blockquote></details>



<details>
<summary>3. <a href="https://www.aclweb.org/anthology/P11-1040/">Event Discovery in Social Media Feeds</a> by<i> Edward Benson, Aria Haghighi, Regina Barzilay</i></summary><blockquote><p align="justify">
We present a novel method for record extraction from social streams such as Twitter. Unlike typical extraction setups, these environments are characterized by short, one sentence messages with heavily colloquial speech. To further complicate matters, individual messages may not express the full relation to be uncovered, as is often assumed in extraction tasks. We develop a graphical model that addresses these problems by learning a latent set of records and a record-message alignment simultaneously; the output of our model is a set of canonical records, the values of which are consistent with aligned messages. We demonstrate that our approach is able to accurately induce event records from Twitter messages, evaluated against events from a local city guide. Our method achieves significant error reduction over baseline methods.
</p></blockquote></details>

### 2013 


<details>
<summary>1. <a href="https://www.ncbi.nlm.nih.gov/pubmed/24565105">Semi-supervised method for biomedical event extraction</a> by<i> Wang J, Xu Q, Lin H, Yang Z, Li Y</i></summary><blockquote><p align="justify">
Background

Biomedical extraction based on supervised machine learning still faces the problem that a limited labeled dataset does not saturate the learning method. Many supervised learning algorithms for bio-event extraction have been affected by the data sparseness. 

Methods

In this study, a semi-supervised method for combining labeled data with large scale of unlabeled data is presented to improve the performance of biomedical event extraction. We propose a set of rich feature vector, including a variety of syntactic features and semantic features, such as N-gram features, walk subsequence features, predicate argument structure (PAS) features, especially some new features derived from a strategy named Event Feature Coupling Generalization (EFCG). The EFCG algorithm can create useful event recognition features by making use of the correlation between two sorts of original features explored from the labeled data, while the correlation is computed with the help of massive amounts of unlabeled data. This introduced EFCG approach aims to solve the data sparse problem caused by limited tagging corpus, and enables the new features to cover much more event related information with better generalization properties. 

Results

The effectiveness of our event extraction system is evaluated on the datasets from the BioNLP Shared Task 2011 and PubMed. Experimental results demonstrate the state-of-the-art performance in the fine-grained biomedical information extraction task. 

Conclusions

Limited labeled data could be combined with unlabeled data to tackle the data sparseness problem by means of our EFCG approach, and the classified capability of the model was enhanced through establishing a rich feature set by both labeled and unlabeled datasets. So this semi-supervised learning approach could go far towards improving the performance of the event extraction system. To the best of our knowledge, it was the first attempt at combining labeled and unlabeled data for tasks related biomedical event extraction. 
</p></blockquote></details>



<details>
<summary>2. <a href="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-175">Wide coverage biomedical event extraction using multiple partially overlapping corpora</a>	 by<i>Makoto Miwa, Sampo Pyysalo, Tomoko Ohta , Sophia Ananiadou </i></summary><blockquote><p align="justify">
Background

Biomedical events are key to understanding physiological processes and disease, and wide coverage extraction is required for comprehensive automatic analysis of statements describing biomedical systems in the literature. In turn, the training and evaluation of extraction methods requires manually annotated corpora. However, as manual annotation is time-consuming and expensive, any single event-annotated corpus can only cover a limited number of semantic types. Although combined use of several such corpora could potentially allow an extraction system to achieve broad semantic coverage, there has been little research into learning from multiple corpora with partially overlapping semantic annotation scopes.

Results

We propose a method for learning from multiple corpora with partial semantic annotation overlap, and implement this method to improve our existing event extraction system, EventMine. An evaluation using seven event annotated corpora, including 65 event types in total, shows that learning from overlapping corpora can produce a single, corpus-independent, wide coverage extraction system that outperforms systems trained on single corpora and exceeds previously reported results on two established event extraction tasks from the BioNLP Shared Task 2011.

Conclusions

The proposed method allows the training of a wide-coverage, state-of-the-art event extraction system from multiple corpora with partial semantic annotation overlap. The resulting single model makes broad-coverage extraction straightforward in practice by removing the need to either select a subset of compatible corpora or semantic types, or to merge results from several models trained on different individual corpora. Multi-corpus learning also allows annotation efforts to focus on covering additional semantic types, rather than aiming for exhaustive coverage in any single annotation effort, or extending the coverage of semantic types annotated in existing corpora.
</p></blockquote></details>

### 2014

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/L14-1091/">Event Extraction Using Distant Supervision</a> by<i> Kevin Reschke, Martin Jankowiak, Mihai Surdeanu, Christopher Manning, Daniel Jurafsky</i></summary><blockquote><p align="justify">
Distant supervision is a successful paradigm that gathers training data for information extraction systems by automatically aligning vast databases of facts with text. Previous work has demonstrated its usefulness for the extraction of binary relations such as a person’s employer or a film’s director. Here, we extend the distant supervision approach to template-based event extraction, focusing on the extraction of passenger counts, aircraft types, and other facts concerning airplane crash events. We present a new publicly available dataset and event extraction task in the plane crash domain based on Wikipedia infoboxes and newswire text. Using this dataset, we conduct a preliminary evaluation of four distantly supervised extraction models which assign named entity mentions in text to entries in the event template. Our results indicate that joint inference over sequences of candidate entity mentions is beneficial. Furthermore, we demonstrate that the SEARN algorithm outperforms a linear-chain CRF and strong baselines with local inference. 
</p></blockquote></details>



<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P14-2136/">Bilingual Event Extraction: a Case Study on Trigger Type Determination</a> by<i> Zhu Zhu, Shoushan Li, Guodong Zhou, Rui Xia</i></summary><blockquote><p align="justify">
Event extraction generally suffers from the data sparseness problem. In this paper, we address this problem by utilizing the labeled data from two different languages. As a preliminary study, we mainly focus on the subtask of trigger type determination in event extraction. To make the training data in different languages help each other, we propose a uniform text representation with bilingual features to represent the samples and handle the difficulty of locating the triggers in the translated text from both monolingual and bilingual perspectives. Empirical studies demonstrate the effectiveness of the proposed approach to bilingual classification on trigger type determination.
</p></blockquote></details>

### 2015 

  

<details>
<summary>1. <a href="https://www.semanticscholar.org/paper/Modeling-Event-Extraction-via-Multilingual-Data-Hsi-Carbonell/fabcbadcb824a7bbe51f21a8492f9ba234cb695d">Modeling Event Extraction via Multilingual Data Sources</a> by<i> Andrew Hsi, Jaime G. Carbonell, Yiming Yang</i></summary><blockquote><p align="justify">
In this paper, we describe our system for the TAC KBP 2015 Event track. We focus in particular on development of multilingual event extraction through the combination of language-dependent and languageindependent features. Our system specifically handles texts in both English and Chinese, but is designed in a manner to be extendable to new languages. Our experiments on the ACE2005 corpus show promising results for future development. 
</p></blockquote></details>

### 2016

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W16-1618/">A Two-stage Approach for Extending Event Detection to New Types via Neural Networks</a> by<i> Thien Huu Nguyen, Lisheng Fu, Kyunghyun Cho, Ralph Grishman</i></summary><blockquote><p align="justify">
We study the event detection problem in the new type extension setting. In particular, our task involves identifying the event instances of a target type that is only specified by a small set of seed instances in text. We want to exploit the large amount of training data available for the other event types to improve the performance of this task. We compare the convolutional neural network model and the feature-based method in this type extension setting to investigate their effectiveness. In addition, we propose a two-stage training algorithm for neural networks that effectively transfers knowledge from the other event types to the target type. The experimental results show that the proposed algorithm outperforms strong baselines for this task.
</p></blockquote></details>


<details>
<summary>2. <a href="https://www.aclweb.org/anthology/D16-1038/">Event Detection and Co-reference with Minimal Supervision</a> by<i> Haoruo Peng, Yangqiu Song, Dan Roth</i></summary><blockquote><p align="justify">
An important aspect of natural language understanding involves recognizing and categorizing events and the relations among them. However, these tasks are quite subtle and annotating training data for machine learning based approaches is an expensive task, resulting in supervised systems that attempt to learn complex models from small amounts of data, which they over-fit. This paper addresses this challenge by developing an event detection and co-reference system with minimal supervision, in the form of a few event examples. We view these tasks as semantic similarity problems between event mentions or event mentions and an ontology of types, thus facilitating the use of large amounts of out of domain text data. Notably, our semantic relatedness function exploits the structure of the text by making use of a semantic-role-labeling based representation of an event. We show that our approach to event detection is competitive with the top supervised methods. More significantly, we outperform stateof-the-art supervised methods for event coreference on benchmark data sets, and support significantly better transfer across domains.
</p></blockquote></details>



<details>
<summary>3. <a href="https://www.aclweb.org/anthology/P16-1201/">Leveraging FrameNet to Improve Automatic Event Detection</a> by<i> Shulin Liu, Yubo Chen, Shizhu He, Kang Liu, Jun Zhao</i></summary><blockquote><p align="justify">
Frames defined in FrameNet (FN) share highly similar structures with events in ACE event extraction program. An event in ACE is composed of an event trigger and a set of arguments. Analogously, a frame in FN is composed of a lexical unit and a set of frame elements, which play similar roles as triggers and arguments of ACE events respectively. Besides having similar structures, many frames in FN actually express certain types of events. The above observations motivate us to explore whether there exists a good mapping from frames to event-types and if it is possible to improve event detection by using FN. In this paper, we propose a global inference approach to detect events in FN. Further, based on the detected results, we analyze possible mappings from frames to event-types. Finally, we improve the performance of event detection and achieve a new state-of-the-art result by using the events automatically detected from FN.
</p></blockquote></details>



<details>
<summary>4. <a href="https://www.aclweb.org/anthology/C16-1114/">Leveraging multilingual training for limited resource event extraction</a> by<i> Andrew Hsi, Yiming Yang, Jaime Carbonell, Ruochen Xu </i></summary><blockquote><p align="justify">
Event extraction has become one of the most important topics in information extraction, but to date, there is very limited work on leveraging cross-lingual training to boost performance. We propose a new event extraction approach that trains on multiple languages using a combination of both language-dependent and language-independent features, with particular focus on the case where target domain training data is of very limited size. We show empirically that multilingual training can boost performance for the tasks of event trigger extraction and event argument extraction on the Chinese ACE 2005 dataset.
</p></blockquote></details>

<details>
<summary>5. <a href="http://aclweb.org/anthology/P16-3003">Identifying Potential Adverse Drug Events in Tweets Using Bootstrapped Lexicons</a> by<i> Benzschawel, Eric </i></summary><blockquote><p align="justify">
Adverse drug events (ADEs) are medical complications co-occurring with a period of drug usage. Identiﬁcation of ADEs is a primary way of evaluating available quality of care. As more social media users begin discussing their drug experiences online, public data becomes available for researchers to expand existing electronic ADE reporting systems, though non-standard language inhibits ease of analysis. In this study, portions of a new corpus of approximately 160,000 tweets were used to create a lexicon-driven ADE detection system using semi-supervised, pattern-based bootstrapping. This method was able to identify misspellings, slang terms, and other non-standard language features of social media data to drive a competitive ADE detection system.
</p></blockquote></details>

### 2017 

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P17-1038/">Automatically Labeled Data Generation for Large Scale Event Extraction</a> by<i> Yubo Chen, Shulin Liu, Xiang Zhang, Kang Liu, Jun Zhao</i></summary><blockquote><p align="justify">
Modern models of event extraction for tasks like ACE are based on supervised learning of events from small hand-labeled data. However, hand-labeled training data is expensive to produce, in low coverage of event types, and limited in size, which makes supervised methods hard to extract large scale of events for knowledge base population. To solve the data labeling problem, we propose to automatically label training data for event extraction via world knowledge and linguistic knowledge, which can detect key arguments and trigger words for each event type and employ them to label events in texts automatically. The experimental results show that the quality of our large scale automatically labeled data is competitive with elaborately human-labeled data. And our automatically labeled data can incorporate with human-labeled data, then improve the performance of models learned from these data.
</p></blockquote></details>


<details>
<summary>2. <a href="https://arxiv.org/abs/1712.03665">Scale Up Event Extraction Learning via Automatic Training Data Generation</a> by<i> Ying Zeng, Yansong Feng, Rong Ma, Zheng Wang, Rui Yan, Chongde Shi, Dongyan Zhao </i></summary><blockquote><p align="justify">
The task of event extraction has long been investigated in a supervised learning paradigm, which is bound by the number and the quality of the training instances. Existing training data must be manually generated through a combination of expert domain knowledge and extensive human involvement. However, due to drastic efforts required in annotating text, the resultant datasets are usually small, which severally affects the quality of the learned model, making it hard to generalize. Our work develops an automatic approach for generating training data for event extraction. Our approach allows us to scale up event extraction training instances from thousands to hundreds of thousands, and it does this at a much lower cost than a manual approach. We achieve this by employing distant supervision to automatically create event annotations from unlabelled text using existing structured knowledge bases or tables.We then develop a neural network model with post inference to transfer the knowledge extracted from structured knowledge bases to automatically annotate typed events with corresponding arguments in text.We evaluate our approach by using the knowledge extracted from Freebase to label texts from Wikipedia articles. Experimental results show that our approach can generate a large number of high quality training instances. We show that this large volume of training data not only leads to a better event extractor, but also allows us to detect multiple typed events. 
</p></blockquote></details>

### 2018

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/N18-2058/">Semi-Supervised Event Extraction with Paraphrase Clusters</a> by<i> James Ferguson, Colin Lockard, Daniel Weld, Hannaneh Hajishirzi</i></summary><blockquote><p align="justify">
Supervised event extraction systems are limited in their accuracy due to the lack of available training data. We present a method for self-training event extraction systems by bootstrapping additional training data. This is done by taking advantage of the occurrence of multiple mentions of the same event instances across newswire articles from multiple sources. If our system can make a high-confidence extraction of some mentions in such a cluster, it can then acquire diverse training examples by adding the other mentions as well. Our experiments show significant performance improvements on multiple event extractors over ACE 2005 and TAC-KBP 2015 datasets.
</p></blockquote></details>



<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P18-1201/">Zero-Shot Transfer Learning for Event Extraction</a> by<i> Lifu Huang, Heng Ji, Kyunghyun Cho, Ido Dagan, Sebastian Riedel, Clare Voss </i>(<a href="https://github.com/wilburOne/ZeroShotEvent">Github</a>)</summary><blockquote><p align="justify">
Most previous supervised event extraction methods have relied on features derived from manual annotations, and thus cannot be applied to new event types without extra annotation effort. We take a fresh look at event extraction and model it as a generic grounding problem: mapping each event mention to a specific type in a target event ontology. We design a transferable architecture of structural and compositional neural networks to jointly represent and map event mentions and types into a shared semantic space. Based on this new framework, we can select, for each event mention, the event type which is semantically closest in this space as its type. By leveraging manual annotations available for a small set of existing event types, our framework can be applied to new unseen event types without additional manual annotations. When tested on 23 unseen event types, our zero-shot framework, without manual annotations, achieved performance comparable to a supervised model trained from 3,000 sentences annotated with 500 event mentions.
</p></blockquote></details>

 

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/C18-1075/">Open-domain event detection using distant supervision</a> by<i> Jun Araki, Teruko Mitamura</i></summary><blockquote><p align="justify">
This paper introduces open-domain event detection, a new event detection paradigm to address issues of prior work on restricted domains and event annotation. The goal is to detect all kinds of events regardless of domains. Given the absence of training data, we propose a distant supervision method that is able to generate high-quality training data. Using a manually annotated event corpus as gold standard, our experiments show that despite no direct supervision, the model outperforms supervised models. This result indicates that the distant supervision enables robust event detection in various domains, while obviating the need for human annotation of events.
</p></blockquote></details>

### 2019



<details>
<summary>1. <a href="https://www.aclweb.org/anthology/N19-1105/">Adversarial Training for Weakly Supervised Event Detection</a> by<i> Xiaozhi Wang, Xu Han, Zhiyuan Liu, Maosong Sun, Peng Li</i> (<a href="https://github.com/thunlp/Adv-ED">Github</a>)</summary><blockquote><p align="justify">
Modern weakly supervised methods for event detection (ED) avoid time-consuming human annotation and achieve promising results by learning from auto-labeled data. However, these methods typically rely on sophisticated pre-defined rules as well as existing instances in knowledge bases for automatic annotation and thus suffer from low coverage, topic bias, and data noise. To address these issues, we build a large event-related candidate set with good coverage and then apply an adversarial training mechanism to iteratively identify those informative instances from the candidate set and filter out those noisy ones. The experiments on two real-world datasets show that our candidate selection and adversarial training can cooperate together to obtain more diverse and accurate training data for ED, and significantly outperform the state-of-the-art methods in various weakly supervised scenarios. The datasets and source code can be obtained from https://github.com/thunlp/Adv-ED.
</p></blockquote></details>



<details>
<summary>2. <a href="https://ieeexplore.ieee.org/document/8643786">Joint Event Extraction Based on Hierarchical Event Schemas From FrameNet</a> by<i>  Wei Li ; Dezhi Cheng ; Lei He ; Yuanzhuo Wang ; Xiaolong Jin</i></summary><blockquote><p align="justify">
Event extraction is useful for many practical applications, such as news summarization and information retrieval. However, the popular automatic context extraction (ACE) event extraction program only defines very limited and coarse event schemas, which may not be suitable for practical applications. FrameNet is a linguistic corpus that defines complete semantic frames and frame-to-frame relations. As frames in FrameNet share highly similar structures with event schemas in ACE and many frames actually express events, we propose to redefine the event schemas based on FrameNet. Specifically, we extract frames expressing event information from FrameNet and leverage the frame-to-frame relations to build a hierarchy of event schemas that are more fine-grained and have much wider coverage than ACE. Based on the new event schemas, we propose a joint event extraction approach that leverages the hierarchical structure of event schemas and frame-to-frame relations in FrameNet. The extensive experiments have verified the advantages of our hierarchical event schemas and the effectiveness of our event extraction model. We further apply the results of our event extraction model on news summarization. The results show that the summarization approach based on our event extraction model achieves significant better performance than several state-of-the-art summarization approaches, which also demonstrates that the hierarchical event schemas and event extraction model are promising to be used in the practical applications.
</p></blockquote></details>



<details>
<summary>3. <a href="https://arxiv.org/abs/1904.07535">Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction</a> by<i> Shun Zheng, Wei Cao, Wei Xu, Jiang Bian </i></summary><blockquote><p align="justify">
Most existing event extraction (EE) methods merely extract event arguments within the sentence scope. However, such sentence-level EE methods struggle to handle soaring amounts of documents from emerging applications, such as finance, legislation, health, etc., where event arguments always scatter across different sentences, and even multiple such event mentions frequently co-exist in the same document. To address these challenges, we propose a novel end-to-end model, Doc2EDAG, which can generate an entity-based directed acyclic graph to fulfill the document-level EE (DEE) effectively. Moreover, we reformalize a DEE task with the no-trigger-words design to ease the document-level event labeling. To demonstrate the effectiveness of Doc2EDAG, we build a large-scale real-world dataset consisting of Chinese financial announcements with the challenges mentioned above. Extensive experiments with comprehensive analyses illustrate the superiority of Doc2EDAG over state-of-the-art methods. Data and codes can be found at this https URL. 
</p></blockquote></details>



<details>
<summary>4. <a href="https://www.aclweb.org/anthology/N19-4019/">Multilingual Entity, Relation, Event and Human Value Extraction</a> by<i> Manling Li, Ying Lin, Joseph Hoover, Spencer Whitehead, Clare Voss, Morteza Dehghani, Heng Ji </i> (<a href="https://github.com/dwadden/dygiepp">Github</a>)</summary><blockquote><p align="justify">
This paper demonstrates a state-of-the-art end-to-end multilingual (English, Russian, and Ukrainian) knowledge extraction system that can perform entity discovery and linking, relation extraction, event extraction, and coreference. It extracts and aggregates knowledge elements across multiple languages and documents as well as provides visualizations of the results along three dimensions: temporal (as displayed in an event timeline), spatial (as displayed in an event heatmap), and relational (as displayed in entity-relation networks). For our system to further support users’ analyses of causal sequences of events in complex situations, we also integrate a wide range of human moral value measures, independently derived from region-based survey, into the event heatmap. This system is publicly available as a docker container and a live demo.
</p></blockquote></details>

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/D19-1068/">Neural Cross-Lingual Event Detection with Minimal Parallel Resources</a> by<i> Jian Liu, Yubo Chen, Kang Liu, Jun Zhao</i></summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>



<details>
<summary>6. <a href="https://www.aclweb.org/anthology/D19-5102/">Financial Event Extraction Using Wikipedia-Based Weak Supervision</a> by<i> Liat Ein-Dor, Ariel Gera, Orith Toledo-Ronen, Alon Halfon, Benjamin Sznajder, Lena Dankin, Yonatan Bilu, Yoav Katz, Noam Slonim</i></summary><blockquote><p align="justify">
Extraction of financial and economic events from text has previously been done mostly using rule-based methods, with more recent works employing machine learning techniques. This work is in line with this latter approach, leveraging relevant Wikipedia sections to extract weak labels for sentences describing economic events. Whereas previous weakly supervised approaches required a knowledge-base of such events, or corresponding financial figures, our approach requires no such additional data, and can be employed to extract economic events related to companies which are not even mentioned in the training data.
</p></blockquote></details>


## Unsupervised learning
[:arrow_up:](#table-of-contents)
### 1998



<details>
<summary>1. <a href="https://dl.acm.org/citation.cfm?doid=290941.290953">A study on retrospective and on-line event detection</a> by<i> Yiming Yang, Tom Pierce, Jaime Carbonell </i></summary><blockquote><p align="justify">
This paper investigates the use and extension of text retrieval and clustering techniques for event detection. The task is to automatically detect novel events from a temporally-ordered stream of news stories, either retrospectively or as the stories arrive. We applied hierarchical and non-hierarchical document clustering algorithms to a corpus of 15,836 stories, focusing on the exploitation of both content and temporal information. We found the resulting cluster hierarchies highly informative for retrospective detection of previously unidentified events, effectively supporting both query-free and
query-driven retrieval. We also found that temporal distribution patterns of document clusters provide useful information for improvement in both retrospective detection and on-line detection of novel events. In an evaluation using manually labelled events to judge the system-detected events, we obtained a result of 82% in the F1 measure for retrospective detection, and a F1 value of 42% for on-line detection
</p></blockquote></details>

### 2001



<details>
<summary>1. <a href="https://dl.acm.org/citation.cfm?doid=383952.384068">Combining Semantic and Syntactic Document Classifiers to Improve First Story Detection</a> by<i> Nicola Stokes, Joe Carthy</i></summary><blockquote><p align="justify">
In this paper we describe a type of data fusion involving the combination of evidence derived from multiple document representations. Our aim is to investigate if a composite representation can improve the online detection of novel events in a stream of broadcast news stories. This classification process otherwise known as first story detection FSD (or in the Topic Detection and Tracking pilot study as online new event detection [1]), is one of three main classification tasks defined by the TDT initiative. Our composite document representation consists of a semantic representation (based on the lexical chains derived from a text) and a syntactic representation (using proper nouns). Using the TDT1 evaluation methodology, we evaluate a number of document representation combinations using these document classifiers.
</p></blockquote></details>



<details>
<summary>2. <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.9651&rep=rep1&type=pdf">A probabilistic model for retrospective news event detection</a> by<i> Zhiwei Li, Bin Wang, Mingjing Li, Wei-Ying Ma</i></summary><blockquote><p align="justify">
Retrospective news event detection (RED) is defined as the discovery of previously unidentified events in historical news corpus. Although both the contents and time information of news articles are helpful to RED, most researches focus on the utilization of the contents of news articles. Few research works have been carried out on finding better usages of time information. In this paper, we do some explorations on both directions based on the following two characteristics of news articles. On the one hand, news articles are always aroused by events; on the other hand, similar articles reporting the same event often redundantly appear on many news sources. The former hints a generative model of news articles, and the latter provides data enriched environments to perform RED. With consideration of these characteristics, we propose a probabilistic model to incorporate both content and time information in a unified framework. This model gives new representations of both news articles and news events. Furthermore, based on this approach, we build an interactive RED system, HISCOVERY, which provides additional functions to present events, Photo Story and Chronicle.
</p></blockquote></details>


### 2003 

<details>
<summary>1. <a href="https://www.sciencedirect.com/science/article/pii/S0957417403000629">Ontology-based fuzzy event extraction agent for Chinese e-news summarization</a> by<i> Chang-Shing Lee, Yea-Juan Chen, Zhi-Wei Jian</i></summary><blockquote><p align="justify">
An Ontology-based Fuzzy Event Extraction (OFEE) agent for Chinese e-news summarization is proposed in this article. The OFEE agent contains Retrieval Agent (RA), Document Processing Agent (DPA) and Fuzzy Inference Agent (FIA) to perform the event extraction for Chinese e-news summarization. First, RA automatically retrieves Internet e-news periodically, stores them into the e-news repository, and sends them to DPA for document processing. Then, the DPA will utilize the Chinese Part-of-speech (POS) tagger provided by Chinese knowledge information processing group to process the retrieved e-news and filter the Chinese term set by Chinese term filter. Next, the FIA and Event Ontology Filter (EOF) extract the e-news event ontology based on the Chinese term set and domain ontology. Finally, the Summarization Agent (SA) will summarize the e-news by the extracted-event ontology. By the simulation, the proposed method can summarize the Chinese weather e-news effectively.
</p></blockquote></details>

### 2004

 

<details>
<summary>1. <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.152.5641&rep=rep1&type=pdf">Event threading within news topics</a> by<i> Ramesh Nallapati, Ao Feng, Fuchun Peng, James Allan</i></summary><blockquote><p align="justify">
With the overwhelming volume of online news available today, there is an increasing need for automatic techniques to analyze and present news to the user in a meaningful and efficient manner. Previous research focused only on organizing news stories by their topics into a flat hierarchy. We believe viewing a news topic as a flat collection of stories is too restrictive and inefficient for a user to understand the topic quickly. 
 In this work, we attempt to capture the rich structure of events and their dependencies in a news topic through our event models. We call the process of recognizing events and their dependencies <i>event threading</i>. We believe our perspective of modeling the structure of a topic is more effective in capturing its semantics than a flat list of on-topic stories.
 We formally define the novel problem, suggest evaluation metrics and present a few techniques for solving the problem. Besides the standard word based features, our approaches take into account novel features such as temporal locality of stories for event recognition and time-ordering for capturing dependencies. Our experiments on a manually labeled data sets show that our models effectively identify the events and capture dependencies among them. 
</p></blockquote></details>

### 2005

  

<details>
<summary>1. <a href="https://link.springer.com/chapter/10.1007/11581772_66">A system for detecting and tracking Internet news event</a> by<i> Zhen Lei, Ling-da Wu, Ying Zhang, Yu-chi Liu</i></summary><blockquote><p align="justify">
News event detection is the task of discovering relevant, yet previously unreported real-life events and reporting it to users in human-readable form, while event tracking aims to automatically assign event labels to news stories when they arrive. A new method and system for performing the event detection and tracking task is proposed in this paper. The event detection and tracking method is based on subject extraction and an improved support vector machine (SVM), in which subject concepts can concisely and precisely express the meaning of a longer text. The improved SVM first prunes the negative examples, reserves and deletes a negative sample according to distance and class label, then trains the new set with SVM to obtain a classifier and maps the SVM outputs into probabilities. The experimental results with the real-world data sets indicate the proposed method is feasible and advanced.
</p></blockquote></details>


### 2012



<details>
<summary>1. <a href="https://www.aclweb.org/anthology/E12-1034/">Skip N-grams and Ranking Functions for Predicting Script Events</a> by<i> Bram Jans, Steven Bethard, Ivan Vulić, Marie Francine Moens</i></summary><blockquote><p align="justify">
In this paper, we extend current state-of-the-art research on unsupervised acquisition of scripts, that is, stereotypical and frequently observed sequences of events. We design, evaluate and compare different methods for constructing models for script event prediction: given a partial chain of events in a script, predict other events that are likely to belong to the script. Our work aims to answer key questions about how best to (1) identify representative event chains from a source text, (2) gather statistics from the event chains, and (3) choose ranking functions for predicting new script events. We make several contributions, introducing skip-grams for collecting event statistics, designing improved methods for ranking event predictions, defining a more reliable evaluation metric for measuring predictiveness, and providing a systematic analysis of the various event prediction models.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P12-1088/">Automatic Event Extraction with Structured Preference Modeling</a> by<i> Wei Lu, Dan Roth</i></summary><blockquote><p align="justify">
This  paper  presents  a  novel  sequence  labeling model based on the latent-variable semi-Markov conditional random fields for jointly extracting argument roles of events from texts. The model takes in coarse mention and type information and predicts argument roles for a given event template. This paper addresses the event extraction problem in a primarily  unsupervised  setting, where no labeled training instances are available. Our key contribution is a novel learning framework called structured preference modeling (PM), that  allows  arbitrary  preference to be assigned to certain structures during the learning procedure.  We establish and discuss connections between this framework and other existing works. We show empirically that the structured preferences are crucial to the success of our task. Our  model,  trained  without  annotated  data  and  with  a  small  number of structured preferences, yields performance competitive to some baseline supervised  approaches.
</p></blockquote></details>

### 2014


<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W14-2905/">Unsupervised Techniques for Extracting and Clustering Complex Events in News</a> by<i> Delia Rusu, James Hodson, Anthony Kimball</i></summary><blockquote><p align="justify">
Structured machine-readable representations of news articles can radically change the way we interact with information. One step towards obtaining these representations is event extraction - the identification of event triggers and arguments in text. With previous approaches mainly focusing on classifying events into a small set of predefined types, we analyze unsupervised techniques for complex event extraction. In addition to extracting event mentions in news articles, we aim at obtaining a more general representation by disambiguating to concepts defined in knowledge bases. These concepts are further used as features in a clustering application. Two evaluation settings highlight the advantages and shortcomings of the proposed approach.
</p></blockquote></details>

<details>
<summary>2. <a href="http://aclweb.org/anthology/P14-1084">Modelling Events through Memory-based, Open-IE Patterns for Abstractive Summarization</a> by<i> Pighin, Daniele and Cornolti, Marco and Alfonseca, Enrique and Filippova, Katja </i></summary><blockquote><p align="justify">
Abstractive text summarization of news requires a way of representing events, such as a collection of pattern clusters in which every cluster represents an event (e.g., marriage) and every pattern in the cluster is a way of expressing the event (e.g., X married Y, X and Y tied the knot). We compare three ways of extracting event patterns: heuristics-based, compression based and memory-based. While the former has been used previously in multidocument abstraction, the latter two have never been used for this task. Compared with the ﬁrst two techniques, the memory based method allows for generating signiﬁcantly more grammatical and informative sentences, at the cost of searching a vast space of hundreds of millions of parse trees of known grammatical utterances. To this end, we introduce a data structure and a search method that make it possible to efﬁciently extrapolate from every sentence the parse sub-trees that match against any of the stored utterances.
</p></blockquote></details>

### 2016 

 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/C16-1309/">Event detection with burst information networks</a> by<i> Tao Ge, Lei Cui, Baobao Chang, Zhifang Sui, Ming Zhou</i></summary><blockquote><p align="justify">
Retrospective event detection is an important task for discovering previously unidentified events in a text stream. In this paper, we propose two fast centroid-aware event detection models based on a novel text stream representation – Burst Information Networks (BINets) for addressing the challenge. The BINets are time-aware, efficient and can be easily analyzed for identifying key information (centroids). These advantages allow the BINet-based approaches to achieve the state-of-the-art performance on multiple datasets, demonstrating the efficacy of BINets for the task of event detection.
</p></blockquote></details>

<details>
<summary>2. <a href="http://aclweb.org/anthology/P16-1026">Jointly Event Extraction and Visualization on Twitter via Probabilistic Modelling</a> by<i> Zhou, Deyu and Gao, Tianmeng and He, Yulan </i></summary><blockquote><p align="justify">
Event extraction from texts aims to detect structured information such as what has happened, to whom, where and when. Event extrahttps://www.aclweb.org/anthology/W13-2322.bibction and visualization are typically considered as two different tasks. In this paper, we propose a novel approach based on probabilistic modelling to jointly extract and visualize events from tweets where both tasks beneﬁt from each other. We model each event as a joint distribution over named entities, a date, a location and event-related keywords. Moreover, both tweets and event instances are associated with coordinates in the visualization space. The manifold assumption that the intrinsic geometry of tweets is a low-rank, non-linear manifold within the high-dimensional space is incorporated into the learning framework using a regularization. Experimental results show that the proposed approach can effectively deal with both event extraction and visualization and performs remarkably better than both the state-of-the-art event extraction method and a pipeline approach for event extraction and visualization.
</p></blockquote></details>


## Event coreference
[:arrow_up:](#table-of-contents)

### 1997

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W97-1311/">Event coreference for information extraction</a> by<i> Kevin Humphreys, Robert Gaizauskas, Saliha Azzam</i></summary><blockquote><p align="justify">
We propose a general approach for performing event coreference and for constructing complex event representations, such as those required for information extraction tasks. Our approach is based on a representation which allows a tight coupling between world or conceptual modelling and discourse modelling. The rep- resentation and the coreference mechanism are fully implemented within the LaSIE information extraction system where the mechanism is used for both object (noun phrase) and event coreference resolution.
</p></blockquote></details>

### 2007

<details>
<summary>1. <a href="https://ieeexplore.ieee.org/document/4338380">Unrestricted Coreference: Identifying Entities and Events in OntoNotes</a> by<i>  Sameer S. Pradhan , Lance Ramshaw , Ralph Weischedel , Jessica MacBride , Linnea Micciulla </i></summary><blockquote><p align="justify">
Most research in the field of anaphora or coreference detection has been limited to noun phrase coreference, usually on a restricted set of entities, such as ACE entities. In part, this has been due to the lack of corpus resources tagged with general anaphoric coreference. The OntoNotes project is creating a large-scale, accurate corpus for general anaphoric coreference that covers entities and events not limited to noun phrases or a limited set of entity types. The coreference layer in OntoNotes constitutes one part of a multi-layer, integrated annotation of shallow semantic structure in text. This paper presents an initial model for unrestricted coreference based on this data that uses a machine learning architecture with state-of-the-art features. Significant improvements can be expected from using such cross-layer information for training predictive models. This paper describes the coreference annotation in OntoNotes, presents the baseline model, and provides an analysis of the contribution of this new resource in the context of recent MUC and ACE results.
</p></blockquote></details>

### 2008

<details>
<summary>1. <a href="http://www.lrec-conf.org/proceedings/lrec2008/pdf/734_paper.pdf">A Linguistic Resource for Discovering Event Structures and Resolving Event Coreference</a> by<i> Cosmin Adrian Bejan, Sanda M. Harabagiu</i></summary><blockquote><p align="justify">
In this paper, we present a linguistic resource that annotates event structures in texts. We consider an event structure as a collection of events that interact with each other in a given situation. We interpret the inter actions between events as event relations. In this regard, we propose and annotate a set of six relations that best capture the concep t of event structure. These relations are: subevent, reason, purpose, enablement, precedence and related. A document from this resource can encode multiple event structures and an event structure can be described across multiple documents. In order to unify event structures, we also annotate inter- and intra-document event coreference. Moreover, we provide methodologies for automatic discovery of event structures from texts. First, we group the events that constitute an event structure into event clusters and then, we use supervised lear ning frameworks to classify the relations that exist between events from the same cluster.
</p></blockquote></details>

### 2009

<details>
<summary>1. <a href="https://papers.nips.cc/paper/3637-nonparametric-bayesian-models-for-unsupervised-event-coreference-resolution">Non parametric Bayesian Models for Unsupervised Event Coreference Resolution</a> by<i> Cosmin Adrian Bejan, Matthew Titsworth, Andrew Hickl, Sanda M. Harabagiu</i></summary><blockquote><p align="justify">
We present a sequence of unsupervised, nonparametric Bayesian models for clus- tering complex linguistic objects. In this approach, we consider a potentially infi- nite number of features and categorical outcomes. We evaluated these models for the task of within- and cross-document event coreference on two corpora. All the models we investigated show significant improvements when c ompared against an existing baseline for this task.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/W09-3208/">Graph-based Event Coreference Resolution</a> by<i> Zheng Chen, Heng Ji</i></summary><blockquote><p align="justify">
In this paper, we address the problem of event coreference resolution as specified in the Automatic Content Extraction (ACE 1) program. In contrast to entity coreference resolution, event coreference resolution has not received great attention from researchers. In this paper, we first demonstrate the diverse scenarios of event coreference by an example. We then model event coreference resolution as a spectral graph clustering problem and evaluate the clustering algorithm on ground truth event mentions using ECM F-Measure. We obtain the ECM-F scores of 0.8363 and 0.8312 respectively by using two methods for computing coreference matrices. 
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/W09-4303/">A Pairwise Event Coreference Model, Feature Impact and Evaluation for Event Coreference Resolution</a> by<i> Zheng Chen, Heng Ji, Robert Haralick</i></summary><blockquote><p align="justify">
In past years, there has been substantial work on the problem of entity coreference resolution whereas much less attention has been paid to event coreference resolution. Starting with some motivating examples, we formally state the problem of event coreference resolution in the ACE program, present an agglomerative clustering algorithm for the task, explore the feature impact in the event coreference model and compare three evaluation metrics that were previously adopted in entity coreference resolution: MUC F-Measure, B-Cubed F-Measure and ECM F-Measure.
</p></blockquote></details>

### 2010

<details>
<summary>1. <a href="https://dl.acm.org/doi/10.5555/1858681.1858824">Unsupervised event coreference resolution with rich linguistic features</a> by<i> Cosmin Adrian Bejan, Sanda M. Harabagiu</i></summary><blockquote><p align="justify">
This paper examines how a new class of nonparametric Bayesian models can be effectively applied to an open-domain event coreference task. Designed with the purpose of clustering complex linguistic objects, these models consider a potentially infinite number of features and categorical outcomes. The evaluation performed for solving both within- and cross-document event coreference shows significant improvements of the models when compared against two baselines for this task.
</p></blockquote></details>

### 2011

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/I11-1012/">A Unified Event Coreference Resolution by Integrating Multiple Resolvers</a> by<i> Bin Chen, Jian Su, Sinno Jialin Pan, Chew Lim Tan</i></summary><blockquote><p align="justify">
Event coreference is an important and complicated task in cascaded event template extraction and other natural language processing tasks. Despite its importance, it was merely discussed in previous studies. In this paper, we present a globally optimized coreference resolution system dedicated to various sophisticated event coreference phenomena. Seven resolvers for both event and object coreference cases are utilized, which include three new resolvers for event coreference resolution. Three enhancements are further proposed at both mention pair detection and chain formation levels. First, the object coreference resolvers are used to effectively reduce the false positive cases for event coreference. Second, A revised instance selection scheme is proposed to improve link level mention-pair model performances. Last but not least, an efficient and glo-bally optimized graph partitioning model is employed for coreference chain formation using spectral partitioning which allows the incorporation of pronoun coreference information. The three techniques contribute to a significant improvement of 8.54% in B 3 F-score for event co-reference resolution on OntoNotes 2.0 corpus.
</p></blockquote></details>


### 2012

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/D12-1045/">Joint Entity and Event Coreference Resolution across Documents</a> by<i> Heeyoung Lee, Marta Recasens, Angel Chang, Mihai Surdeanu, Dan Jurafsky</i></summary><blockquote><p align="justify">
We introduce a novel coreference resolution system that models entities and events jointly. Our iterative method cautiously constructs clusters of entity and event mentions using linear regression to model cluster merge operations. As clusters are built, information flows between entity and event clusters through features that model semantic role dependencies. Our system handles nominal and verbal events as well as entities, and our joint formulation allows information from event coreference to help entity coreference, and vice versa. In a cross-document domain with comparable documents, joint coreference resolution performs significantly better (over 3 CoNLL F1 points) than two strong baselines that resolve entities and events separately.
</p></blockquote></details>

<details>
<summary>2. <a href="https://ieeexplore.ieee.org/document/6188406">Improving event co-reference by context extraction and dynamic feature weighting</a> by<i>  Katie McConky , Rakesh Nagi , Moises Sudit , William Hughes </i></summary><blockquote><p align="justify">
Event co-reference is the process of identifying descriptions of the same event across sentences, documents, or structured databases. Existing event co-reference work focuses on sentence similarity models or feature based similarity models requiring slot filling. This work shows the effectiveness of using a hybrid approach where the similarity of two events is determined by a combination of the similarity of the two event descriptions, in addition to the similarity of the event context features of location and time. A dynamic weighting approach is taken to combine the three similarity scores together. The described approach provides several benefits including improving event resolution and requiring less reliance on sophisticated natural language processing.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.researchgate.net/publication/268585541_Event_Coreference_Resolution_using_Mincut_based_Graph_Clustering">Event Coreference Resolution using Mincut based Graph Clustering</a> by<i>  Sangeetha.S and Michael Arock </i></summary><blockquote><p align="justify">
To extract participants of an event instance, it is necessary to identify all the sentences that describe the event instance. The set of all sentences referring to the same event instance are said to be corefering each other. Our proposed approach formulates the event coreference resolution as a graph based clustering model. It identifies the corefering sentences using minimum cut (mincut) based on similarity score between each pair of sentences at various levels such as trigger word similarity, time stamp similarity, entity similarity and semantic similarity. It achieves good B-Cubed F-measure score with some loss in recall.
</p></blockquote></details>

### 2013 
<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P13-2083/">A Structured Distributional Semantic Model for Event Co-reference</a> by<i> Kartik Goyal, Sujay Kumar Jauhar, Huiying Li, Mrinmaya Sachan, Shashank Srivastava, Eduard Hovy</i></summary><blockquote><p align="justify">
In this paper we present a novel approach to modelling distributional semantics that represents meaning as distributions over relations in syntactic neighborhoods. We argue that our model approximates meaning in compositional conﬁgurations more effectively than standard distributional vectors or bag-of-words models. We test our hypothesis on the problem of judging event coreferentiality, which involves compositional interactions in the predicate-argument structure of sentences, and demonstrate that our model outperforms both state-of-the-art window-based word embeddings as well as simple approaches to compositional semantics previously employed in the literature.
</p></blockquote></details>

### 2014 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/W14-2910/">Evaluation for Partial Event Coreference</a> by<i> Jun Araki, Eduard Hovy, Teruko Mitamura</i></summary><blockquote><p align="justify">
This paper proposes an evaluation scheme to measure the performance of a systemthat detects hierarchical event structure forevent  coreference resolution. We  show that each system output is represented as a forest of unordered trees, and introduce the notion of conceptual event hierarchy to simplify the evaluation process.  We enumerate the desiderata for a similarity metric to measure the system  performance. We examine three metrics along with the desiderata, and show that metrics extended from  MUC  and  BLANC  are  more  adequate than a metric based on Simple Tree Matching.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/J14-2004/">Unsupervised Event Coreference Resolution</a> by<i> Cosmin Adrian Bejan, Sanda Harabagiu</i></summary><blockquote><p align="justify">
The task of event coreference resolution plays a critical role in many natural language processing applications such as information extraction, question answering, and topic detection and tracking. In this article, we describe a new class of unsupervised, nonparametric Bayesian models with the purpose of probabilistically inferring coreference clusters of event mentions from a collection of unlabeled documents. In order to infer these clusters, we automatically extract various lexical, syntactic, and semantic features for each event mention from the document collection. Extracting a rich set of features for each event mention allows us to cast event coreference resolution as the task of grouping together the mentions that share the same features (they have the same participating entities, share the same location, happen at the same time, etc.). Some of the most important challenges posed by the resolution of event coreference in an unsupervised way stem from (a) the choice of representing event mentions through a rich set of features and (b) the ability of modeling events described both within the same document and across multiple documents. Our first unsupervised model that addresses these challenges is a generalization of the hierarchical Dirichlet process. This new extension presents the hierarchical Dirichlet process’s ability to capture the uncertainty regarding the number of clustering components and, additionally, takes into account any finite number of features associated with each event mention. Furthermore, to overcome some of the limitations of this extension, we devised a new hybrid model, which combines an infinite latent class model with a discrete time series model. The main advantage of this hybrid model stands in its capability to automatically infer the number of features associated with each event mention from data and, at the same time, to perform an automatic selection of the most informative features for the task of event coreference. The evaluation performed for solving both within- and cross-document event coreference shows significant improvements of these models when compared against two baselines for this task.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/L14-1099/">SinoCoreferencer: An End-to-End Chinese Event Coreference Resolver</a> by<i> Chen Chen, Vincent Ng</i></summary><blockquote><p align="justify">
This paper describes the design, implementation, and evaluation of SinoCoreferencer, a publicly-available end-to-end ACE-style Chinese event coreference system that achieves state-of-the-art performance on the ACE 2005 corpus. SinoCoreferencer comprises eight information extraction system components, including those for entity extraction, entity coreference resolution, and event extraction. Its modular design makes it possible to run each component in a standalone manner, thus facilitating the development of high-level Chinese natural language applications that make use of any of these core information extraction components. To our knowledge, SinoCoreferencer is the first publicly-available Chinese event coreference resolution system.
</p></blockquote></details>


<details>
<summary>4. <a href="https://www.semanticscholar.org/paper/Using-a-sledgehammer-to-crack-a-nut-Lexical-and-Cybulska-Vossen/0fabeb29eee19ca80b6f424d8cd86ac52ac96eb0">Using a sledgehammer to crack a nut? Lexical diversity and event coreference resolution</a> by<i> Agata Cybulska, Piek T. J. M. Vossen</i></summary><blockquote><p align="justify">
In this paper we examine the representativeness of the EventCorefBank (ECB) (Bejan and Harabagiu, 2010) with regards to the language population of large-volume streams of news. The ECB corpus is one of the data sets used for evaluation of the task of event coreference resolution. Our analysis shows that the ECB in most cases covers one seminal event per domain, what considerably simplifies event and so language diversity that one comes across in the news. We augmented the corpus with a new corpus component, consisting of 502 texts, describing different instances of event types that were already captured by the 43 topics of the ECB, making it more representative of news articles on the web. The new ”ECB+” corpus is available for further research. 
</p></blockquote></details>

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/L14-1513/">Supervised Within-Document Event Coreference using Information Propagation</a> by<i> Zhengzhong Liu, Jun Araki, Eduard Hovy, Teruko Mitamura</i></summary><blockquote><p align="justify">
Event coreference is an important task for full text analysis. However, previous work uses a variety of approaches, sources and evaluation, making the literature confusing and the results incommensurate. We provide a description of the differences to facilitate future research. Second, we present a supervised method for event coreference resolution that uses a rich feature set and propagates information alternatively between events and their arguments, adapting appropriately for each type of argument.
</p></blockquote></details>

### 2015 

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/N15-1116/">Chinese Event Coreference Resolution: An Unsupervised Probabilistic Model Rivaling Supervised Resolvers</a> by<i> Chen Chen, Vincent Ng</i></summary><blockquote><p align="justify">
Recent work has successfully leveraged the semantic information extracted from lexical knowledge bases such as WordNet and FrameNet to improve English event coreference resolvers. The lack of comparable resources in other languages, however, has made the design of high-performance non-English event coreference resolvers, particularly those employing unsupervised models, very difficult. We propose a generative model for the under-studied task of Chinese event coreference resolution that rivals its supervised counterparts in performance when evaluated on the ACE 2005 corpus.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/W15-0801/">Translating Granularity of Event Slots into Features for Event Coreference Resolution.</a> by<i> Agata Cybulska, Piek Vossen</i></summary><blockquote><p align="justify">
Using clues from event semantics to solve coreference, we present an “event template” approach to cross-document event coreference resolution on news articles. The approach uses a pairwise model, in which event information is compared along five semantically motivated slots of an event template. The templates, filled in on the sentence level for every event mention from the data set, are used for supervised classification. In this study, we determine granularity of events and we use the grain size as a clue for solving event coreference. We experiment with a newly-created granularity ontology employing granularity levels of locations, times and human participants as well as event durations as features in event coreference resolution. The granularity ontology is available for research. Results show that determining granularity along semantic event slots, even on the sentence level exclusively, improves precision and solves event coreference with scores comparable to those achieved in related work.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.aclweb.org/anthology/Q15-1037/">A Hierarchical Distance-dependent Bayesian Model for Event Coreference Resolution</a> by<i> Bishan Yang, Claire Cardie, Peter Frazier</i></summary><blockquote><p align="justify">
We present a novel hierarchical distance-dependent Bayesian model for event coreference resolution. While existing generative models for event coreference resolution are completely unsupervised, our model allows for the incorporation of pairwise distances between event mentions — information that is widely used in supervised coreference models to guide the generative clustering processing for better event clustering both within and across documents. We model the distances between event mentions using a feature-rich learnable distance function and encode them as Bayesian priors for nonparametric clustering. Experiments on the ECB+ corpus show that our model outperforms state-of-the-art methods for both within- and cross-document event coreference resolution.
</p></blockquote></details>

<details>
<summary>4. <a href="http://aclweb.org/anthology/D15-1020">Cross-document Event Coreference Resolution based on Cross-media Features</a> by<i> Zhang, Tongtao and Li, Hongzhi and Ji, Heng and Chang, Shih-Fu </i></summary><blockquote><p align="justify">
In this paper we focus on a new problem of event coreference resolution across television news videos. Based on the observation that the contents from multiple data modalities are complementary, we develop a novel approach to jointly encode effective features from both closed captions and video key frames. Experiment results demonstrate that visual features provided 7.2\% absolute F-score gain on stateof-the-art text based event extraction and coreference resolution.
</p></blockquote></details>

### 2016

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/C16-1308/">Joint Inference for Event Coreference Resolution</a> by<i> Jing Lu, Deepak Venugopal, Vibhav Gogate, Vincent Ng</i></summary><blockquote><p align="justify">
Event coreference resolution is a challenging problem since it relies on several components of the information extraction pipeline that typically yield noisy outputs. We hypothesize that exploiting the inter-dependencies between these components can significantly improve the performance of an event coreference resolver, and subsequently propose a novel joint inference based event coreference resolver using Markov Logic Networks (MLNs). However, the rich features that are important for this task are typically very hard to explicitly encode as MLN formulas since they significantly increase the size of the MLN, thereby making joint inference and learning infeasible. To address this problem, we propose a novel solution where we implicitly encode rich features into our model by augmenting the MLN distribution with low dimensional unit clauses. Our approach achieves state-of-the-art results on two standard evaluation corpora.
</p></blockquote></details>

<details>
<summary>2. <a href="https://dl.acm.org/doi/10.5555/3016100.3016310">Joint inference over a lightly supervised information extraction pipeline: towards event coreference resolution for resource-scarce languages</a> by<i> Chen Chen, Vincent Ng</i></summary><blockquote><p align="justify">
We address two key challenges in end-to-end event coreference resolution research: (1) the error propagation problem, where an event coreference resolver has to assume as input the noisy outputs produced by its upstream components in the standard information extraction (IE) pipeline; and (2) the data annotation bottleneck, where manually annotating data for all the components in the IE pipeline is prohibitively expensive. This is the case in the vast majority of the world's natural languages, where such annotated resources are not readily available. To address these problems, we propose to perform joint inference over a lightly supervised IE pipeline, where all the models are trained using either active learning or unsupervised learning. Using our approach, only 25% of the training sentences in the Chinese portion of the ACE 2005 corpus need to be annotated with entity and event mentions in order for our event coreference resolver to surpass its fully supervised counterpart in performance.
</p></blockquote></details>

<details>
<summary>3. <a href="https://www.semanticscholar.org/paper/Illinois-CCG-Entity-Discovery-and-Linking%2C-Event-Tsai-Mayhew/427b0cdb647cfbb9982bb9a4def5772618ab26fe">Illinois CCG Entity Discovery and Linking, Event Nugget Detection and Co-reference, and Slot Filler Validation Systems for TAC 2016</a> by<i> Chen-Tse Tsai, Stephen Mayhew, Haoruo Peng, Mark Sammons, Bhargav Mangipudi, Pavankumar Reddy, Dan Roth</i></summary><blockquote><p align="justify">
The University of Illinois CCG team participated in three TAC 2016 tasks: Entity Discovery and Linking (EDL); Event Nugget Detection and Co-reference (ENDC); and Slot Filler Validation (SFV). The EDL system includes Spanish and Chinese named entity recognition, crosslingual wikification, and nominal head detection. The ENDC system identifies event nugget mentions and puts them into co-reference chains. We develop ENDC based on English and it works on Spanish and Chinese through translations. The SFV system uses a set of classifiers, one per target relation, trained with the gold assessed TAC Cold Start Knowledge Base Population responses, filtered using performance on this data.
</p></blockquote></details>

<details>
<summary>4. <a href="https://www.aclweb.org/anthology/L16-1631/">Event Coreference Resolution with Multi-Pass Sieves</a> by<i> Jing Lu, Vincent Ng</i></summary><blockquote><p align="justify">
Multi-pass sieve approaches have been successfully applied to entity coreference resolution and many other tasks in natural language processing (NLP), owing in part to the ease of designing high-precision rules for these tasks. However, the same is not true for event coreference resolution: typically lying towards the end of the standard information extraction pipeline, an event coreference resolver assumes as input the noisy outputs of its upstream components such as the trigger identification component and the entity coreference resolution component. The difficulty in designing high-precision rules makes it challenging to successfully apply a multi-pass sieve approach to event coreference resolution. In this paper, we investigate this challenge, proposing the first multi-pass sieve approach to event coreference resolution. When evaluated on the version of the KBP 2015 corpus available to the participants of EN Task 2 (Event Nugget Detection and Coreference), our approach achieves an Avg F-score of 40.32%, outperforming the best participating system by 0.67% in Avg F-score.
</p></blockquote></details>

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/D16-1038/">Event Detection and Co-reference with Minimal Supervision</a> by<i> Haoruo Peng, Yangqiu Song, Dan Roth</i></summary><blockquote><p align="justify">
An important aspect of natural language understanding involves recognizing and categorizing events and the relations among them. However, these tasks are quite subtle and annotating training data for machine learning based approaches is an expensive task, resulting in supervised systems that attempt to learn complex models from small amounts of data, which they over-fit. This paper addresses this challenge by developing an event detection and co-reference system with minimal supervision, in the form of a few event examples. We view these tasks as semantic similarity problems between event mentions or event mentions and an ontology of types, thus facilitating the use of large amounts of out of domain text data. Notably, our semantic relatedness function exploits the structure of the text by making use of a semantic-role-labeling based representation of an event. We show that our approach to event detection is competitive with the top supervised methods. More significantly, we outperform stateof-the-art supervised methods for event coreference on benchmark data sets, and support significantly better transfer across domains.
</p></blockquote></details>

<details>
<summary>6. <a href="http://aclweb.org/anthology/D16-1038">Event Detection and Co-reference with Minimal Supervision</a> by<i> Peng, Haoruo and Song, Yangqiu and Roth, Dan </i></summary><blockquote><p align="justify">
An important aspect of natural language understanding involves recognizing and categorizing events and the relations among them. However, these tasks are quite subtle and annotating training data for machine learning based approaches is an expensive task, resulting in supervised systems that attempt to learn complex models from small amounts of data, which they over-ﬁt. This paper addresses this challenge by developing an event detection and co-reference system with minimal supervision, in the form of a few event examples. We view these tasks as semantic similarity problems between event mentions or event mentions and an ontology of types, thus facilitating the use of large amounts of out of domain text data. Notably, our semantic relatedness function exploits the structure of the text by making use of a semantic-role-labeling based representation of an event.
</p></blockquote></details>

<details>
<summary>7. <a href="http://aclweb.org/anthology/P16-1025">Liberal Event Extraction and Event Schema Induction</a> by<i> Huang, Lifu and Cassidy, Taylor and Feng, Xiaocheng and Ji, Heng and Voss, Clare R. and Han, Jiawei and Sil, Avirup </i></summary><blockquote><p align="justify">
We propose a brand new “Liberal” Event Extraction paradigm to extract events and discover event schemas from any input corpus simultaneously. We incorporate symbolic (e.g., Abstract Meaning Representation) and distributional semantics to detect and represent event structures and adopt a joint typing framework to simultaneously extract event types and argument roles and discover an event schema. Experiments on general and speciﬁc domains demonstrate that this framework can construct high-quality schemas with many event and argument role types, covering a high proportion of event types and argument roles in manually deﬁned schemas. We show that extraction performance using discovered schemas is comparable to supervised models trained from a large amount of data labeled according to predeﬁned event types. The extraction quality of new event types is also promising.
</p></blockquote></details>

### 2017

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/D17-1226/">Event Coreference Resolution by Iteratively Unfolding Inter-dependencies among Events</a> by<i> Prafulla Kumar Choubey, Ruihong Huang</i></summary><blockquote><p align="justify">
We introduce a novel iterative approach for event coreference resolution that gradually builds event clusters by exploiting inter-dependencies among event mentions within the same chain as well as across event chains. Among event mentions in the same chain, we distinguish within- and cross-document event coreference links by using two distinct pairwise classifiers, trained separately to capture differences in feature distributions of within- and cross-document event clusters. Our event coreference approach alternates between WD and CD clustering and combines arguments from both event clusters after every merge, continuing till no more merge can be made. And then it performs further merging between event chains that are both closely related to a set of other chains of events. Experiments on the ECB+ corpus show that our model outperforms state-of-the-art methods in joint task of WD and CD event coreference resolution.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/P17-1009/">Joint Learning for Event Coreference Resolution</a> by<i> Jing Lu, Vincent Ng</i></summary><blockquote><p align="justify">
While joint models have been developed for many NLP tasks, the vast majority of event coreference resolvers, including the top-performing resolvers competing in the recent TAC KBP 2016 Event Nugget Detection and Coreference task, are pipeline-based, where the propagation of errors from the trigger detection component to the event coreference component is a major performance limiting factor. To address this problem, we propose a model for jointly learning event coreference, trigger detection, and event anaphoricity. Our joint model is novel in its choice of tasks and its features for capturing cross-task interactions. To our knowledge, this is the first attempt to train a mention-ranking model and employ event anaphoricity for event coreference. Our model achieves the best results to date on the KBP 2016 English and Chinese datasets.
</p></blockquote></details>

<details>
<summary>3. <a href="https://ieeexplore.ieee.org/document/8260622">Learning Antecedent Structures for Event Coreference Resolution</a> by<i> Jing Lu, Vincent Ng</i></summary><blockquote><p align="justify">
The vast majority of existing work on learning-based event coreference resolution has employed the so-called mentionpair model, which is a binary classifier that determines whether two event mentions are coreferent. Though conceptually simple, this model is known to suffer from several major weaknesses. Rather than making pairwise local decisions, we view event coreference as a structured prediction task, where we propose a probabilistic model that selects an antecedent for each event mention in a given document in a collective manner. Our model achieves the best results reported to date on the new KBP 2016 English and Chinese event coreference resolution datasets.
</p></blockquote></details>

<details>
<summary>4. <a href="https://www.semanticscholar.org/paper/UTD%E2%80%99s-Event-Nugget-Detection-and-Coreference-System-Lu-Ng/782de88b47641d804ae36919db08f86752d24ed3">UTD’s Event Nugget Detection and Coreference System at KBP 2017</a> by<i> Jing Lu, Vincent Ng</i></summary><blockquote><p align="justify">
We describe UTD’s participating system in the event nugget detection and coreference task at TAC-KBP 2017. We designed and implemented a pipeline system that consists of three components: event nugget identification and subtyping, REALIS value identification, and event coreference resolution. We proposed using an ensemble of 1-nearest-neighbor classifiers for event nugget identification and subtyping, a 1-nearest-neighbor classifier for REALIS value identification, and a learningbased multi-pass sieve approach consisting of 1-nearest-neighbor classifiers for event coreference resolution. Though conceptually simple, our system compares favorably with other participating systems, achieving F1 scores of 50.37, 40.91, and 33.87 on these three tasks respectively on the English dataset, and F1 scores of 46.76, 35.19, and 28.01 on the Chinese dataset. In particular, it ranked first on Chinese event nugget coreference.
</p></blockquote></details>


## Surveys
[:arrow_up:](#table-of-contents)
<details>
<summary>1. <a href="https://www.sciencedirect.com/science/article/pii/S0167923616300173">A Survey of event extraction methods from text for decision support systems</a> by<i> Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong, Emiel Carona</i></summary><blockquote><p align="justify">
Event extraction, a specialized stream of information extraction rooted back into the 1980s, has greatly gained in popularity due to the advent of big data and the developments in the related fields of text mining and natural language processing. However, up to this date, an overview of this particular field remains elusive. Therefore, we give a summarization of event extraction techniques for textual data, distinguishing between data-driven, knowledge-driven, and hybrid methods, and present a qualitative evaluation of these. Moreover, we discuss common decision support applications of event extraction from text corpora. Last, we elaborate on the evaluation of event extraction systems and identify current research issues.
</p></blockquote></details>

<details>
<summary>2. <a href="https://pdfs.semanticscholar.org/0eef/643c744ac3e4ffd68d4328b5f445dbf9e10e.pdf">A Survey of Open Domain Event Extraction </a> by<i> Zecheng Zhang, Yuncheng Wu, Zesheng Wang </i></summary><blockquote><p align="justify">
The development of the Internet and social media have insti-gated the increasing of text information communication andsharing.  Tremendous volume of text data is generated everysecond.  To mine useful and structured knowledge from theunstructured text corpus, many information extraction sys-tems have been created.  One of the crucial methods of thosesystems is the open domain event extraction, which aims atextracting  meaningful  event  information  without  any  pre-defined  domain  assumption.   Combining  with  some  recentpromising  techniques  including  Question  Answer  pairing,entity linking, entity coreference and deep learning, the re-sult of open domain extraction seems to be improved. In thissurvey, we will first briefly give introduction to the pipelineof event extraction.  Then we will give relatively detailed de-scription on recent promising open domain event extractionapproaches including examination of some recent papers.
</p></blockquote></details>

<details>
<summary>3. <a href="http://ceur-ws.org/Vol-779/derive2011_submission_1.pdf">An Overview of Event Extraction from Text </a> by<i> Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, and Franciska de Jong </i></summary><blockquote><p align="justify">
One common application of text mining is event extraction,which encompasses deducing specific knowledge concerning incidents re-ferred to in texts. Event extraction can be applied to various types ofwritten text, e.g., (online) news messages, blogs, and manuscripts. Thisliterature survey reviews text mining techniques that are employed forvarious event extraction purposes. It provides general guidelines on howto choose a particular event extraction technique depending on the user,the available content, and the scenario of use., we provide detailed analysis for the most representative methods, especially their origins, basics, strengths and weaknesses. Last, we also present our envisions about future research directions.
</p></blockquote></details>

<details>
<summary>4. <a href="https://www.semanticscholar.org/paper/A-Survey-of-Textual-Event-Extraction-from-Social-Mejri-Akaichi/2ed13ebbad66c3a8599dd9aa7c20fbdaaee22b11">A Survey of Textual Event Extraction from Social Networks </a> by<i> Mohamed Mejri, Jalel Akaichi</i></summary><blockquote><p align="justify">
In the last decade, mining textual content on social networks to extract relevant data and useful knowledge is becoming an omnipresent task. One common application of text mining is Event Extraction, which is considered as a complex task divided into multiple sub-tasks of varying di culty. In this paper, we present a survey of the main existing text mining techniques which are used for many di erent event extraction aims. First, we present the main data-driven approaches which are based on statistics models to convert data to knowledge. Second, we expose the knowledgedriven approaches which are based on expert knowledge to extract knowledge usually by means of pattern-based approaches. Then we present the main existing hybrid approaches that combines data-driven and data-knowledge approaches. We end this paper with a comparative study that recapitulates the main features of each presented method
</p></blockquote></details>

<details>
<summary>5. <a href="http://rali.iro.umontreal.ca/rali/sites/default/files/publis/Atefeh_et_al-2013-Computational_Intelligence-2.pdf">A Survey of Techniques for Event Detection in Twitter </a> by<i> Farzindar  Atefeh, Wael  Khreich </i></summary><blockquote><p align="justify">
Twitter is among the fastest-growing microblogging and online social networking services. Messages posted on Twitter tweets have been reporting everything from daily life stories to the latest local and global news and events. Monitoring and analyzing this rich and continuous user-generated content can yield unprecedentedly valuable information, enabling users and organizations to acquire actionable knowledge. This article provides a survey of techniques for event detection from Twitter streams. These techniques aim at finding real-world occurrences that unfold over space and time. In contrast to conventional media, event detection from Twitter streams poses new challenges. Twitter streams contain large amounts of meaningless messages and polluted content, which negatively affect the detection performance. In addition, traditional text mining techniques are not suitable, because of the short length of tweets, the large number of spelling and grammatical errors, and the frequent use of informal and mixed language. Event detection techniques presented in literature address these issues by adapting techniques from various fields to the uniqueness of Twitter. This article classifies these techniques according to the event type, detection task, and detection method and discusses commonly used features. Finally, it highlights the need for public benchmarks to evaluate the performance of different detection approaches and various features.
</p></blockquote></details>

<details>
<summary>6. <a href="https://ieeexplore.ieee.org/abstract/document/8729158">Survey on Event Extraction Technology in Information Extraction Research Area </a> by<i> Liying Zhan, Xuping Jiang  </i></summary><blockquote><p align="justify">
Event extraction is one of the most challenging tasks in the field of information extraction. The research of event extraction technology has important theoretical significance and wide application value. This paper makes a survey of event extraction technology, describes the tasks and related concepts of event extraction, analyzes, compares and generalizes the relevant descriptions in different fields. Then analyzes, compares and summarizes the three main methods of event extraction. These methods have their own advantages and disadvantages. The methods based on rules and templates are more mature, the method based on statistical machine learning is dominant, and the method based on deep learning is the future development trend. At the same time, this paper also reviews the research status and key technologies of event extraction, and finally summarizes the current challenges and future research trends of event extraction technology.
</p></blockquote></details>

<details>
<summary>7. <a href="http://www.cfilt.iitb.ac.in/resources/surveys/Temporal_Information_Extraction-Naman-June15.pdf">Temporal Information Extraction Extracting Events and Temporal Expressions A Literature Survey </a> by<i> Naman Gupta </i></summary><blockquote><p align="justify">
Information  on  web  is  increasing  at  infinitum.   There  exists  plethora  of  data  on  World  Wide  Web (WWW) in various electronic and digital form.  Thus, web has become an unstructured global area where information even if available, cannot be directly used for desired applications.  One is often faced with an information overload and demands for some automated help.  Information extraction (IE) is the task of automatically extracting structured information from unstructured and/or semi-structured machine-readable  documents  by  means  of  Text  Mining  and  Natural  Language  Processing  (NLP)  techniques. Extracted structured information can be used for variety of enterprise or personal level task of varying complexity.   In  this  survey  report  we  will  discuss  literature  related  to  temporal  expressions  and  event extraction
</p></blockquote></details>


<details>
<summary>8. <a href="https://ieeexplore.ieee.org/document/8918013">A Survey of Event Extraction From Text </a> by<i> Wei Xiang, Bang Wang </i></summary><blockquote><p align="justify">
Numerous important events happen everyday and everywhere but are reported in different media sources with different narrative styles. How to detect whether real-world events have been reported in articles and posts is one of the main tasks of event extraction. Other tasks include extracting event arguments and identifying their roles, as well as clustering and tracking similar events from different texts. As one of the most important research themes in natural language processing and understanding, event extraction has a wide range of applications in diverse domains and has been intensively researched for decades. This article provides a comprehensive yet up-to-date survey for event extraction from text. We not only summarize the task definitions, data sources and performance evaluations for event extraction, but also provide a taxonomy for its solution approaches. In each solution group, we provide detailed analysis for the most representative methods, especially their origins, basics, strengths and weaknesses. Last, we also present our envisions about future research directions.
</p></blockquote></details>

<details>
<summary>9. <a href="https://www.cs.cmu.edu/~nbach/papers/A-survey-on-Relation-Extraction.pdf">A Review of Relation Extraction</a> by<i> Nguyen Bach, Sameer Badaskar </i></summary><blockquote><p align="justify">
Many applications in information extraction, natural language understanding, information retrieval require an understanding of the semantic relations between entities. We present a comprehensive review of various aspects of the entity relation extraction task. Some of the most important supervised and semi-supervised classiﬁcation approaches to the relation extraction task are covered in sufﬁcient detail along with critical analyses. We also discuss extensions to higher-order relations. Evaluation methodologies for both supervised and semi-supervised methods are described along with pointers to the commonly used performance evaluation datasets. Finally, we also give short descriptions of two important applications of relation extraction, namely question answering and biotext mining.
</p></blockquote></details>

## Others 
[:arrow_up:](#table-of-contents)


### 2009

<details>
<summary>1. <a href="http://portal.acm.org/citation.cfm?doid=1667583.1667697">Predicting unknown time arguments based on cross-event propagation</a> by<i> Gupta, Prashant and Ji, Heng </i></summary><blockquote><p align="justify">
Many events in news articles don’t include time arguments. This paper describes two methods, one based on rules and the other based on statistical learning, to predict the unknown time argument for an event by the propagation from its related events. The results are promising – the rule based approach was able to correctly predict 74\% of the unknown event time arguments with 70\% precision.
</p></blockquote></details>

### 2011

<details>
<summary>1. <a href="http://dl.acm.org/citation.cfm?doid=1978942.1978975">Twitinfo: aggregating and visualizing microblogs for event exploration</a> by<i> Marcus, Adam and Bernstein, Michael S. and Badar, Osama and Karger, David R. and Madden, Samuel and Miller, Robert C. </i></summary><blockquote><p align="justify">
Microblogs are a tremendous repository of user-generated content about world events. However, for people trying to understand events by querying services like Twitter, a chronological log of posts makes it very difﬁcult to get a detailed understanding of an event. In this paper, we present TwitInfo, a system for visualizing and summarizing events on Twitter. TwitInfo allows users to browse a large collection of tweets using a timeline-based display that highlights peaks of high tweet activity. A novel streaming algorithm automatically discovers these peaks and labels them meaningfully using text from the tweets. Users can drill down to subevents, and explore further via geolocation, sentiment, and popular URLs. We contribute a recall-normalized aggregate sentiment visualization to produce more honest sentiment overviews. An evaluation of the system revealed that users were able to reconstruct meaningful summaries of events in a small amount of time. An interview with a Pulitzer Prize-winning journalist suggested that the system would be especially useful for understanding a long-running event and for identifying eyewitnesses. Quantitatively, our system can identify 80-100\% of manually labeled peaks, facilitating a relatively complete view of each event studied.
</p></blockquote></details>

### 2013

<details>
<summary>1. <a href="http://dl.acm.org/citation.cfm?doid=2487575.2487718">EventCube: multi-dimensional search and mining of structured and text data</a> by<i> Tao, Fangbo and Ji, Heng and Kanade, Rucha and Kao, Anne and Li, Qi and Li, Yanen and Lin, Cindy and Liu, Jialu and Oza, Nikunj and Srivastava, Ashok and Tjoelker, Rod and Lei, Kin Hou and Wang, Chi and Zhang, Duo and Zhao, Bo and Han, Jiawei and Zhai, Chengxiang and Cheng, Xiao and Danilevsky, Marina and Desai, Nihit and Ding, Bolin and Ge, Jing Ge </i></summary><blockquote><p align="justify">
A large portion of real world data are either text data or structured (e.g., relational) data. Moreover, such data are often linked together (e.g., structured speciﬁcation of products linking with the corresponding product descriptions and customer comments). Even for text data such as news data, typed entities can be extracted with entity extraction tools. The EventCube project constructs TextCube and TopicCube from interconnected structured and text data (or from text data via entity extraction and dimension building), and performs multidimensional search and analysis on such datasets, in an informative, powerful, and userfriendly manner. This proposed EventCube demo will show the power of the system not only on the originally designed ASRS (Aviation Safety Report System) data sets, but also on news datasets, collected from multiple news agencies. The system has high potential to be extended in many powerful ways and serve as a general platform for search, OLAP (online analytical processing) and data mining on integrated text and structured data. After the system demo in the conference (if accepted), the system will be put on the web for public access and evaluation.
</p></blockquote></details>

### 2014

<details>
<summary>1. <a href="https://www.aclweb.org/anthology/P14-2082/">An Annotation Framework for Dense Event Ordering</a> by<i> Taylor Cassidy, Bill McDowell, Nathanael Chambers, Steven Bethard</i></summary><blockquote><p align="justify">
Today’s event ordering research is heavily dependent on annotated corpora. Current corpora inﬂuence shared evaluations and drive algorithm development. Partly due to this dependence, most research focuses on partial orderings of a document’s events. For instance, the TempEval competitions and the TimeBank only annotate small portions of the event graph, focusing on the most salient events or on speciﬁc types of event pairs (e.g., only events in the same sentence). Deeper temporal reasoners struggle with this sparsity because the entire temporal picture is not represented. This paper proposes a new annotation process with a mechanism to force annotators to label connected graphs. It generates 10 times more relations per document than the TimeBank, and our TimeBank-Dense corpus is larger than all current corpora. We hope this process and its dense corpus encourages research on new global models with deeper reasoning.
</p></blockquote></details>

<details>
<summary>2. <a href="http://aclweb.org/anthology/P14-1094">Cross-narrative Temporal Ordering of Medical Events</a> by<i> Raghavan, Preethi and Fosler-Lussier, Eric and Elhadad, Noémie and Lai, Albert M. </i></summary><blockquote><p align="justify">
Cross-narrative temporal ordering of medical events is essential to the task of generating a comprehensive timeline over a patient’s history. We address the problem of aligning multiple medical event sequences, corresponding to different clinical narratives, comparing the following approaches: (1) A novel weighted ﬁnite state transducer representation of medical event sequences that enables composition and search for decoding, and (2) Dynamic programming with iterative pairwise alignment of multiple sequences using global and local alignment algorithms. The cross-narrative coreference and temporal relation weights used in both these approaches are learned from a corpus of clinical narratives. We present results using both approaches and observe that the ﬁnite state transducer approach performs performs signiﬁcantly better than the dynamic programming one by 6.8\% for the problem of multiple-sequence alignment.
</p></blockquote></details>

<details>
<summary>3. <a href="http://aclweb.org/anthology/P14-1093">Toward Future Scenario Generation: Extracting Event Causality Exploiting Semantic Relation, Context, and Association Features</a> by<i> Hashimoto, Chikara and Torisawa, Kentaro and Kloetzer, Julien and Sano, Motoki and Varga, István and Oh, Jong-Hoon and Kidawara, Yutaka </i></summary><blockquote><p align="justify">
We propose a supervised method of extracting event causalities like conduct slash-and-burn agriculture→exacerbate desertiﬁcation from the web using semantic relation (between nouns), context, and association features. Experiments show that our method outperforms baselines that are based on state-of-the-art methods. We also propose methods of generating future scenarios like conduct slash-and-burn agriculture→exacerbate desertiﬁcation→increase Asian dust (from China)→asthma gets worse. Experiments show that we can generate 50,000 scenarios with 68\% precision. We also generated a scenario deforestation continues→global warming worsens→sea temperatures rise→vibrio parahaemolyticus fouls (water), which is written in no document in our input web corpus crawled in 2007. But the vibrio risk due to global warming was observed in Baker-Austin et al. (2013). Thus, we “predicted” the future event sequence in a sense.
</p></blockquote></details>

### 2015

<details>
<summary>1. <a href="http://aclweb.org/anthology/P15-1019">Generative Event Schema Induction with Entity Disambiguation</a> by<i> Nguyen, Kiem-Hieu and Tannier, Xavier and Ferret, Olivier and Besançon, Romaric </i></summary><blockquote><p align="justify">
This paper presents a generative model to event schema induction. Previous methods in the literature only use head words to represent entities. However, elements other than head words contain useful information. For instance, an armed man is more discriminative than man. Our model takes into account this information and precisely represents it using probabilistic topic distributions. We illustrate that such information plays an important role in parameter estimation. Mostly, it makes topic distributions more coherent and more discriminative. Experimental results on benchmark dataset empirically conﬁrm this enhancement.
</p></blockquote></details>

<details>
<summary>2. <a href="http://aclweb.org/anthology/P15-4023">Storybase: Towards Building a Knowledge Base for News Events</a> by<i> Wu, Zhaohui and Liang, Chen and Giles, C. Lee </i></summary><blockquote><p align="justify">
To better organize and understand online news information, we propose Storybase1, a knowledge base for news events that builds upon Wikipedia current events and daily Web news. It ﬁrst constructs stories and their timelines based on Wikipedia current events and then detects and links daily news to enrich those Wikipedia stories with more comprehensive events. We encode events and develop efﬁcient event clustering and chaining techniques in an event space. We demonstrate Storybase with a news events search engine that helps ﬁnd historical and ongoing news stories and inspect their dynamic timelines.
</p></blockquote></details>

### 2016

<details>
<summary>1. <a href="http://aclweb.org/anthology/W16-1004">A Comparison of Event Representations in DEFT</a> by<i> Ann Bies , Zhiyi Song, Jeremy Getman, Joe Ellis, Justin Mott, Stephanie Strassel,
Martha Palmer, Teruko Mitamura, Marjorie Freedman, Heng Ji, Tim O'Gorman</i></summary><blockquote><p align="justify">
This paper will discuss and compare event representations across a variety of types of event annotation: Rich Entities, Relations, and Events (Rich ERE), Light Entities, Relations, and Events (Light ERE), Event Nugget (EN), Event Argument Extraction (EAE), Richer Event Descriptions (RED), and Event-Event Relations (EER). Comparisons of event representations are presented, along with a comparison of data annotated according to each event representation. An event annotation experiment is also discussed, including annotation for all of these representations on the same set of sample data, with the purpose of being able to compare actual annotation across all of these approaches as directly as possible. We walk through a brief example to illustrate the various annotation approaches, and to show the intersections among the various annotated data sets.
</p></blockquote></details>

<details>
<summary>2. <a href="http://aclweb.org/anthology/W16-1701">Building a Cross-document Event-Event Relation Corpus</a> by<i> Hong, Yu and Zhang, Tongtao and O'Gorman, Tim and Horowit-Hendler, Sharone and Ji, Heng and Palmer, Martha </i></summary><blockquote><p align="justify">
We propose a new task of extracting eventevent relations across documents. We present our efforts at designing an annotation schema and building a corpus for this task. Our schema includes ﬁve main types of relations: Inheritance, Expansion, Contingency, Comparison and Temporality, along with 21 subtypes. We also lay out the main challenges based on detailed inter-annotator disagreement and error analysis. We hope these resources can serve as a benchmark to encourage research on this new problem.
</p></blockquote></details>

<details>
<summary>3. <a href="http://aclweb.org/anthology/N16-3015">Cross-media Event Extraction and Recommendation</a> by<i> Lu, Di and Voss, Clare and Tao, Fangbo and Ren, Xiang and Guan, Rachel and Korolov, Rostyslav and Zhang, Tongtao and Wang, Dongang and Li, Hongzhi and Cassidy, Taylor and Ji, Heng and Chang, Shih-fu and Han, Jiawei and Wallace, William and Hendler, James and Si, Mei and Kaplan, Lance </i></summary><blockquote><p align="justify">
The sheer volume of unstructured multimedia data (e.g., texts, images, videos) posted on the Web during events of general interest is overwhelming and difﬁcult to distill if seeking information relevant to a particular concern. We have developed a comprehensive system that searches, identiﬁes, organizes and summarizes complex events from multiple data modalities. It also recommends events related to the user’s ongoing search based on previously selected attribute values and dimensions of events being viewed. In this paper we brieﬂy present the algorithms of each component and demonstrate the system’s capabilities 1.
</p></blockquote></details>

<details>
<summary>4. <a href="http://dl.acm.org/citation.cfm?doid=2964284.2964287">Event Specific Multimodal Pattern Mining for Knowledge Base Construction</a> by<i> Li, Hongzhi and Ellis, Joseph G. and Ji, Heng and Chang, Shih-Fu </i></summary><blockquote><p align="justify">
Knowledge bases, which consist of a collection of entities, attributes, and the relations between them are widely used and important for many information retrieval tasks. Knowledge base schemas are often constructed manually using experts with speciﬁc domain knowledge for the ﬁeld of interest. Once the knowledge base is generated then many tasks such as automatic content extraction and knowledge base population can be performed, which have so far been robustly studied by the Natural Language Processing community. However, the current approaches ignore visual information that could be used to build or populate these structured ontologies. Preliminary work on visual knowledge base construction only explores limited basic objects and scene relations. In this paper, we propose a novel multimodal pattern mining approach towards constructing a high-level “event” schema semi-automatically, which has the capability to extend text only methods for schema construction. We utilize a large unconstrained corpus of weakly-supervised image-caption pairs related to high-level events such as “attack” and “demonstration” to both discover visual aspects of an event, and name these visual components automatically. We compare our method with several state-of-the-art visual pattern mining approaches and demonstrate that our proposed method can achieve dramatic improvements in terms of the number of concepts discovered (33\% gain), semantic consistence of visual patterns (52\% gain), and correctness of pattern naming (150\% gain).
</p></blockquote></details>

<details>
<summary>5. <a href="http://aclweb.org/anthology/P16-1207">Temporal Anchoring of Events for the TimeBank Corpus</a> by<i> Reimers, Nils and Dehghani, Nazanin and Gurevych, Iryna </i></summary><blockquote><p align="justify">
Today’s extraction of temporal information for events heavily depends on annotated temporal links. These so called TLINKs capture the relation between pairs of event mentions and time expressions. One problem is that the number of possible TLINKs grows quadratic with the number of event mentions, therefore most annotation studies concentrate on links for mentions in the same or in adjacent sentences. However, as our annotation study shows, this restriction results for 58\% of the event mentions in a less precise information when the event took place.
</p></blockquote></details>

### 2017

<details>
<summary>1. <a href="http://aclweb.org/anthology/P17-2101">Determining Whether and When People Participate in the Events They Tweet About</a> by<i> Sanagavarapu, Krishna Chaitanya and Vempala, Alakananda and Blanco, Eduardo </i></summary><blockquote><p align="justify">
This paper describes an approach to determine whether people participate in the events they tweet about. Speciﬁcally, we determine whether people are participants in events with respect to the tweet timestamp. We target all events expressed by verbs in tweets, including past, present and events that may occur in the future. We present new annotations using 1,096 event mentions, and experimental results showing that the task is challenging.
</p></blockquote></details>

<details>
<summary>2. <a href="https://www.aclweb.org/anthology/W17-2712/">The Rich Event Ontology</a> by<i> Susan Brown, Claire Bonial, Leo Obrst, Martha Palmer </i></summary><blockquote><p align="justify">
In this paper we describe a new lexical semantic resource, The Rich Event On-tology, which provides an independent conceptual backbone to unify existing semantic role labeling (SRL) schemas and augment them with event-to-event causal and temporal relations. By unifying the FrameNet, VerbNet, Automatic Content Extraction, and Rich Entities, Relations and Events resources, the ontology serves as a shared hub for the disparate annotation schemas and therefore enables the combination of SRL training data into a larger, more diverse corpus. By adding temporal and causal relational information not found in any of the independent resources, the ontology facilitates reasoning on and across documents, revealing relationships between events that come together in temporal and causal chains to build more complex scenarios. We envision the open resource serving as a valuable tool for both moving from the ontology to text to query for event types and scenarios of interest, and for moving from text to the ontology to access interpretations of events using the combined semantic information housed there.
</p></blockquote></details>

### Linguistics
[:arrow_up:](#table-of-contents)

1. <a href="http://semantics.uchicago.edu/scalarchange/vendler57.pdf">Verbs and times</a> by<i> Zeno Vendler </i>

2. <a href="https://user.phil.hhu.de/~filip/Mourelatos%2078:81.PDF">Events, processes, and states</a> by<i> Alexander P. D. Mourelatos </i>

3. <a href="https://brill.com/view/book/edcoll/9789004373112/BP000004.xml">Aspect and Quantification</a> by<i> Lauri Carlson </i>

4. <a href="https://www.researchgate.net/publication/226895496_The_Algebra_of_Events">The Algebra of Events</a> by<i> Emmon Bach </i>

<details>
<summary>5. <a href="https://www.aclweb.org/anthology/J88-2003.pdf">Temporal Ontology and Temporal Reference</a> by<i> Marc Moens, Mark Steedman </i></summary><blockquote><p align="justify">
A semantics of temporal categories in language and a theory of their use in defining the temporal relations between events both require a more complex structure on the domain underlying the meaning representations than is commonly assumed. This paper proposes an ontology based on such notions as causation and consequence, rather than on purely temporal primitives. A central notion in the ontology is that of an elementary event-complex called a "nucleus." A nucleus can be thought of as an association of a goal event, or "culmination," with a "preparatory process" by which it is accomplished, and a "consequent state," which ensues. Natural-language categories like aspects, futurates, adverbials, and when-clauses are argued to change the temporal/aspectual category of propositions under the control of such a nucleic knowledge representation structure. The same concept of a nucleus plays a central role in a theory of temporal reference, and of the semantics of tense, which we follow McCawley, Partee, and Isard in regarding as an anaphoric category. We claim that any manageable formalism for natural- language temporal descriptions will have to embody such an ontology, as will any usable temporal database for knowledge about events which is to be interrogated using natural language. 
</p></blockquote></details>

<details>
<summary>6. <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-968X.00020">Aspectual Shift as Type Coercion</a> by<i> Stephen G. Pulman </i></summary><blockquote><p align="justify">
The phenomenon of aspectual shift has been discussed by several people over the last ten years using an analogy with type coercion in programming languages. This paper tries to take the analogy literally and to spell out the details for an analysis of some common kinds of aspectual shift in English under the influence of some types of temporal modification. A model‐theoretic basis for this kind of type coercion is supplied, and an illustrative fragment is worked out.
</p></blockquote></details>

<details>
<summary>7. <a href="https://www.press.uchicago.edu/ucp/books/book/distributed/E/bo3645761.html">Events as Grammatical Objects : The Converging Perspectives of Lexical Semantics and Syntax</a> by<i> Carol L. Tenny and James Pustejovsky </i></summary><blockquote><p align="justify">
Researchers in lexical semantics, logical semantics, and syntax have traditionally employed different approaches in their study of natural languages. Yet, recent research in all three fields have demonstrated a growing recognition that the grammars of natural languages structure and refer to events in particular ways. This convergence on the theory of events as grammatical objects is the motivation for this volume, which brings together premiere researchers in these disciplines to specifically address the topic of event structure. The selection of works presented in this volume originated from a 1997 workshop funded by the National Science Foundation regarding Events as Grammatical Objects, from the Combined Perspectives of Lexical Semantics, Logical Semantics and Syntax.
</p></blockquote></details>

8. <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.550.9185&rep=rep1&type=pdf">The Origins of Telicity</a> by<i> Manfred Krifka </i>

9. <a href="https://www.cs.rochester.edu/u/james/Papers/Pustejovsky-event-structure.pdf">The Syntax of Event Structure</a> by<i> James Pustejovsky </i>

10. <a href="http://semantics.uchicago.edu/kennedy/classes/s07/events/krifka90.pdf">Four thousand ships passed through the lock: object-induced measure functions on events</a> by<i> Manfred Krifka  </i>

11. <a href="https://pdfs.semanticscholar.org/aa5a/5634bca33fab72c287b9594d6bbe2b0593ee.pdf">Nominal reference, temporal constitution and quantification in event semantics</a> by<i> Manfred Krifka  </i>

12. <a href="https://www.degruyter.com/view/product/21691?lang=en">Event Structures in Linguistic Form and Interpretation</a> by<i> Dölling Johannes, Heyde-Zybatow Tatjana, Schäfer Martin  </i>

13. <a href="https://www.researchgate.net/publication/257924792_Event_Structures_in_Linguistic_Form_and_Interpretation_Introduction">Event Structures in Linguistic Form and Interpretation (Introduction)</a> by<i> Dölling Johannes, Heyde-Zybatow Tatjana, Schäfer Martin  </i>

14. <a href="https://www.researchgate.net/publication/290107347_Time_in_Language_Language_in_Time">Time in Language, Language in Time</a> by<i> Wolfgang Klein  </i>
 
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

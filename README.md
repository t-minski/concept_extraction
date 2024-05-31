# concept_extraction

# Keyword Extraction using Unsupervised Methods


# Keyword Extraction using Large language Models (LLMs)


# How to download datasets for data_cs directory?
[Reference](https://github.com/LIAAD/KeywordExtractor-Datasets/tree/master)

Download the three datasets and unzip them in data_cs directory

| Dataset                         | Language | Type of Doc     | Domain        | #Docs  | #Gold Keys (per doc) | #Tokens per doc | Absent GoldKey | 
| ------------------------------- | -------- | --------------- | ------------- | -----  | -------------------- | --------------- | -------------- |
| [__Krapivin2009__](#Krapivin)   | EN       | Paper           | Comp. Science | 2304   | 14599 (6.34)         | 8040.74         | 15.3%          |
| [__SemEval2010__](#SemEval2010) | EN       | Paper           | Comp. Science | 243    | 4002 (16.47)         | 8332.34         | 11.3%          |
| [__SemEval2017__](#SemEval2017) | EN       | Paragraph       | Misc.         | 493    | 8969 (18.19)         | 178.22          | 0.0%           |
<!--| [__KWTweet__](#KWTweet)         | EN       | Tweets          | Misc.         | 7736  | 31759 (4.12)         | 19.79           | 7.87%          |-->

---
<a name="Krapivin"></a>
### Krapivin2009

**Dateset**: [Krapivin2009](datasets/Krapivin2009.zip)

**Cite**: [Large dataset for keyphrases extraction](http://eprints.biblio.unitn.it/1671/)

**Description**: The Krapivin2009 is the biggest dataset in terms of documents, with 2,304 full papers from the Computer Science domain, which were published by ACM in the period ranging from 2003 to 2005. The papers were downloaded from CiteSeerX Autonomous Digital Library and each one has its keywords assigned by the authors and verified by the reviewers.


---

<a name="SemEval2010"></a>
### SemEval2010

**Dateset**: [SemEval2010](datasets/SemEval2010.zip)

**Cite**: [Semeval-2010 task 5: Automatic keyphrase extraction from scientific articles](https://dl.acm.org/citation.cfm?id=1859668)

**Description**: SemEval2010 consists of 244 full scientific papers extracted from the ACM Digital Library (one of the most popular datasets which have been previously used for keyword extraction evaluation), each one ranging from 6 to 8 pages and belonging to four different computer science research areas (distributed systems; information search and retrieval; distributed artificial intelligence – multiagent systems; social and behavioral sciences – economics). Each paper has an author-assigned set of keywords (which are part of the original pdf file) and a set of keywords assigned by professional editors, both of which, may or may not appear explicitly in the text.

---
<a name="SemEval2017"></a>
### SemEval2017

**Dateset**: [SemEval2017](datasets/SemEval2017.zip)

**Cite**: [Semeval 2017 task 10: Scienceie-extracting keyphrases and relations from scientific publications](https://arxiv.org/abs/1704.02853)

**Description**: SemEval2017 consists of 500 paragraphs selected from 500 ScienceDirect journal articles, evenly distributed among the domains of Computer Science, Material Sciences and Physics. Each text has a number of keywords selected by one undergraduate student and an expert annotator. The expert's annotation is prioritized whenever there is disagreement between both annotators. The original purpose is extracting keywords and relations from scientific publications.

---
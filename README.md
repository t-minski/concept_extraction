# concept_extraction

# Keyword Extraction using Unsupervised Methods


# Keyword Extraction using Large language Models (LLMs)


# How to download datasets for data_cs directory?
[Reference](https://github.com/LIAAD/KeywordExtractor-Datasets/tree/master)

Download the three datasets and unzip them in data_cs directory

| Dataset                         | Language | Type of Doc     | Domain        | #Docs  | #Gold Keys (per doc) | #Tokens per doc | Absent GoldKey | 
| ------------------------------- | -------- | --------------- | ------------- | -----  | -------------------- | --------------- | -------------- |
| [__Inspec__](#Inspec)           | EN       | Abstract        | Comp. Science | 2000   | 29230 (14.62)        | 128.20          | 37.7%          |
| [__SemEval2010__](#SemEval2010) | EN       | Paper           | Comp. Science | 243    | 4002 (16.47)         | 8332.34         | 11.3%          |
| [__www__](#www)                 | EN       | Paper           | Comp. Science | 1330   | 7711 (5.80)          | 84.08           | 55.0%          |
<!--| [__KWTweet__](#KWTweet)         | EN       | Tweets          | Misc.         | 7736  | 31759 (4.12)         | 19.79           | 7.87%          |-->

---
<a name="Inspec"></a>
### Inspec

**Dateset**: [Inspec](datasets/Inspec.zip)

**Cite**: [Improved automatic keyword extraction given more linguistic knowledge](https://dl.acm.org/citation.cfm?id=1119383)

**Description**: Inspec consists of 2,000 abstracts of scientific journal papers from Computer Science collected between the years 1998 and 2002. Each document has two sets of keywords assigned: the controlled keywords, which are manually controlled assigned keywords that appear in the Inspec thesaurus but may not appear in the document, and the uncontrolled keywords which are freely assigned by the editors, i.e., are not restricted to the thesaurus or to the document. In our repository, we consider a union of both sets as the ground-truth.

---

<a name="SemEval2010"></a>
### SemEval2010

**Dateset**: [SemEval2010](datasets/SemEval2010.zip)

**Cite**: [Semeval-2010 task 5: Automatic keyphrase extraction from scientific articles](https://dl.acm.org/citation.cfm?id=1859668)

**Description**: SemEval2010 consists of 244 full scientific papers extracted from the ACM Digital Library (one of the most popular datasets which have been previously used for keyword extraction evaluation), each one ranging from 6 to 8 pages and belonging to four different computer science research areas (distributed systems; information search and retrieval; distributed artificial intelligence – multiagent systems; social and behavioral sciences – economics). Each paper has an author-assigned set of keywords (which are part of the original pdf file) and a set of keywords assigned by professional editors, both of which, may or may not appear explicitly in the text.

---
<a name="www"></a>
### www

**Dateset**: [www](datasets/www.zip)

**Cite**: [Extracting Keyphrases from Research Papers using Citation Networks](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8662/8618)

**Description**: the WWW collection is based on the abstracts of papers collected from the World Wide Web Conference (WWW) published during the period 2004-2014, with 1330 documents. The gold-keywords of these papers are the author-labeled terms. 

---
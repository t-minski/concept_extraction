# concept_extraction

# data_cs
Reference](https://github.com/LIAAD/KeywordExtractor-Datasets/tree/master)

Download the three datasets and unzip them in data_cs directory

| Dataset                         | Language | Type of Doc     | Domain        | #Docs  | #Gold Keys (per doc) | #Tokens per doc | Absent GoldKey | 
| ------------------------------- | -------- | --------------- | ------------- | -----  | -------------------- | --------------- | -------------- |
| [__Schutz2008__](#Schutz)       | EN       | Paper           | Comp. Science | 1231   | 55013 (44.69)        | 3901.31         | 13.6%          |
| [__SemEval2010__](#SemEval2010) | EN       | Paper           | Comp. Science | 243    | 4002 (16.47)         | 8332.34         | 11.3%          |
| [__wicc__](#wicc)               | ES       | Paper           | Comp. Science | 1640   | 7498 (4.57)          | 1955.56         | 2.7%           |
<!--| [__KWTweet__](#KWTweet)         | EN       | Tweets          | Misc.         | 7736  | 31759 (4.12)         | 19.79           | 7.87%          |-->

<br><br>
---
<a name="Schutz"></a>
### Schutz2008

**Dateset**: [Schutz2008](datasets/Schutz2008.zip)

**Cite**: [Keyphrase Extraction from Single Documents in the Open Domain Exploiting Linguistic and Statistical Methods](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.394.5372&rep=rep1&type=pdf)

**Description**: Schutz2008 dataset is based on full-text papers collected from PubMed Central, which comprises over 26 million citations for biomedical literature from MEDLINE, life science journals, and online books. It consists of 1,231 papers selected from PubMed Central that the documents are distributed across 254 different journals, ranging from Abdominal Imaging to World
Journal of Urology. These keywords assigned by the authors are hidden in the article and used as gold keywords.

---

<a name="SemEval2010"></a>
### SemEval2010

**Dateset**: [SemEval2010](datasets/SemEval2010.zip)

**Cite**: [Semeval-2010 task 5: Automatic keyphrase extraction from scientific articles](https://dl.acm.org/citation.cfm?id=1859668)

**Description**: SemEval2010 consists of 244 full scientific papers extracted from the ACM Digital Library (one of the most popular datasets which have been previously used for keyword extraction evaluation), each one ranging from 6 to 8 pages and belonging to four different computer science research areas (distributed systems; information search and retrieval; distributed artificial intelligence – multiagent systems; social and behavioral sciences – economics). Each paper has an author-assigned set of keywords (which are part of the original pdf file) and a set of keywords assigned by professional editors, both of which, may or may not appear explicitly in the text.

---
<a name="wicc"></a>
### wicc

**Dateset**: [wicc](datasets/wicc.zip)

**Cite**: [Keyword Identification in Spanish Documents using Neural Networks](http://sedici.unlp.edu.ar/handle/10915/50087)

**Description**: The wicc dataset is composed of 1640 scientific articles published between 1999 and 2012 of the Workshop of Researchers in Computer Science [WICC](http://redunci.info.unlp.edu.ar/wicc.html). 

---
#!/bin/bash

#nohup python3 requirements.py > requirements.log 2>&1 &
#wait

nohup python3 Keyword_Extraction_llama2.py > llama2.log 2>&1 &
wait

nohup python3 Keyword_Extraction_mistral.py > mistral.log 2>&1 &
wait

nohup python3 Keyword_Extraction_unsupervised.py > unsupervised.log 2>&1 &
wait

nohup python3 Keyword_Extraction_BERT.py > bert.log 2>&1 &
wait

nohup python3 Keyword_Extraction_llama3.py > llama3.log 2>&1 &
wait

nohup python3 Keyword_Extraction_mixtral.py > mixtral.log 2>&1 &
wait

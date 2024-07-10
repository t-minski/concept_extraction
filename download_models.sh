#!/bin/bash

huggingface-cli download meta-llama/Llama-2-7b-chat-hf --revision f5db02d
huggingface-cli download meta-llama/Llama-2-70b-chat-hf --revision e9149a1
huggingface-cli download meta-llama/Llama-2-13b-chat-hf --revision a2cb7a7

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --revision e1945c4
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --revision 7129260


huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --revision 0417f4b
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 --revision bbae113
huggingface-cli download mistralai/Mixtral-8x22B-v0.1 --revision ec9e31e


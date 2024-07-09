from conexion.models.prompt_utils import get_prepared_prompt_as_text, SystemPrompt, UserPrompt, AssistantPrompt, FewShotPrompt
from collections import namedtuple
from transformers import AutoTokenizer

prompt_non_chat = """
{% for document, keywords in examples %}
{{bos_token}}I have the following document:
{{document}}
Give me all the keywords that are present in this document and separate them with commas.
{{keywords|join(', ')}} {{eos_token}}

{% endfor %}
{{bos_token}}I have the following document:
{{predictionDocument}}

Give me all the keywords that are present in this document and separate them with commas.
"""


easy_non_chat_prompt = """{% for document, keywords in trainingData -%}
{{bos_token}}Doc:{{document}}
{{keywords|join(', ')}}{{eos_token}}

{% endfor -%}

pred: {{predictionDocument}}
"""

ground_truth_non_chat ="""my_bosDoc:docOne
keywordOneA, keywordOneB, keywordOneCmy_eos

my_bosDoc:docTwo
keywordTwoA, keywordTwoB, keywordTwoCmy_eos

pred: my_doc_pred"""

chat_prompt = [
    SystemPrompt("mysys"),
    FewShotPrompt(
        example_prompt = [
            UserPrompt("Doc: {{document}}"),
            AssistantPrompt("{{keywords|join(', ')}}")
        ]
    ),
    UserPrompt("pred: {{predictionDocument}}"),
]

ground_truth_chat ="""<s>[INST] <<SYS>>
mysys
<</SYS>>

Doc: docOne [/INST] keywordOneA, keywordOneB, keywordOneC </s><s>[INST] Doc: docTwo [/INST] keywordTwoA, keywordTwoB, keywordTwoC </s><s>[INST] pred: my_doc_pred [/INST]"""

training_data = [
        ("docOne", ["keywordOneA", "keywordOneB", "keywordOneC"]),
        ("docTwo", ["keywordTwoA", "keywordTwoB", "keywordTwoC"])
    ]
pred_doc = "my_doc_pred"

class mytokenizer:
    def __init__(self):
        self.special_tokens_map = {'bos_token': 'my_bos', 'eos_token': 'my_eos'}

def test_non_chat():
    #MyTokenizer = namedtuple('MyTokenizer', ['bos_token', 'eos_token'])
    #tokenizer = MyTokenizer(bos_token="my_bos", eos_token="my_eos")
    
    rendered_prompt = get_prepared_prompt_as_text(easy_non_chat_prompt, pred_doc , training_data, mytokenizer())
    assert(rendered_prompt == ground_truth_non_chat)


def test_chat():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    rendered_prompt = get_prepared_prompt_as_text(chat_prompt, pred_doc, training_data, tokenizer)
    print("===============================")
    print(rendered_prompt)
    print("===============================")
    assert(rendered_prompt == ground_truth_chat)




from jinja2 import Template
from typing import Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

def get_prepared_prompt_as_chat(prompt : Union[str, List], predictionDocument : str, trainingData : List[Tuple[str, List[str]]]) -> List[Tuple]:
    """This function is just for models which only accept chat prompts. It will return a list of tuples where each tuple contains the role and the content of the prompt.
    Better use get_prepared_prompt_as_text for models which accept text prompts."""
    replacement_dict = {
        'predictionDocument': predictionDocument,
        'trainingData': trainingData,
        'bos_token': '',
        'eos_token': '',
        'unk_token': ''
    }

    # check for predefined prompts
    if isinstance(prompt, str) and prompt in PREDEFINED_PROMPTS:
        prompt = PREDEFINED_PROMPTS[prompt]

    # process the prompt
    chat_history = []
    if isinstance(prompt, str):
        preprocessed_template = Template(prompt).render(replacement_dict)
        chat_history.append({"role":"user", "content": preprocessed_template})
        return chat_history
    elif isinstance(prompt, list):
        chat_history = []
        for p in prompt:
            chat_history.extend(p.get_chat_prompts(replacement_dict))
        return chat_history
    else:
        logger.error("Invalid prompt type: {}".format(type(prompt)))
        return ""


def get_prepared_prompt_as_text(prompt : Union[str, List], predictionDocument : str, trainingData : List[Tuple[str, List[str]]], tokenizer) -> str:
    replacement_dict = {
        'predictionDocument': predictionDocument,
        'trainingData': trainingData,
        **tokenizer.special_tokens_map, # for bos_token and eos_token
    }

    # check for predefined prompts
    if isinstance(prompt, str) and prompt in PREDEFINED_PROMPTS:
        prompt = PREDEFINED_PROMPTS[prompt]

    # process the prompt
    if isinstance(prompt, str):
        return Template(prompt).render(replacement_dict)
    elif isinstance(prompt, list):
        chat_history = []
        for p in prompt:
            chat_history.extend(p.get_chat_prompts(replacement_dict))
        return tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    else:
        logger.error("Invalid prompt type: {}".format(type(prompt)))
        return ""


class ChatPrompt:
    def __init__(self):
        pass

    def get_chat_prompts(self, replaceDict) -> List[Tuple[str, str]]:
        pass

class UserPrompt(ChatPrompt):
    def __init__(self, prompt : str):
        self.prompt = prompt

    def get_chat_prompts(self, replaceDict) -> List[Tuple[str, str]]:
        renderedPrompt = Template(self.prompt).render(replaceDict)
        return [{"role":"user", "content": renderedPrompt}]
    
class SystemPrompt(ChatPrompt):
    def __init__(self, prompt : str):
        self.prompt = prompt

    def get_chat_prompts(self, replaceDict) -> List[Tuple[str, str]]:
        renderedPrompt = Template(self.prompt).render(replaceDict)
        return [{"role":"system", "content": renderedPrompt}]
    
class AssistantPrompt(ChatPrompt):
    def __init__(self, prompt : str):
        self.prompt = prompt

    def get_chat_prompts(self, replaceDict) -> List[Tuple[str, str]]:
        renderedPrompt = Template(self.prompt).render(replaceDict)
        return [{"role":"assistant", "content": renderedPrompt}]


class FewShotPrompt(ChatPrompt):
    def __init__(self, example_prompt : List[ChatPrompt]):
        self.example_prompt = example_prompt

    def get_chat_prompts(self, replaceDict) -> List[Tuple[str, str]]:
        trainingData = replaceDict['trainingData']

        return_list = []
        for doc, keywords in trainingData:
            new_replace_dict = replaceDict.copy()
            new_replace_dict['document'] = doc
            new_replace_dict['keywords'] = keywords
            for example in self.example_prompt:
                return_list.extend(example.get_chat_prompts(new_replace_dict))        
        return return_list


def get_default_user_prompt(documentType : str, meta_keyword : str) -> str:
    return """I have the following document:
{{"""+ documentType + """}}

Please give me the """ + meta_keyword + """ that are present in this document and separate them with commas:
"""

def get_domain_user_prompt(documentType : str, meta_keyword : str) -> str:
    return """I have the following document:
{{"""+ documentType + """}}

Please give me the """ + meta_keyword + """ related to the domains of Computer Science, Control, and Information Technology that are present in this document and separate them with commas:
"""

PREDEFINED_PROMPTS = {}

for key in ["keywords", "keyphrases", "concepts", "classes", "entities", "topics"]:
    PREDEFINED_PROMPTS["simple_" + key] = [
        UserPrompt(get_default_user_prompt("predictionDocument", key))
    ]

for key in ["keywords", "keyphrases", "concepts", "classes", "entities", "topics"]:
    PREDEFINED_PROMPTS["simple_continuation_" + key] = """Given the following document: {{predictionDocument}}
The """ + key + """ in this document are: """
    

for key in ["keywords", "keyphrases", "concepts", "classes", "entities", "topics"]:
    PREDEFINED_PROMPTS["fs_" + key] = [
        FewShotPrompt(
            example_prompt = [
                UserPrompt(get_default_user_prompt("document", key)),
                AssistantPrompt("{{keywords|join(',')}}")
            ]
        ),
        UserPrompt(get_default_user_prompt("predictionDocument", key))
    ]

# ZS + Domain 
for key in ["keywords", "keyphrases", "concepts", "classes", "entities", "topics"]:
    PREDEFINED_PROMPTS["zs_domain_" + key] = [
        UserPrompt(get_domain_user_prompt("predictionDocument", key))
    ]
    
# ZS + Extracting Context
for key in ["keywords", "keyphrases", "concepts", "classes", "entities", "topics"]:
    PREDEFINED_PROMPTS["zs_extracting_context_" + key] = [
    SystemPrompt("You are a helpful, respectful and honest assistant for extracting " + key + " from the provided document."),
    UserPrompt(get_default_user_prompt("predictionDocument", key))
]

# ZS + Expert Context
for key in ["keywords", "keyphrases", "concepts", "classes", "entities", "topics"]:
    PREDEFINED_PROMPTS["zs_expert_context_" + key] = [
        SystemPrompt("You are an ontology expert in extracting keywords from the document."),
        UserPrompt(get_default_user_prompt("predictionDocument", key))
    ]

# ZS + Task Context
PREDEFINED_PROMPTS["zs_task_context"] = [
    SystemPrompt("You are an expert in extracting keyphrases from documents. Keyphrases are important multi- or single noun phrases that cover main topics of the document."),
    UserPrompt(get_default_user_prompt("predictionDocument", "keyphrases"))
]

#z = [
#    SystemPrompt("Hello, how can I help you?"),
#    FewShotPrompt(
#        example_prompt = [
#            UserPrompt("Given the following document: {{document}} What are the keywords:"),
#            AssistantPrompt("{{keywords|join(',')}}")
#        ]
#    ),
#    UserPrompt("Given the following document: {{predictionDocument}} What are the keywords:"),
#]
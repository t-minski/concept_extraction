templates = {
    "template_1": {
        "system_prompt": """
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for extracting concepts related to computer science from provided document.
        <</SYS>>
        """,
        "main_prompt": """
        [INST]
        I have the following document:
        [DOCUMENT]

        Give me all the concepts that are present in this document and related to computer science and separate them with commas.
        Make sure you only return the concepts and say nothing else. For example, don't say:
        "Sure, I'd be happy to help! Based on the information provided in the document".
        [/INST]
        """
    },
    "template_2": {
        "system_prompt": """
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for extracting concepts related to computer science from provided document.
        <</SYS>>
        """,
        "example_prompt ": """
        I have the following document:
        - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

        Give me all the concepts that are present in this document and related to computer science and separate them with commas.
        Make sure you only return the concepts and say nothing else. For example, don't say:
        "Sure, I'd be happy to help! Based on the information provided in the document".
        [/INST] machine learning algorithms,data analysis,complex computational techniques,neural networks, decision trees,ensemble methods,pattern recognition,predictive analytics,
        """,
        "main_prompt": """
        [INST]
        Here is a document:
        [DOCUMENT]

        Give me all the concepts that are present in this document and related to computer science and separate them with commas.
        Make sure you only return the concepts and say nothing else. For example, don't say:
        "Sure, I'd be happy to help! Based on the information provided in the document".
        [/INST]
        """
    }
}

# LLM as an ontology expert
templates["template_3"] = {
    "system_prompt": """
    <s>[INST] <<SYS>>
    You are an ontology expert in extracting concepts from document in computer science.
    <</SYS>>
    """,
    "example_prompt ": """
    I have the following document:
    - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

    Extract the concepts that can be represented as classes in ontology and are present in this document, separate them with commas, and provide no additional information.
    [/INST] machine learning algorithms,data analysis,complex computational techniques,neural networks, decision trees,ensemble methods,pattern recognition,predictive analytics,
    """,
    "main_prompt": """
    [INST]
    I have the following document:
    [DOCUMENT]

    Extract the concepts that can be represented as classes in ontology and are present in this document, separate them with commas, and provide no additional information.
    Make sure you only return the concepts and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST]
    """
}

# LLM as an ontology expert using training example
templates["template_4"] = {
    "system_prompt": """
    <s>[INST] <<SYS>>
    You are an ontology expert in extracting concepts from document in computer science.
    <</SYS>>
    """,
    
    "example_prompt ": """
    I have the following document:
    - A synergic analysis for Web-based enterprise resources planning systems. As the central nervous system for managing an organization's mission and critical business data, Enterprise Resource Planning (ERP) system has evolved to become the backbone of e-business implementation. Since an ERP system is multimodule application software that helps a company manage its important business functions, it should be versatile enough to automate every aspect of business processes, including e-business.
    
    Extract the concepts that can be represented as classes in ontology and are present in this document, separate them with commas, and provide no additional information.
    Make sure you only return the concepts and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST] synergic analysis, Web-based enterprise resources planning, Enterprise Resource Planning, ERP, e-business, customer relationship management
    """,
    
    "main_prompt": """
    [INST]
    I have the following document:
    [DOCUMENT]

    Extract the concepts that can be represented as classes in ontology and are present in this document, separate them with commas, and provide no additional information.
    Make sure you only return the concepts and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST]
    """
}

# prompt for keyphrase instead of concept
templates["template_5"] = {
    "system_prompt": """
    <s>[INST] <<SYS>>
    You are an expert in extracting keyphrases from documents in computer science.
    <</SYS>>
    """,
    "example_prompt ": """
    I have the following document:
    - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

    Extract the keyphrases related to computer science and separate them with commas. Ensure each keyphrase represents a main topic from the document and provide no additional information.
    Make sure you only return the keyphrases and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST] machine learning algorithms,data analysis,complex computational techniques,neural networks, decision trees,ensemble methods,pattern recognition,predictive analytics,
    """,
    "main_prompt": """
    [INST]
    I have the following document:
    [DOCUMENT]

    Extract the keyphrases related to computer science and separate them with commas. Ensure each keyphrase represents a main topic from the document and provide no additional information.
    Make sure you only return the keyphrases and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST]
    """
}

# Describing Keywords and Keyphrases
templates["template_6"] = {
    "system_prompt": """
    <s>[INST] <<SYS>>
    You are an expert in extracting concepts, keywords, and keyphrases from documents in computer science. A concept represents a set or class of oun phrases entities or 'things' within a domain. Keyword are concepts that are more important. Keyphrase are important multi or single words that cover main topics.
    <</SYS>>
    """,
    "example_prompt ": """
    I have the following document:
    - The development of machine learning algorithms for data analysis involves complex computational techniques and models such as neural networks, decision trees, and ensemble methods, which are used to enhance pattern recognition and predictive analytics.

    Extract the key concepts, keywords, and keyphrases related to computer science. Separate each with commas and provide no additional information.
    Make sure you only return them and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST] machine learning algorithms,data analysis,complex computational techniques,neural networks, decision trees,ensemble methods,pattern recognition,predictive analytics,
    """,
    "main_prompt": """
    [INST]
    I have the following document:
    [DOCUMENT]

    Extract the key concepts, keywords, and keyphrases related to computer science. Separate each with commas and provide no additional information.
    Make sure you only return the concepts and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST]
    """
}

templates["template_7"] = {
    "system_prompt": """
    <s>[INST] <<SYS>>
    You are a knowledgeable assistant skilled in refining extracted concepts from documents in computer science.
    <</SYS>>
    """,
    "main_prompt": """
    [INST]
    I have the following document:
    [DOCUMENT]
    
    Here are the extracted concepts from this document:
    """,
    "main_prompt_": """
    Select maximum 10 concepts from these extracted keywords. Separate them with commas and provide no additional information.
    Make sure you only return the concepts and say nothing else. For example, don't say:
    "Sure, I'd be happy to help! Based on the information provided in the document".
    [/INST]
    """
}
from conexion.models.llm_confidence_models import LLMClosestTraining
from conexion.models.spacy_models import SpacyEntities


corpus = [
        ("A man is eating food.", ['one', 'two', 'three']),
        ("A man is eating a piece of bread.", ['four', 'five', 'six']),
        ("The girl is carrying a baby.", ['seven', 'eight', 'nine']),
        ("A man is riding a horse.", ['ten', 'eleven', 'twelve']),
        ("A woman is playing violin.", ['thirteen', 'fourteen', 'fifteen']),
        ("Two men pushed carts through the woods.", ['sixteen', 'seventeen', 'eighteen']),
        ("A man is riding a white horse on an enclosed ground.", ['nineteen', 'twenty', 'twenty-one']),
        ("A monkey is playing drums.", ['twenty-two', 'twenty-three', 'twenty-four']),
        ("A cheetah is running behind its prey.", ['twenty-five', 'twenty-six', 'twenty-seven'])
    ]

queries = [
        "A man is eating pasta.",
        "Someone in a gorilla costume is playing a set of drums.",
        "A cheetah chases prey on across a field.",
    ]

ground_truth = [
        [
            ("A man is eating food.", ['one', 'two', 'three']),
            ("A man is eating a piece of bread.", ['four', 'five', 'six']),
        ],
        [
            ("A monkey is playing drums.", ['twenty-two', 'twenty-three', 'twenty-four']),
            ("A cheetah is running behind its prey.", ['twenty-five', 'twenty-six', 'twenty-seven'])
        ],
        [
            ("A cheetah is running behind its prey.", ['twenty-five', 'twenty-six', 'twenty-seven']),
            ("A man is riding a white horse on an enclosed ground.", ['nineteen', 'twenty', 'twenty-one']),
        ]
    ]

def test_closest_training_data():
    model = LLMClosestTraining("simple_keyword", "meta-llama/Llama-2-7b-chat-hf", number_of_examples=2)
    model.fit([c[0] for c in corpus], [c[1] for c in corpus])
    result = model.compute_all_training_data(queries)
    assert ground_truth == result

def test_closest_training_data_one_by_one():
    model = LLMClosestTraining("simple_keyword", "meta-llama/Llama-2-7b-chat-hf", number_of_examples=2)
    model.fit([c[0] for c in corpus], [c[1] for c in corpus])

    for query, quer_ground_truth in zip(queries, ground_truth):
        result = model.compute_one_training_data(query)
        assert result == quer_ground_truth
    
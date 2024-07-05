def preprocess_data(examples, tokenizer):
    inputs = []
    targets = []

    try:
        for query, answers in zip(examples['query'], examples['answers']):
            if len(answers) == 0:
                inputs.append(query)
                targets.append('')
            else:
                for answer in answers:
                    inputs.append(query)
                    targets.append(answer)
    except:
        for query, answers in zip(examples['question'], examples['answers']):
            if len(answers) == 0:
                inputs.append(query)
                targets.append('')
            else:
                for answer in answers:
                    inputs.append(query)
                    targets.append(answer)

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', text_target=targets)

    return model_inputs

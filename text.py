from transformers import AutoTokenizer, BertForQuestionAnswering

import torch

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")

with torch.no_grad():

    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()

answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

result = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
'a nice puppet'

# target is "nice puppet"

print(result)
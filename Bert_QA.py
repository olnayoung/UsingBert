import torch
from transformers import BertTokenizer, BertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_1 = "Who was Jim Henson?"
text_2 = "Jim Henson was a nice puppet"

input_ids = tokenizer.encode(text_1, text_2)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

assert answer == "a nice puppet"
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_1 = "Who was Jim Henson?"
text_2 = "Jim Henson was a nice puppet"

input_ids = tokenizer.encode(text_1, text_2)    # [CLS], text_1, [SEP], text_2, [SEP]
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))] # 102 = [SEP]

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]) 
print(answer)
assert answer == "a nice puppet"


##### Masked LM #####

masked_index = 8
input_ids[masked_index] = tokenizer.convert_tokens_to_ids('[MASK]')

text_1 = "Who was Jim Henson?"
text_2 = "Jim Henson was a nice puppet"

text1_tokens = ["[CLS]"] + tokenizer.tokenize(text_1) + ["[SEP]"]
# '[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]'
text2_tokens = tokenizer.tokenize(text_2) + ["[SEP]"]
# 'jim', 'henson', 'was', 'a', ‘nice’, 'puppet', '[SEP]'

indexed_ids = tokenizer.convert_tokens_to_ids(text1_tokens + text2_tokens)
# [101, 2040, 2001, 3958, 27227, 1029, 102, 3958, 27227, 2001, 1037, 3835, 13997, 102]
segments_ids = [0]*len(text1_tokens) + [1]*len(text2_tokens)
# [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

tokens_tensor = torch.tensor([input_ids])
segments_tensors = torch.tensor([segments_ids])

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]


predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
assert predicted_token == 'henson'
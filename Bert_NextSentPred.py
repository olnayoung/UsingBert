import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_1 = "what does a technical SEO do?"
text_2 = "A technical seo optimizes websites blah."
label = 0

text1_tokens = ["[CLS]"] + tokenizer.tokenize(text_1) + ["[SEP]"]
text2_tokens = tokenizer.tokenize(text_2) + ["[SEP]"]

indexed_ids = tokenizer.convert_tokens_to_ids(text1_tokens + text2_tokens)
segments_ids = [0]*len(text1_tokens) + [1]*len(text2_tokens)

tokens_tensor = torch.tensor([indexed_ids])
segments_tensors = torch.tensor([segments_ids])

model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()

with torch.no_grad():
    prediction = model(tokens_tensor, segments_tensors)

softmax = torch.nn.Softmax(dim=1)
prediction_sm = softmax(prediction[0])
print(prediction_sm)



##### TRAIN #####
# bert_optimizer = BertAdam(model.parameters(), 
#                                lr = 0.002, 
#                                warmup = 0.1, 
#                                max_grad_norm=-1, 
#                                weight_decay=-0.0001,
#                                t_total = 1
#                               )

# model.train()
# loss = model(tokens_tensor, segments_tensors, next_sentence_label=torch.tensor([label]))
# print("Loss with label {}:".format(label),loss.item())
# loss.backward()
# bert_optimizer.step()
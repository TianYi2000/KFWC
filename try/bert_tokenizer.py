from transformers import BertTokenizer
from torch import from_numpy
import numpy as np


tokenizer = BertTokenizer.from_pretrained("/home/hejiawen/pytorch/wet_AMD_signs_multilabel_classification/net/bert/bert-base-chinese/")
print("词典大小:", tokenizer.vocab_size)
texts = ["左眼突然视物不清1天", "双眼突然视物成双13天", "右眼突然视物不清33天", "左眼硅油填充术后2年半要求取出硅油",
         "晨起突然发现左眼视物不见1天", "双眼逐渐视物模糊4年，加重4个月"]

captions = list()
for text in texts:
    tokens = tokenizer.tokenize(text)
    # print("分词：", tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("id-token转换:", input_ids)
    # input_ids = from_numpy(np.array(input_ids))
    captions.append(input_ids)

caption_len = 0
for caption in captions:
    if len(caption) > caption_len:
        caption_len = len(caption)

for caption in captions:
    if len(caption) < caption_len:
        caption.extend([0] * (caption_len - len(caption)))

print(from_numpy(np.array(captions)))

from transformers import BertModel,BertTokenizer
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
vocab=[token for token in tokenizer.vocab]
model=BertModel.from_pretrained('./pre_model/english')
embedding=model.embeddings.word_embeddings.weight.data
embedding=embedding.numpy()
word_list=[]
with open('dict.txt','r',encoding='utf-8') as f:
    for each in f.readlines():
        word_list.append(each.strip())
with open("{}.{}.{}d.vec".format("bert",len(vocab),embedding.shape[-1]),'w',encoding='utf-8') as fout:
    fout.write("{} {}\n".format(len(vocab),embedding.shape[-1]))
    assert len(vocab)==len(embedding)
    for token,e in zip(vocab,embedding):
        e=[str(i) for i in e ]
        if token in word_list:
            fout.write("{} {}\n".format(token,' '.join(e)))
    empty=['0']*embedding.shape[-1]
    for each in word_list:
        if  each not in vocab:
            fout.write("{} {}\n".format(each,' '.join(empty)))#加入后来做的一些label和属性
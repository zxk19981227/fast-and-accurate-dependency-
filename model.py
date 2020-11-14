import torch
from transformers import BertModel
import numpy as np
class Train_model(torch.nn.Module):
    def __init__(self,bert_dimension,target):
        super().__init__()
        embedding=[]
        word_dict={}
        with open("bert.30522.768d.vec",'r') as f:
            lines=f.readlines()
            vocab_num=int(lines[0].strip().split(' ')[0])
            embedding_size=int(lines[0].strip().split(' ')[1])
            for line in lines[1:]:
                info=line.strip().split(' ')
                current_line=[float(i) for i in info[1:]]
                word =info[0]
                embedding.append(current_line)
                word_dict[word]=len(word_dict.keys())
        self.embedding=torch.nn.Embedding(vocab_num,embedding_size)
        self.embedding.from_pretrained(torch.from_numpy(np.array(embedding)))
        self.linear=torch.nn.Linear(bert_dimension*48,bert_dimension)
        self.linear2=torch.nn.Linear(bert_dimension,200)
        self.linear3=torch.nn.Linear(200,target)
        self.softmax=torch.nn.Softmax()
    def cube(self,tensor:torch.tensor):
        return tensor.mul(tensor.mul(tensor))
    def forward(self, source):
        source=self.embedding(source)
        source=source.view(source.size()[0],-1)
        res=self.linear(source)
        res=self.cube(res)
        res=self.linear2(res)
        res=torch.relu(res)
        res=self.linear3(res)
        return res


import torch
class Data_loader(torch.utils.data.DataLoader):
    def __init__(self,file_path,dict_path,label_dict=None):
        self.dict={}
        self.label=[]
        self.sentence=[]
        if label_dict==None:
            self.label_dict={}
        else:
            self.label_dict=label_dict=label_dict
        with open(dict_path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                if line.strip() not in self.dict.keys():
                    self.dict[line.strip()]=len(self.dict.keys())
        with open(file_path,'r',encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                line=line.strip().split(' ')
                label=line[0]
                if label_dict==None:
                    if label not in self.label_dict.keys():
                        self.label_dict[label]=len(self.label_dict.keys())
                self.label.append(self.label_dict[label])
                feature=[self.dict[i] for i in line[1:]]
                self.sentence.append(feature)
        self.sentence=torch.tensor(self.sentence).long()
        self.label=torch.tensor(self.label)
    def __getitem__(self, item):
        return self.sentence[item],self.label[item]
    def __len__(self):
        return self.sentence.size()[0]


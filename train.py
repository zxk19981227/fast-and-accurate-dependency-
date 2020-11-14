import torch
from model import Train_model
from new_loader import Data_loader
import fitlog
from numpy import mean
import logging
epoch=10000
fitlog.set_log_dir("logs/")
train_data=Data_loader("./data/test/train.txt",'./dict.txt')
test_data=Data_loader("./data/test/train.txt",'./dict.txt',train_data.label_dict)#由于label动态生成因此需要加入以保证顺序相同
train_loader=torch.utils.data.DataLoader(train_data,batch_size=512)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=512)
model=Train_model(768,len(train_data.label_dict))
model=model.cuda()
loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)#,weight_decay=1e-8)
for num in range(epoch):
    total=0
    correct=0
    losses=[]
    for i,(data,label) in enumerate(train_loader):
        data=data.cuda()
        label=label.cuda()
        predict=model(data)
        loss=loss_function(predict,label)
        predict_label=torch.argmax(predict,1)
        correct+=(predict_label==label).cpu().sum().item()
        total+=label.size()[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    # fitlog.add_metric(list(mean(losses)),step=num,name='train_loss')
    fitlog.add_loss(mean(losses),step=num,name='train_loss')
    fitlog.add_metric({"train":{"acc":correct/total}},step=num)
    print("current_epoch_loss:{}".format(str(mean(losses))))
    print("current_epoch_accuracy:{}".format(correct/total))
    total=0
    correct=0
    losses=[]
    for i,(data,label) in enumerate(test_loader):
        data=data.cuda()
        label=label.cuda()
        predict=model(data)
        loss=loss_function(predict,label)
        predict_label=torch.argmax(predict,1)
        correct+=(predict_label==label).cpu().sum().item()
        total+=label.size()[0]
        losses.append(loss.item())
        print("predict",predict_label)
        print("\n")
        print("label",label)
        print(train_data.label_dict)
    print("test_epoch_loss:{}".format(str(mean(losses))))
    print("test_epoch_accuracy:{}".format(correct/total))
    fitlog.add_loss(mean(losses),step=num,name='test_loss')
    fitlog.add_metric({"test":{"acc":correct/total}},step=num,name='train_acc')
fitlog.finish()





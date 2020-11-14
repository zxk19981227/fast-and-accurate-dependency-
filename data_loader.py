import torch
import math
from transformers import BertTokenizer
from tqdm import tqdm
unknown_word="<NULL>"
unknown_lc="<NONE>"
unknown_rc="<NONE>"
unknown_lc2="<NONE>"
unknown_rc2="<NONE>"
unknown_lc2_label="<NONE>"
unknown_lc_label="<NONE>"
unknown_rc_label="<NONE>"
unknown_rc2_label="<NONE>"
unknown_POS="<NONE>"
class Data_reader(torch.utils.data.DataLoader):
    def __init__(self,path,label_dict=None,tokenizer=None):
        self.features=[]
        self.label=[]
        def get_feature(sentence,stack,buffer,Pos,lc1,rc1,lc2,rc2):
            word_features=[]
            word_num=[]#原本我没想到这里，还需要用一个来记录位置用来访问特征
            for word in stack[-3:]:#因为此表中都是句子中相对位置，这里转换成单词
                word_features.append(sentence[word])
                word_features.append(Pos[word])#z这个是词性特征，我盘算了一下如果最后再加进去感觉代价会很高，毕竟还要加上一次循环。
                word_num.append(word)
            while len(word_features)<6:
                word_features.append(unknown_word)
                word_features.append(unknown_POS)
                word_num.append(-1)
            word_num=word_num[:2]#因为lc，rc都是两个栈区前两个
            for word in buffer[:3]:
                word_features.append(sentence[word])
                word_features.append(Pos[word])
            while len(word_features)<12:
                word_features.append(unknown_word)
                word_features.append(unknown_POS)
            #到这里一共出现了12个特征
            # word_num=word_num[:4]#前面解释了，之前写错了orz，应该只有栈区前两个，记错了
            #有些时候buffer和stack长度不足3需要补齐，但是我还没有想好这里word的使用……因为token没有想明白？等完成了bert中添加新的词汇就可以了
            #原本我希望的是返回直接就是id,但是考虑之后发现似乎并不是这样，因为我没有办法保证每一次的dependency以及pos都在列表中，因此这里直接返回标记，然后集体添加到字典中。
            #在最后进行一次encode，并且剔除掉token
            near_word_feature=[]#我妄图在这里按照某一种顺序拼接特征，但是我反应过来，实质上全连接层并不存在顺序问题，甚至可以理解为同一平面上多个数据
            #构成一个graph然后往下映射。
            #current feature num:12,top six words and their pos tag
            for each in word_num:
                if each ==-1:#首先需要判断一下有没有足够的单词
                    #这里还需要加上一个之前我没注意的左孩子的左孩子，右孩子的右孩子
                    near_word_feature.append(unknown_lc)
                    near_word_feature.append(unknown_POS)
                    near_word_feature.append(unknown_lc_label)
                    near_word_feature.append(unknown_lc)
                    near_word_feature.append(unknown_POS)
                    near_word_feature.append(unknown_lc_label)#-1d代表没有单词，所以只能用来unknow
                    near_word_feature.append(unknown_rc)
                    near_word_feature.append(unknown_POS)
                    near_word_feature.append(unknown_rc_label)
                    near_word_feature.append(unknown_rc)
                    near_word_feature.append(unknown_POS)
                    near_word_feature.append(unknown_rc_label)#这里就是右孩子的右孩子的依存关系
                else:
                    if lc1[each]!=-1:
                        near_word_feature.append(sentence(lc1[each]))
                        near_word_feature.append(Pos[each])
                        near_word_feature.append(lc1_label[each])
                        if lc1[lc1[each]]!=-1:
                            near_word_feature.append(sentence[lc1[lc1[each]]])
                            near_word_feature.append(Pos[lc1[each]])
                            near_word_feature.append(lc1_label[lc1[each]])
                        else:
                            near_word_feature.append(unknown_lc)
                            near_word_feature.append(unknown_POS)
                            near_word_feature.append(unknown_lc_label)
                    else:
                        near_word_feature.append(unknown_lc)
                        near_word_feature.append(unknown_POS)
                        near_word_feature.append(unknown_lc_label)
                        near_word_feature.append(unknown_lc)
                        near_word_feature.append(unknown_POS)
                        near_word_feature.append(unknown_lc_label)
                    if rc1[each]!=-1:
                        near_word_feature.append(sentence[rc1[each]])
                        near_word_feature.append(Pos[each])
                        near_word_feature.append(rc1_label[each])
                        if rc1[rc1[each]]!=-1:
                            near_word_feature.append(sentence[rc1[rc1[each]]])
                            near_word_feature.append(Pos[rc1[each]])
                            near_word_feature.append(rc1_label[rc1[each]])
                        else:
                            near_word_feature.append(unknown_rc)
                            near_word_feature.append(unknown_POS)
                            near_word_feature.append(unknown_rc_label)
                    else:
                        near_word_feature.append(unknown_rc)
                        near_word_feature.append(unknown_POS)
                        near_word_feature.append(unknown_rc_label)
                        near_word_feature.append(unknown_rc)
                        near_word_feature.append(unknown_POS)
                        near_word_feature.append(unknown_rc_label)
            #上面一共出现了4*6=24，加上单词出现了36中特征
            for each in word_num:#但是可能是出于我个人想法，我依旧选择将循环写两遍，第二遍处理第远的单词
                #这种写法非常必要，因为这里都没有参与上面的循环
                if each ==-1:#首先需要判断一下有没有足够的单词,如果不写也可以正常跑，但是默认为最后一个
                    near_word_feature.append(unknown_lc2)
                    near_word_feature.append(unknown_POS)
                    near_word_feature.append(unknown_lc2_label)
                    near_word_feature.append(unknown_rc2)
                    near_word_feature.append(unknown_POS)
                    near_word_feature.append(unknown_rc2_label)
                else:
                    if lc2[each]!=-1:#这里也就是有第二远的点
                        near_word_feature.append(sentence[lc2[each]])
                        near_word_feature.append(Pos[each])
                        near_word_feature.append(lc2_label[each])
                    else:
                        near_word_feature.append(unknown_lc2)
                        near_word_feature.append(Pos[each])
                        near_word_feature.append(unknown_lc2_label)
                    if rc2[each]!=-1:
                        near_word_feature.append(sentence[rc2[each]])
                        near_word_feature.append(Pos[each])
                        near_word_feature.append(rc2_label[each])
                    else:
                        near_word_feature.append(unknown_rc2)
                        near_word_feature.append(unknown_POS)
                        near_word_feature.append(unknown_rc2_label)
            for each in near_word_feature:
                word_features.append(each)
            # print(len(word_features))
            assert(len(word_features)==48)#检验读取没有出错
            return word_features
        with open(path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            # buffer=[]
            arc_label=['<NONE>']#因为dependency是从1开始计数，所以最好所有的都加一个初始化数值方便管理
            sentence=['<ROOT>']
            dependency=['<NONE>']
            Pos_tag=["<ROOT>"]
            for line in tqdm(lines):
                line=line.strip().split("\t")
                if line!=['']:
                    sentence.append(line[1])
                    dependency.append(int(line[6]))
                    arc_label.append(line[7])
                    Pos_tag.append(line[4])
                if line==[''] :
                    stack=[0]
                    lc1=[-1 for i in range(len(sentence))]
                    rc1=[-1 for i in range(len(sentence))]
                    lc2=[-1 for i in range(len(sentence))]
                    rc2=[-1 for i in range(len(sentence))]
                    lc1_label=['<NONE>' for i in range(len(sentence))]
                    lc2_label=['<NONE>' for i in range(len(sentence))]
                    rc1_label=['<NONE>' for i in range(len(sentence))]
                    rc2_label=['<NONE>' for i in range(len(sentence))]
                    buffer=[i for i in range(1,len(sentence))]
                    while len(stack)>1 or buffer!=[]:
                        if len(stack)==1:
                            """
                            这里是直接预测shift操作，只需要增加stack部分的长度就可以
                            """
                            feature=get_feature(sentence,stack,buffer,Pos_tag,lc1,rc1,lc2,rc2)
                            label="shift"
                            stack.append(buffer.pop(0))
                        elif len(stack)>=2:
                            """
                            这里长度大于2代表内部可能存在依赖，需要考虑lc和rc的更新
                            我原本的想法是完成整个句子后在进行右规约，但是这样做会使得右规约运算并不是出于两个单词之间，因此我决定加上要给检验函数判断
                            待规约词在后面是否还有依赖，也就是buffer中词是否还有depend 右边单词，每一次都检查，如果没有就直接进行shift
                            如果是shift，更新操作之对于buffer和stack进行
                            如果是arc_left和arc_right的话，需要：更新lc1,lc2或者rc1,rc2以及对应label
                            """
                            if dependency[stack[-1]]==stack[-2]:
                                is_denpency=False
                                for each in buffer:
                                    if dependency[each]==stack[-1]:
                                        is_denpency=True
                                        break
                                if is_denpency:
                                    label="shift"
                                    feature=get_feature(sentence,stack,buffer,Pos_tag,lc1,rc1,lc2,rc2)
                                    stack.append(buffer.pop(0))#这里为什么要加入shift，因为根据以来图不存在交叉线，可以推断出依赖必然连续
                                else:
                                    label='arc_right'+arc_label[stack[-1]]
                                    feature=get_feature(sentence,stack,buffer,Pos_tag,lc1,rc1,lc2,rc2)
                                    #这里无法保证最近……所以还是需要比较一下
                                    if rc1[stack[-2]]==-1:
                                        rc1[stack[-2]]=stack[-1]
                                        rc1_label[stack[-2]]=arc_label[stack[-1]]#按照文章格式，这里实质上是要从子节点获得标签
                                    elif rc2[stack[-2]]==-1 and math.fabs(stack[-2]-rc1[stack[-2]])>math.fabs(stack[-2]-stack[-1]):
                                        rc2[stack[-2]]=stack[-1]
                                        rc2_label[stack[-2]]=arc_label[stack[-1]]
                                    else:
                                        distance=math.fabs(stack[-1]-stack[-2])#z这里由于可能会乱序，因此还是比较一下top2的位置
                                        dis1=math.fabs(stack[-2]-rc1[stack[-2]])
                                        if dis1>distance:
                                            #这里我默认每一次计算顺序都是保证lc1的dis始终小于lc2的dis
                                            rc2_label[stack[-2]]=rc1_label[stack[-2]]
                                            rc2[stack[-2]]=rc1[stack[-2]]

                                            rc1_label[stack[-2]]=arc_label[stack[-1]]
                                            rc1[stack[-2]]=stack[-1]
                                        else:
                                            dis2=math.fabs(stack[-2]-rc2[stack[-2]])
                                            if distance<dis2:
                                                rc2_label[stack[-2]] = arc_label[stack[-1]]
                                                rc2[stack[-2]] = stack[-1]
                                    #更新完毕，这里需要将没有右边依赖并且下沉完整的数据出栈
                                    stack.pop()#弹出最后一个数据


                                #更新一次lc，保证接下来计算包含前缀信息
                                # if rc1[stack[-2]]==-1:#我原本准备在这里采用topk逐个遍历，但是后来发现其实这里i第一次出现的距离必定最小，第二次必定排名第二
                                #     rc1[stack[-2]]=stack[-1]
                                #     rc1_label[stack[-2]]=dependency[stack[-2]]
                                # elif rc2[stack[-2]]==-1:
                                #     rc2[stack[-2]]=stack[-1]
                                #     rc2_label[stack[-2]]=dependency[stack[-2]]
                                # stack.pop()#规约，就是右边这个直接出去,也叫做下沉
                            elif dependency[stack[-2]]==stack[-1]:
                                #z这里与上面完全相同，是进行左规约，不同的是，右规约可能会使得后面的左规约无法进行，但是右规约不会
                                #因此这里直接进行不需要等到buffer完全为空
                                label='arc_left_'+arc_label[stack[-2]]
                                feature=get_feature(sentence,stack,buffer,Pos_tag,lc1,rc1,lc2,rc2)
                                if lc1[stack[-2]]==-1:
                                    lc1[stack[-2]]=stack[-1]
                                    lc1_label[stack[-2]]=arc_label[stack[-2]]
                                elif lc2[stack[-2]]==-1:
                                    lc2[stack[-2]]=stack[-1]
                                    lc2_label[stack[-2]]=arc_label[stack[-2]]
                                stack.pop(-2)
                            else:
                                label='shift'
                                stack.append(buffer.pop(0))
                                feature = get_feature(sentence, stack, buffer, Pos_tag, lc1, rc1, lc2, rc2)
                        self.features.append(feature)
                        self.label.append(label)
                    arc_label = ['<NONE>']  # 因为dependency是从1开始计数，所以最好所有的都加一个初始化数值方便管理
                    sentence = ['<ROOT>']
                    dependency = ['<NONE>']
                    Pos_tag = ["<ROOT>"]
        features=[]
        labels=[]
        if tokenizer==None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',never_split=True)
            tokenizer.add_tokens(['<NULL>', '<ROOT>', '<NONE>'])
            for feature in self.features:
                ids=[]
                for word in feature:
                    if word not in tokenizer.vocab().keys() and feature not in tokenizer.get_added_vocab().keys():
                        tokenizer.add_tokens([word])
                    ids.append(tokenizer.encode(word)[1])
                assert len(ids)==48
                features.append(ids)
        else:
            for feature in self.features:
                ids=[]
                for word in feature:
                    ids.append(tokenizer.encode(word)[1])
                assert len(ids)==48
                features.append(ids)
        if label_dict==None:
            label_dict={}
            for label in self.label:
                if label not in label_dict.keys():
                    label_dict[label]=len(label_dict.keys())
                labels.append(label_dict[label])
        else:
            for label in self.label:
                labels.append(label_dict[label])
        self.label=torch.tensor(labels)
        self.features=torch.tensor(features)
        #将label、feature转化为数字信号
        self.label_dict=label_dict
        self.word_token=tokenizer

    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        return self.features[item],self.label[item]

# data=Data_reader('./data/data/')



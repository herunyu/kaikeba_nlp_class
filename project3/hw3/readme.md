# HW2 小结

## 中文语料

### nlu.yml
原有的data中的nlu.yml只有英文语料，全部更换为中文语料，并添加了function这个语料，用于询问机器人的功能
### rules.yml
修改原有的英文rules为中文并添加，分别为问候，再见，机器人，和功能
### stories.yml
添加了function path
### test_stories.yml
修改为中文tests并添加关于function的test
### domain.yml
添加function intent，并修改responses为中文

## pipeline设置
由于原pipeline是英文聊天机器人的pipeline，特别是WhitespaceTokenizer是通过判断空格来进行分词，而中文并不是通过空格来区分不同的单词，所以需要尝试修改为中文的pipeline。
查询到可以简单的用spacynlp来做，于是按照官方文档使用SpacyNLP,SpacyTokenizer,SpacyFeaturizer来对中文进行分词并提取Feature。

## 训练
* 训练nlu并测试
```bash
rasa train nlu
rasa test nlu
```
* 训练core并测试
```bash
rasa train core
rasa test core
```
* 也可以nlu和core一起训练、测试
```bash
rasa train
rasa test
```

## 小结
### spaCy
由于切换为spaCy来做分词，所以需要安装spaCy库，一开始没有安装所以训练会报错，安装可用pip直接安装
```bash
pip install spacy
```
然而除此之外还不够，如果直接又运行`rasa train`的话还是会报错  
<font color=red>
InvalidModelError: Model 'zh' is not a linked spaCy model. Please download and/or link a spaCy model, e.g. by running:  
python -m spacy download en_core_web_md  
python -m spacy link en_core_web_md en
</font>  
所以还需要安装中文模型以及link对应的模型作为spaCy zh的模型
```bash
python -m spacy download zh_core_web_sm
python -m spacy link zh_core_web_sm zh
```
注：spaCy中有三个中文模型，可以按需下载链接，这样就可以顺利训练了  

### FallbackClassifier
测试storiest时发现，做了utter_cheer_up的action后，由于模型判断ambiguity=0.3 > 0.1，所以并没有按照预期继续给出utter_did_that_help的action，而是给出了action_default_fallback的action，看了一下pipeline后分析这应该是由于pipeline中最后的FallbackClassifier所导致的原因。查看文档知道
FallbackClassifier的作用为如果nlu无法识别出用户的意图，就会给出action_default_fallback，由于没有设置这个action，所以默认在对话中没有输出。
* threshold:如果nlu输出的所有intent的概率都小于这个参数的值，则输出nlu_fallback对应的action
* ambiguity_threshold: 设置了这个参数，则nlu也会输出nlu_fallback的action以防止最高得分的两个intent也小于这个ambiguity_threshold.

所以回到config中删除了FallbackClassifier，重新训练后模型的表现和预期一样，后面要继续深入研究pipeline中的各种classifier。


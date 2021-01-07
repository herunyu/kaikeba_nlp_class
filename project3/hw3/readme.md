# HW3 小结

## 中文语料

### nlu.yml
添加了天气类的intent数据, 以及可查询天气的城市的lookup tables
### stories.yml
添加了weather path
### domain.yml
添加entities city和slots, 以及action_ask_weather

## pipeline设置
修改spacy为JiebaTokenizer, 分词效果会更好，添加RegexFeatureizer和RegexEntityExtractor

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

### ActionAskWeather
添加了查询天气的action，通过tracker获得填充在slot中NER识别的实体，把实体即待查询的天气城市，通过API查询天气结果后，把结果通过dispachter返回到前端。
### 尝试使用lookup table，但还存在问题
由于可查询天气的城市很多，我们不可能对每一个城市都写很多训练数据，这样的模型反而变成了类似规则的方法。所以采用部分训练数据并添加lookup table的方法。通过Regex Featurizer和RegexEntityExtractor来识别用户提问的实体。、在尝试做天气查询的action时，对于已经在训练数据中的实体比如北京，上海等城市，模型能准确识别出来，但是不在训练数据中的却无法识别出来，lookup table似乎并没有起到作用，初始以为是由于模型过拟合的问题导致的，但是把epoch减小之后，还是没有用处，目前暂时不知道如何解决该问题，但action和tracker中的slot的简单应用已大致掌握

### lookup table问题解决
经过询问老师，得知问题在于pipeline中使用了Regex Featurizer以及使用了带有ngram参数的CountVectorsFeaturizer, 这是由于这两个都是适用于英文而不是中文。移除这两个后即可表现正常。另外如果在训练数据中sample过少的话也会导致无法使用lookup table

### 尝试使用knowledge base
这个新功能貌似有挺多坑。做了一个city.json的知识图谱数据，用的是天气api中提供的数据。尝试使用省份去查找对应的城市，有时可以成功有时不能成功，而且回答是英文，并且并不准确类似这样：
```
问：广东城市

答：
Found the following objects of type 'city':
1: 梅州
2: 仁化
3: 龙门
4: 河源
5: 南雄

```
可以看到仁化，龙门，南雄明显不属于广东，不确定这是训练数据的问题还是模型的问题  

另外通过城市去查找所属的省份，会显示为：
```
'<function MyKnowledgeBaseAction.init.. at 0x7fa088f93940>' has the value '山西' for attribute 'provinceZh'.
```
这样的内容，感觉还需要琢磨如何能够定制化回答
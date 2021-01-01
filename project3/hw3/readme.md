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



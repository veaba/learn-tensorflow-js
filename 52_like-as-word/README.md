# 文本相似度比较

## 参考链接 

- [用Python进行简单的文本相似度分析](https://blog.csdn.net/xiexf189/article/details/79092629)
- [一个Python的数据分析库-中文社区文档](https://www.pypandas.cn/)
- [Basic regression: Predict fuel efficiency 回归](https://tensorflow.google.cn/tutorials/keras/regression)
- [使用 Keras 和 Tensorflow Hub 对电影评论进行文本分类](https://tensorflow.google.cn/tutorials/keras/text_classification_with_hub)

## 文本分析，根据现有的报告范文，一共有7段

分析
- 早些的数据不一定有境外输入
- 同上不一定有无症状数据统计
- 范文：http://www.nhc.gov.cn/xcs/yqtb/202004/6b7e8905b62f4cf89517cb0ebdf24d00.shtml 信息来自国家卫生健康委员会官方网站

|   段落序号     |描述的内容|python 解析的内容|
|--------------|---------|---------------|
|第一段，label 一|时间，新增的数据||
|第二段，label 二|治愈、解除密切观察、重症减少||
|第三段，label 三|境外输入||
|第四段，label 四|全国大体数据统计||
|第五段，label 五|湖北地区数据统计||
|第六段，label 六|无症状数据统计||
|第七段，label 七|港澳台数据||


## 软件设计

- 完成基础框架后
- 写爬虫，爬取当前全部数据，90%作为训练集
- 定时去爬取来判断，并写入数据库（前面几天先发送邮件出来看正确性）

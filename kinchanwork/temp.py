from torchvision import models, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import codecs

# データセットの読み込み
with codecs.open('gender-classifier-DFE-791531.csv', 'r', 'utf-8', 'ignore') as f:
    dataset = pd.read_csv(f)
# 列ごとに読み込み
for column_name, item in dataset.iteritems():
    print(f'[column_name] type: {type(column_name)}, value: {column_name}')
    print(f'[item] type: {type(item)}, value: ↓')
    print(item)
    print('================================')

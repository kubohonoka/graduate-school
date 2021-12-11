import torch
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

# VGG学習済みモデルの生成
vgg16 = models.vgg16(pretrained=True)
vgg16(# データセット['img'])

# TF-IDFの計算
tfidf_vectorizer = TfidfVectorizer(use_idf=True,lowercase=False)
# 文章内の全単語のTfidf値を取得
tfidf_matrix = tfidf_vectorizer.fit_transform(# データセット ['text'])

# PCAで次元圧縮
features=[]
gts = []
with torch.no_grad():
    for img, gt in eval_loader:
        img, gt = img.to(device), gt.to(device)
        feature = clf.extract(img)
        features.extend(feature.detach().cpu().numpy())
        gts.extend(gt.detach().cpu().numpy())
    features = np.concatenate([x.reshape(1, -1) for x in features], axis=0)
    pca = PCA(n_components=2)
    compressed = pca.fit_transform(features)

# CCAで共通の正準空間を学習

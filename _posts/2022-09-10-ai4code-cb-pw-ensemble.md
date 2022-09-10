---
title: graphcodebert+pairwise ensemble 0.8462
author: mng
date: 2022-09-10 11:15:04 +0900
categories: [NLP, KAGGLE, AI4CODE]
tags: [nlp, kaggle, graphcodebert, pairwise, ensemble]
math: true
sitemap :
  priority : 1.0
---

codebert에 model.from_pretrained에서, 단순히 codebert를 graphcodebert로 바꾸고 pairwise의 결과와의 단순 가중치 앙상블

# Rank Ensemble

캐글 리더보드에 저 비율을 쓰더라

```python
# Reading the submissions
df_1 = pd.read_csv('submission1.csv')
df_2 = pd.read_csv('submission2.csv')

# Averaging the indices and sorting the resulting submission by the aggregated ensembled indices
new_samples = []
for sample_idx in range(len(df_1)):
    # {'0a226b6a': 0, ...}
    sample_1 = {k: v for v, k in enumerate(df_1.iloc[sample_idx]['cell_order'].split(' '))}
    sample_2 = {k: v for v, k in enumerate(df_2.iloc[sample_idx]['cell_order'].split(' '))}
    for key in sample_1: 
        sample_1[key] = ((sample_1[key] * 0.748) + (sample_2[key] * 0.252))
    new_samples.append(' '.join([i[0] for i in list(sorted(sample_1.items(), key = lambda x: x[1]))]))
df_1['cell_order'] = new_samples
```

# Result

public: 0.8462

private: 0.8410 (holy shit…)
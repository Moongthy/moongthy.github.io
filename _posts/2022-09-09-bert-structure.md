---
title: BERT의 구조
author: mng
date: 2022-09-09 12:30:00 +0900
categories: [NLP, BERT]
tags: [nlp, bert]
math: true
sitemap :
  priority : 1.0
---

BERT 논문 저자들은 아래와 같이 두 가지 구성의 모델을 제시했다.

- BERT-base
- BERT-large

각각을 자세히 알아보자.

# 2.3.1 BERT-base

BERT-base는 12개의 인코더 레이어가 스택처럼 쌓인 형태로 구성되어 있다. 모든 인코더는 12개의 어텐션 헤드를 사용하며, 인코더의 피드포워드 네트워크는 768개 차원의 은닉 유닛으로 구성된다. 따라서 BERT-base에서 얻은 표현의 크기는 768이다.

앞으로 다음 표기법을 사용할 것이다.

- 인코더 레이어의 수는 $L$로 표시할 것이다.
- 어텐션 헤드는 $A$로 표시한다.
- 은닉 유닛은 $H$로 표시한다.

BERT-base 모델은 $(L, A, H)=(12, 12, 768)$ 가 되며, 총 변수의 수는 1억 1천만 개다. [그림 2-5]는 BERT-base 모델을 시각화 한 것이다.

<p>
  <img src="/assets/img/bert/fig2-5.png" alt>
  <em>그림 2-5 BERT-base</em>
</p>

# 2.3.2 BERT-large

BERT-large는 24개의 인코더 레이어가 스택처럼 쌓인 형태로 구성되어 있다. 모든 인코더는 16개의 어텐션 헤드를 사용하며, 인코더의 피드포워드 네트워크는 1024개의 은닉 유닛으로 구성된다. 따라서 BERT-large에서 얻은 표현의 크기는 1024가 된다.

BERT-large 모델은 $(L, A, H)=(24, 16, 1024)$가 되며, 총 변수의 수는 3억 4천만 개다. [그림 2-6]은 BERT-large 모델을 시각화 한 것이다.

<p>
  <img src="/assets/img/bert/fig2-6.png" alt>
  <em>그림 2-6 BERT-large</em>
</p>

# 2.3.3 그 밖의 여러 BERT 구조

앞의 두 가지 표준 구조 외에도 다른 조합으로 BERT를 구축할 수 있다. 더 작은 구조 중 일부는 다음과 같다.

- BERT-tiny: $(L, A, H)=(2, 2, 128)$
- BERT-mini: $(L, A, H)=(4, 4, 256)$
- BERT-small: $(L, A, H)=(4, 8, 521)$
- BERT-medium: $(L, A, H)=(8, 8, 521)$

컴퓨팅 리소스가 제한된 환경에서는 더 작은 BERT가 적합할 수 있다. 하지만 BERT-base, BERT-large와 같은 표준 구조가 더 정확한 결과를 제공하기 때문에 가장 널리 사용되고 있다.
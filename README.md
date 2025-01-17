# Datascience-Interview-Questions
- Source of questions: [Seongyun Byeon](https://github.com/zzsza/Datascience-Interview-Questions)

## Contents
- [통계 및 수학](#통계-및-수학)
- [분석 일반](#분석-일반)
- [머신러닝](#머신러닝)
- [딥러닝](#딥러닝)	 
	- [딥러닝 일반](#딥러닝-일반)
	- [컴퓨터 비전](#컴퓨터-비전)
	- [자연어 처리](#자연어-처리)
- [추천 시스템](#추천-시스템)
- [데이터베이스](#데이터베이스)
- [데이터 시각화](#데이터-시각화)
- [대 고객 사이드](#대-고객-사이드)
- [개인정보](#개인정보)

## 통계 및 수학
- 고유값(eigen value)와 고유벡터(eigen vector)에 대해 설명해주세요. 그리고 왜 중요할까요?  
  - <img src="https://latex.codecogs.com/svg.latex?\large&space;A\mathbf{v}=\lambda\mathbf{v},&space;\mathbf{v}\neq\mathbf{0}"/>  
  - 선형 변환 전과 후의 벡터가 서로 평행하면, 그 벡터는 선형  고유벡터이다. 고유벡터를 선형 변환의 축(axes)로 생각할 수 있다.
- 샘플링(Sampling)과 리샘플링(Resampling)에 대해 설명해주세요. 리샘플링은 무슨 장점이 있을까요?  
  - sampling: 단순 무작위, 계통, 층화, 군집 추출 등
  - resampling: bootstrap, cross-validation 등. 샘플의 통계량의 정밀도를 측정하거나 샘플의 임의의 부분 집합을 이용하여 모델을 검증함
- 확률 모형과 확률 변수는 무엇일까요?
  - 확률 변수: 정의역이 표본 공간인 함수. [**[ref]**](https://freshrimpsushi.github.io/posts/random-variable-and-probability-distribution/)  
  - 확률 모형: 데이터의 분포를 근사하는 모형  
- 누적 분포 함수와 확률 밀도 함수는 무엇일까요? 수식과 함께 표현해주세요
  - 확률 밀도 함수: 연속확률분포에서 확률변수가 특정값일 때의 우도(likelihood)
    - <img src="https://latex.codecogs.com/svg.latex?\small\displaystyle&space;\mathrm{Pr}(a\leq&space;X\leq&space;b)=\int_{a}^{b}f_{X}(x)dx"/>  
    - non-negative function
  - 누적 분포 함수
    - <img src="https://latex.codecogs.com/svg.latex?\small&space;F_{X}(x)=\mathrm{Pr}(X\leq&space;x)"/>  
    - <img src="https://latex.codecogs.com/svg.latex?\small&space;F_{X}(x)=\int_{-\infty}^{x}f_{X}(t)dt"/>  
- 베르누이 분포 / 이항 분포 / 카테고리 분포 / 다항 분포 / 가우시안 정규 분포 / t 분포 / 카이제곱 분포 / F 분포 / 베타 분포 / 감마 분포 / 디리클레 분포에 대해 설명해주세요. 혹시 연관된 분포가 있다면 연관 관계를 설명해주세요
- 조건부 확률은 무엇일까요?
  - <img src="https://latex.codecogs.com/svg.latex?\small&space;P(A|B)=\frac{P(A\cap&space;B)}{P(B)}"/>
- 공분산과 상관계수는 무엇일까요? 수식과 함께 표현해주세요
  - 공분산: <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;Cov[x,y]=E[(x-\bar{x})(y-\bar{y})]&space;"/>  
  - 상관계수: <img src="https://latex.codecogs.com/svg.latex?\small&space;r_{x,y}=\frac{E[(x-\bar{x})(y-\bar{y})]}{\sigma_{x}&space;\sigma_{y}}&space;"/>  
- 신뢰 구간의 정의는 무엇인가요?
  - L: lower bound, U: upper bound
  - <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;(L,U)\subset\mathbb{R}\text{&space;where&space;}(1-\alpha)=P[\theta\in&space;(L,U)]&space;"/>
- p-value를 고객에게는 뭐라고 설명하는게 이해하기 편할까요?
  - 먼저 어떠한 명제의 사실 여부를 판단하기 위해 가설(귀무가설과 대립가설)을 설정
  - 통계적 검정 전에 미리 유의수준을 설정함. 통상적으로 5%
  - 통계적 검정으로 통계량과 유의확률을 얻음
  - 유의확률이 유의수준보다 작으면 귀무가설을 기각하고 대립가설을 채택함
- p-value는 요즘 시대에도 여전히 유효할까요? 언제 p-value가 실제를 호도하는 경향이 있을까요?  
  - [**[ref]**](https://blog.minitab.com/en/understanding-statistics/what-can-you-say-when-your-p-value-is-greater-than-005)  
  - [**[ref1]**](https://freshrimpsushi.github.io/posts/significance-probability-p-value/)  
  - p-value가 유의수준보다 작으면 귀무가설을 기각한다. 유의수준은 일종의 threshold로만 기능하며 p-value의 대소에 대해 해석의 여지는 없다  
  - 예를 들어 유의수준 0.05에서 p-value가 0.01이든 0.001이든 귀무가설을 기각할 뿐, 양자 중 어느 것이 더 확실하게 혹은 강하게 기각한다고 말할 순 없다  
  - 반대로 유의수준 0.05에서 p-value가 0.7이든 0.051이든 똑같이 귀무가설을 기각할 수 없다. 아무리 유의수준에 가깝더라도 유의수준을 초과하는 순간 협상의 여지는 없다  
  - 그러나 심지어 학술적인 글에서조차 이를 교묘히 호도하는 표현을 사용한다  
- A/B Test 등 현상 분석 및 실험 설계 상 통계적으로 유의미함의 여부를 결정하기 위한 방법에는 어떤 것이 있을까요?
- R square의 의미는 무엇인가요? 
  - 데이터셋 종속변수(label)의 분산 가운데 모델 연산에 따라 독립변수(features)로 설명되는 분산의 비율
  - <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;R^2(y,\hat{y})=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}&space;"/>
- 평균(mean)과 중앙값(median)중에 어떤 케이스에서 뭐를 써야할까요?
  - 데이터에 이상치가 있을 때, 그 이상치에 의해 평균이 왜곡될 수 있음. 이를 피하면서 한 샘플의 대푯값을 얻고자 한다면, 중앙값을 사용하는 게 바람직함
- 중심극한정리는 왜 유용한걸까요?
  - 표본의 크기가 커질수록 표본의 분포는 정규분포를 따름. 따라서 표본이 충분히 크다면 그 분포를 정규분포로 간주하고 분석을 진행할 수 있음
- 엔트로피(entropy)에 대해 설명해주세요. 가능하면 Information Gain도요.
  - <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;Entropy=-\sum{p_i\log{p_i}}&space;"/>
  - 엔트로피가 클수록 분기된 두 그룹의 분포가 유사함. 즉 분기 기준이 효과적이지 못하고 불순도(impurity)가 크다고 할 수 있음
  - Information gain은 한 분기 스텝에서 불순도의 감소량
  - tree-based 알고리즘은 불순도가 가장 크게 감소하는 즉 Information gain이 가장 큰 분기 기준을 찾아 이를 근거로 부모 노드를 자식 노드로 나눔
- 요즘같은 빅데이터(?)시대에는 정규성 테스트가 의미 없다는 주장이 있습니다. 맞을까요?
  - 통계적 모델링의 목적: 분포를 가정하여 적은 수의 샘플로 모수를 추정  
  - 빅데이터, 머신러닝: estimation < prediction. 통계적 유의성보다도 모델의 성능(정확도, 오차 등)에 주목. 높은 성능의 모델로 비즈니스적 목표를 달성하고자 함  
  - 빅데이터에 대해서는 정규성 테스트가 팔요 없을 수 있음. 하지만 분석하려는 데이터가 빅데이터인지 먼저 살펴야 함  
  - 충분히 큰지, 모집단을 대표하는지, 오염은 없는지 등  
  - garbage in, garbage out
- 어떨 때 모수적 방법론을 쓸 수 있고, 어떨 때 비모수적 방법론을 쓸 수 있나요?
  - 비모수적 방법론: 샘플 크기가 작고 샘플의 분포를 가정하기 어려울 때
  - 빈도, 부호, 순위 등을 사용하므로 이상치에 영향을 덜 받음
- “likelihood”와 “probability”의 차이는 무엇일까요?
  - 연속 확률 변수에서
  - probability: pdf의 정적분 값
  - likelihood: pdf의 함숫값
- 통계에서 사용되는 bootstrap의 의미는 무엇인가요.
  - 복원 추출을 매우 많이 시행하였을 때 어떤 샘플이 한 번도 추출되지 않을 확률은 약 36.79%로 수렴함
  - 따라서 어떤 표본(표본집단)을 두 그룹으로 나눌 수 있음. 더불어 매 시행마다 그룹의 구성이 달라짐
  - 이를 학습에 적용하여 한정된 샘플로 다수의 분류기를 생성하는 기법이 bagging
- 모수(population)가 매우 적은 (수십개 이하) 케이스의 경우 어떤 방식으로 예측 모델을 수립할 수 있을까요?
  - overfitting 방지: 단순한 모델 채택 및 보수적인 적용, regularization 사용
  - outliers 관리
  - over-sampling
  - voting: 여러 단일 모델의 결과를 voting
- 베이지안과 프리퀀티스트간의 입장차이를 설명해주실 수 있나요?
  - [**[ref]**](https://freshrimpsushi.github.io/posts/bayesian-paradigm/)
  - frequentist: 표본이 모집단과 동질할 것이라는 가정. 따라서 (확률화를 통해 추출된) 표본의 크기가 클수록 모집단을 잘 표현함. 새로운 관측값이 표본을 통해 추정한 분포에 얼마나 잘 부합하는지 확인
  - bayesian: 사전분포를 가정하고 새로운 관측값을 반영해 베이즈 정리로 사후분포를 구하며 모수를 추정
- 검정력(statistical power)은 무엇일까요?
  - 귀무가설이 거짓일 때 대립가설을 채택할 확률
- missing value가 있을 경우 채워야 할까요? 그 이유는 무엇인가요?
  - 어떤 값으로든 채워야 함. 변수값이 하나라도 비어 있으면 모델이 학습 및 inference할 수 없다  
- 아웃라이어의 판단하는 기준은 무엇인가요?
  - 박스 플롯(사분위수): max, min을 벗어나면 아웃라이어로 판단
  - 시각화, 마할라노비스 거리
  - ESD(3-sigma), 기하평균 ± (2.5-sigma), Z-score
- 콜센터 통화 지속 시간에 대한 데이터가 존재합니다. 이 데이터를 코드화하고 분석하는 방법에 대한 계획을 세워주세요. 이 기간의 분포가 어떻게 보일지에 대한 시나리오를 설명해주세요
- 출장을 위해 비행기를 타려고 합니다. 당신은 우산을 가져가야 하는지 알고 싶어 출장지에 사는 친구 3명에게 무작위로 전화를 하고 비가 오는 경우를 독립적으로 질문해주세요. 각 친구는 2/3로 진실을 말하고 1/3으로 거짓을 말합니다. 3명의 친구가 모두 "그렇습니다. 비가 내리고 있습니다"라고 말했습니다. 실제로 비가 내릴 확률은 얼마입니까?
  - <img src="https://latex.codecogs.com/svg.latex?\normalsize\textstyle&space;1-\left(\frac{1}{3}\right)^3=\frac{26}{27}"/>
- 필요한 표본의 크기를 어떻게 계산합니까?
  - [**[ref]**](https://en.wikipedia.org/wiki/Sample_size_determination#Estimation_of_a_proportion)
  - Z: z-score, p: population proportion, epsilone: margin of error
  - <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;n=\frac{Z^{2}&space;p(1-p)}{\epsilon^2}&space;"/>
- Bias를 통제하는 방법은 무엇입니까?
  - 불편추정량을 사용
  - 일반적으로 분산과 편의(bias)는 trade-off 관계. 통계학에서는 편의를 0으로 통제하는 불편추정량을 사용하나 머신러닝에서는 오차가 최소화되도록 편의와 분산을 
  - <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;\mathrm{MSE}(\theta)=\mathrm{Var}\hat{\theta}+(\mathrm{Bias}{\hat{\theta}})^2&space;"/>
- 로그 함수는 어떤 경우 유용합니까? 사례를 들어 설명해주세요
  - 우도함수: 로그 우도함수를 사용하면 각 항의 곱셈을 덧셈으로 처리할 수 있음
  - 우측 편포(오른편 꼬리 분포): 로그 함수를 적용하여 대칭형 분포로 변환
  - 스케일을 줄이는 데 사용

##### [목차로 이동](#contents)

## 분석 일반
- 좋은 feature란 무엇인가요. 이 feature의 성능을 판단하기 위한 방법에는 어떤 것이 있나요?
  - 회귀: t-test 결과 p-value가 낮은 features
  - 분류: feature importance가 큰 features
  - coefficients, feature importances를 사용함
- "상관관계는 인과관계를 의미하지 않는다"라는 말이 있습니다. 설명해주실 수 있나요?
  - 여름에는 아이스크림이 많이 팔린다
  - 더불어 여름에는 익사사고가 늘어난다
  - 여름 중 특정 기간에 아이스크림 판매량과 익사사고 건수를 조사하면 양의 상관관계를 보일 수 있다
  - 그러나 이는 상관관계일뿐 인과관계를 의미하지 않는다. 아이스크림 판매가 익사사고를 유발한다고 말할 수 없다
- A/B 테스트의 장점과 단점, 그리고 단점의 경우 이를 해결하기 위한 방안에는 어떤 것이 있나요?
- 각 고객의 웹 행동에 대하여 실시간으로 상호작용이 가능하다고 할 때에, 이에 적용 가능한 고객 행동 및 모델에 관한 이론을 알아봅시다.
- 고객이 원하는 예측모형을 두가지 종류로 만들었다. 하나는 예측력이 뛰어나지만 왜 그렇게 예측했는지를 설명하기 어려운 random forest 모형이고, 또다른 하나는 예측력은 다소 떨어지나 명확하게 왜 그런지를 설명할 수 있는 sequential bayesian 모형입니다.고객에게 어떤 모형을 추천하겠습니까?
- 고객이 내일 어떤 상품을 구매할지 예측하는 모형을 만들어야 한다면 어떤 기법(예: SVM, Random Forest, logistic regression 등)을 사용할 것인지 정하고 이를 통계와 기계학습 지식이 전무한 실무자에게 설명해봅시다.
- 나만의 feature selection 방식을 설명해봅시다.
  - [Blog post](https://ywkim92.github.io/machine_learning/feature_selection/)
  - [implementation](https://github.com/ywkim92/Paper-implementation/blob/main/machine_learning/Feature_selection.ipynb)
  - recursive feature elimination
  - sequential feature selection
  - t-test, chi2 contingency
- 데이터 간의 유사도를 계산할 때, feature의 수가 많다면(예: 100개 이상), 이러한 high-dimensional clustering을 어떻게 풀어야할까요?
  - 차원축소

##### [목차로 이동](#contents)

## 머신러닝
- Cross Validation은 무엇이고 어떻게 해야하나요?
  - 학습 데이터를 K개로 나눔
  - K-1개 folds를 학습하고 남은 1개 fold에 대해 모델 검증
  - 보다 엄밀한 검증 가능. 하이퍼 파라미터 튜닝 등에 사용
- 회귀 / 분류시 알맞은 metric은 무엇일까요?
  - 회귀: MAE, MSE, RMSE, MAPE, R2  
  - 분류: Accuracy, recall, precision, F1 score, AUC  
- 알고 있는 metric에 대해 설명해주세요(ex. RMSE, MAE, recall, precision ...)
- 정규화(regularization)를 왜 해야할까요? 정규화의 방법은 무엇이 있나요?
  - 목적: 오버피팅 방지
  - L1(LASSO): <img src="https://latex.codecogs.com/svg.latex?\textstyle\min_{w}\sum_{i}L(y_i,x_i\cdot&space;w)+\lambda&space;||w||_{1}"/>   
  - L2(Rigde): <img src="https://latex.codecogs.com/svg.latex?\textstyle\min_{w}\sum_{i}L(y_i,x_i\cdot&space;w)+\lambda&space;||w||_{2}^{2}"/>   
  - Elastic Net: <img src="https://latex.codecogs.com/svg.latex?\textstyle\min_{w}\sum_{i}L(y_i,x_i\cdot&space;w)+\lambda(\alpha||w||_{1}+(1-\alpha)||w||_{2}^2)"/>   
- Local Minima와 Global Minima에 대해 설명해주세요.
- 차원의 저주에 대해 설명해주세요
  - 변수(feature)의 개수 즉 데이터의 차원이 증가할수록 변수공간 내 임의의 두 점 사이의 평균거리는 빠르게 증가하며 전체 공간에서 데이터가 차지하는 공간이 매우 적어짐  
  - 필요한 데이터의 수가 기하급수적으로 증가, 훈련이 느려짐  
- dimension reduction기법으로 보통 어떤 것들이 있나요?
- PCA는 차원 축소 기법이면서, 데이터 압축 기법이기도 하고, 노이즈 제거기법이기도 합니다. 왜 그런지 설명해주실 수 있나요?
  - [Blog post](https://ywkim92.github.io/machine_learning/PCA/)
  - [implementation](https://github.com/ywkim92/Paper-implementation/blob/main/machine_learning/Paper%20implementation_PCA.ipynb)  
  - 분산설명량 즉 공분산행렬의 고윳값이 큰 순서대로 상위 N개의 고유 벡터만을 components로 채택하므로 설명력이 낮은 변수 즉 노이즈가 제거될 수 있음  
  - 데이터 행렬을 특정 벡터에 투영(projection)했을 때 값들의 분산이 최대화하는 벡터를 찾는 알고리즘. 즉 데이터 행렬의 공분산 행렬의 고윳값 분해를 통해 얻음  
- LSA, LDA, SVD 등의 약자들이 어떤 뜻이고 서로 어떤 관계를 가지는지 설명할 수 있나요?
  - SVD
    - [Blog post](https://ywkim92.github.io/machine_learning/SVD/)  
    - [implementation](https://github.com/ywkim92/Paper-implementation/blob/main/data_preprocessing/SVD_implementation.ipynb)  
    - Singular Value Decomposition  
  - LSA
    - Latent Semantic Analysis  
    - truncated SVD를 활용하여 데이터의 차원을 축소시킴. 텍스트 데이터의 경우, 축소된 각각의 components를 하나의 topic으로 간주할 수 있음  
  - LDA
    - Latent Dirichlet Allocation  
    - 각 문서(=문장)의 단어를 분석하여 토픽별 단어의 분포 나아가 문서의 토픽 비율을 산출. [**[ref]**](https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda)
    - <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;p(z,\theta,\beta|w,\alpha,\eta)=\frac{p(z,\theta,\beta|\alpha,\eta)}{p(w|\alpha,\eta)}"/>
- Markov Chain을 고등학생에게 설명하려면 어떤 방식이 제일 좋을까요?
- 텍스트 더미에서 주제를 추출해야 합니다. 어떤 방식으로 접근해 나가시겠나요?  
  - 키워드 추출: TF-iDF, Bert
  - 텍스트 요약: TextRank, Seq2Seq
- SVM은 왜 반대로 차원을 확장시키는 방식으로 동작할까요? 거기서 어떤 장점이 발생했나요?
  - rbf, linear, sigmoid 등 kernel을 통해 차원을 확장
  - (특히 분류 문제에서) 확장된 공간에 mapping된 data points를 선형으로 구분하는 hyperplane을 찾을 수 있음
  - 예시: 서로 다른 두 클래스에 속하는 두 동심원. [**[ref]**](https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py)
- 다른 좋은 머신 러닝 대비, 오래된 기법인 나이브 베이즈(naive bayes)의 장점을 옹호해보세요.  
  - <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;\hat{y}=\displaystyle\arg\max_y&space;P(y)\prod_{i=1}^{n}P(x_i|y)"/>  
  - 구현이 간단하며 샘플과 변수의 수가 많더라도 빠른 학습 및 예측이 가능하다. 전제(각 변수의 독립, 특정 분포를 따름)가 타당하다면 양호한 정확도를 기대할 수 있다.  
- Association Rule의 Support, Confidence, Lift에 대해 설명해주세요.
  - support: <img src="https://latex.codecogs.com/svg.latex?P(A\cap&space;B)"/>  
  - confidence(A to B): <img src="https://latex.codecogs.com/svg.latex?P(B|A)"/>  
  - lift: <img src="https://latex.codecogs.com/svg.latex?\textstyle\frac{P(B|A)}{P(B)}"/>  
- 최적화 기법중 Newton’s Method와 Gradient Descent 방법에 대해 알고 있나요?
  - Newton's method: <img src="https://latex.codecogs.com/svg.latex?\small&space;x_{n+1}=x_{n}-\frac{f(x_{n})}{f'(x_{n})}"/>
    - <img src="https://latex.codecogs.com/svg.latex?\small&space;f(x)=0"/> 의 해를 찾는 수치해석적 방법  
  
  - Gradient decent: <img src="https://latex.codecogs.com/svg.latex?\normalsize&space;\theta_{n+1}=\theta_{n}-\eta\nabla_{\theta}&space;J(\theta_{n})"/>  
    - 손실함수 <img src="https://latex.codecogs.com/svg.latex?\small&space;J(\theta)"/> 를 최소화하는 파라미터를 찾는 과정  
    - 그래디언트 <img src="https://latex.codecogs.com/svg.latex?\small&space;\nabla&space;J(\theta)=\frac{\partial&space;J}{\partial\theta}"/> 를 이용함
- 머신러닝(machine learning)적 접근방법과 통계(statistics)적 접근방법의 둘간에 차이에 대한 견해가 있나요?
  - 통계적 접근: 특정 현상이나 상관관계를 잘 설명하는 분포나 모형이 있다고 가정하고 이를 표현하는 모수(parameters)를 찾고자 함  
  - 머신러닝: 손실함수 혹은 성능지표(metric)을 상정하고 이를 최적화하는 방향으로 모델을 데이터에 적합(fit, train)시킴  
- 인공신경망(deep learning이전의 전통적인)이 가지는 일반적인 문제점은 무엇일까요?
  - XOR problem: 하나의 perceptron만으로는 해결할 수 없음
- 지금 나오고 있는 deep learning 계열의 혁신의 근간은 무엇이라고 생각하시나요?
  - 여러 layers를 쌓으면서도 적절한 activation function을 활용하여 기울기 소실 없이 유의미한 학습 가능
  - 오차 역전파를 통해 gradient를 계산하고 이를 사용해 가중치 업데이트. 휴리스틱 방법론 등 사람의 개입 여지 축소
  - 빅데이터를 학습한 사전 학습 모델을 나의 과제에 활용할 수 있음
- ROC 커브에 대해 설명해주실 수 있으신가요?
  - false positive rate를 x축으로 true positive rate(recall=sensitivity)를 y축으로 하는 그래프
  - AUC: ROC 커브와 y=0, x=1로 둘러쌓인 면적의 넓이
- 여러분이 서버를 100대 가지고 있습니다. 이때 인공신경망보다 Random Forest를 써야하는 이유는 뭘까요?
  - Random Forest는 많은 Decision tree의 결과를 앙상블하는 기법  
  - 트리를 각 서버에 할당하여 계산 수행할 수 있음  
- K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)
  - 이상치에 민감함
  - 군집의 개수를 미리 설정해야 함
  - non-convex shape 데이터에 적용하기 어려움
- L1, L2 정규화에 대해 설명해주세요
  - 특정 가중치가 업데이트를 거듭하며 0에 가까워진다. 따라서 신경망이 단순해 짐
  - L1: <img src="https://latex.codecogs.com/svg.latex?\small&space;L(x,y)=\sum_{i=0}^{n}(y_{i}-\sum_{j=0}^{m}x_{ij}W_{j})+\lambda\sum_{j=0}^{m}|W_j|"/>
  - L2: <img src="https://latex.codecogs.com/svg.latex?\small&space;L(x,y)=\sum_{i=0}^{n}(y_{i}-\sum_{j=0}^{m}x_{ij}W_{j})+\lambda\sum_{j=0}^{m}W_{j}^{2}"/>
- XGBoost을 아시나요? 왜 이 모델이 캐글에서 유명할까요?
- 앙상블 방법엔 어떤 것들이 있나요?
  - bagging: bootstrap를 사용해 하나의 원 데이터에서 다수의 데이터셋을 만들고 기본 분류기가 서로 다른 데이터셋을 학습하게 함. 각 분류기의 예측 결과를 합산
  - boosting: 잘못 분류된 샘플에 큰 가중치를 두어 학습, 약한 분류기를 강건하게 하는 기법
  - random forest: 다수의 의사결정나무를 앙상블하여 예측
- SVM은 왜 좋을까요?
- feature vector란 무엇일까요?
  - 각 feature를 하나의 basis로 보고 데이터셋을 모든 bases의 선형결합으로 표현되는 공간으로 봄
- 좋은 모델의 정의는 무엇일까요?
  - 양호한 성능 지표를 보이는 모델
  - 어떠한 데이터셋에 대해서도 유사한 성능을 보이는 모델
- 50개의 작은 의사결정 나무는 큰 의사결정 나무보다 괜찮을까요? 왜 그렇게 생각하나요?
  - 나무의 개수가 많을수록 임의성이 확보되어 모델이 보다 robust하다
- 스팸 필터에 로지스틱 리그레션을 많이 사용하는 이유는 무엇일까요?  
  - coefficients 기반, 직관적 모델, 결과 해석에 유리  
  - Binary classification 문제에 적합, 스팸 확률 산출 가능  
- OLS(ordinary least squre) regression의 공식은 무엇인가요?
  - <img src="https://latex.codecogs.com/svg.latex?\large&space;\mathbf{y}=X\beta+\epsilon"/>  
  - <img src="https://latex.codecogs.com/svg.latex?\large&space;\hat{\beta}=(X^{T}&space;X)^{-1}X^{T}\mathbf{y}"/>  

##### [목차로 이동](#contents)

## 딥러닝
## 딥러닝 일반
- 딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?
  - 딥러닝: 인공신경망을 깊게 쌓아 학습하는, 머신러닝 기법 중 하나
  - 차이
    - 전통 머신러닝 모델의 성능을 개선하기 위해서는 사용자 즉 사람의 간섭이 불가피. 그러나 딥러닝은 환경과 앞선 학습의 오차를 기반으로 (스스로) 학습 및 업데이트 가능
    - 출력값 형태의 종류: 머신러닝의 출력값 즉 예측값은 카테고리나 숫자. 그러나 딥러닝은 자연어, 이미지, 음성, 비디오 등을 출력(inference)할 수 있음
- 왜 갑자기 딥러닝이 부흥했을까요?
  - XOR 문제, gradient vanishing 등 문제가 있었지만 그에 대한 솔루션이 제시되어 신경망을 깊이 쌓아도 정상적으로 학습할 수 있게 됨
  - 신경망을 깊이 쌓을수록 데이터의 다양한 특성을 학습할 수 있으며 분류 등 성능 향상을 기대할 수 있었음
- 마지막으로 읽은 논문은 무엇인가요? 설명해주세요
- Cost Function과 Activation Function은 무엇인가요?
  - Cost Function: 모델 예측의 오차 정도를 측정하는 함수, 손실 함수를 최소화하는 방향으로 모델 학습
  - Activation Function: 뉴런(노드)에서 출력값을 결정하는 함수
- Tensorflow, Keras, PyTorch, Caffe, Mxnet 중 선호하는 프레임워크와 그 이유는 무엇인가요?
- Data Normalization은 무엇이고 왜 필요한가요?
  - 각 feature의 평균, 표준편차나 최댓값, 최솟값 등 scale을 통일하는 처리 기법
  - 이점: 학습 시간 단축 및 모델 성능 향상
  - 데이터 정규화를 하지 않으면 feature에 따라 가중치 변동 폭이 달라(손실함수에서 경사가 가장 가파른 방향이 일정치 않음) optimization 시 zigzagging이 심해짐
- 알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등)
  - sigmoid: <img src="https://latex.codecogs.com/svg.latex?\small\textstyle&space;\sigma(x)=\frac{1}{1+e^{-x}}"/>
  - tanh: <img src="https://latex.codecogs.com/svg.latex?\small\textstyle&space;\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}"/>
  - ReLU: <img src="https://latex.codecogs.com/svg.latex?\small\textstyle&space;f(x)=\mathrm{max}(0,x)"/>
  - Leaky ReLU: <img src="https://latex.codecogs.com/svg.latex?\small\textstyle&space;f(x)=\begin{cases}x,&\text{if}\;\;x\geq0\\0.01x,&\text{if}\;\;x<0\end{cases}"/>
- 오버피팅일 경우 어떻게 대처해야 할까요?
  - 데이터 양 증가, 모델 복잡도(layers 개수, parameters 개수 등) 감소, weight decay, dropout
- 하이퍼 파라미터는 무엇인가요?
  - 모델 내부 연산에 의해 산출되는 값이 아닌, 사용자가 직접 입력해주어야 하는 값
  - CNN의 filter 개수, kernel size, stride / dense layer의 node 개수 등
- Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요?
  - [implementation](https://github.com/ywkim92/Paper-implementation/blob/main/neural_network/Initializer.ipynb)
  - 잘못된 가중치 초기화는 신경망의 표현력을 제한할 수 있음
  - Xavier(혹은 Glorot): 활성화 함수가 Sigmoid나 tanh일 때 사용  
    - truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in+fan_out))
  - He: 활성화 함수가 ReLU일 때 사용  
    - truncated normal distribution centered on 0 with stddev = sqrt(2 / fan_in)
- 볼츠만 머신은 무엇인가요?
- 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?
	- Non-Linearity라는 말의 의미와 그 필요성은?
	  - linear map이 아닌 경우
	    - <img src="https://latex.codecogs.com/svg.latex?\small&space;f(x+y)=f(x)+f(y)"/>  
	    - <img src="https://latex.codecogs.com/svg.latex?\small&space;f(\alpha&space;x)=\alpha&space;f(x)"/>
	  - 선형이 아닌 관계를 표현하기 위해 필요함
	  - 심층신경망에서는 은닉층을 다수 쌓는 게 관건. 그러나 활성화 함수가 선형함수이면 은닉층을 쌓아도 결국 은닉층이 하나인 모델과 매한가지  
	  - 따라서 활성화 함수는 비선형 함수여야 함
	- ReLU로 어떻게 곡선 함수를 근사하나?
	  - ReLU는 선형 함수와 비선형 함수가 결합한 형태
	  - 여러 은닉층에 걸쳐 ReLU가 사용되면, 각 선형 함수가 부분적으로 결합되어 비선형 함수를 근사하게 됨
	- ReLU의 문제점은?
	  - dying ReLu: 입력이 음수이면 함숫값이 0이 됨
	- Bias는 왜 있는걸까?
	  - 활성화 함수의 trigger를 조절하는 데 사용됨
- Gradient Descent에 대해서 쉽게 설명한다면?
	- 왜 꼭 Gradient를 써야 할까? 그 그래프에서 가로축과 세로축 각각은 무엇인가? 실제 상황에서는 그 그래프가 어떻게 그려질까?
	- GD 중에 때때로 Loss가 증가하는 이유는?
	 - 최적화 기법에 따라 gradient, learning rate가 달라짐. 경우에 따라 loss가 증가할 수도 있음
	- 중학생이 이해할 수 있게 더 쉽게 설명 한다면?
	- Back Propagation에 대해서 쉽게 설명 한다면?
	  - 손실 함수의 그래디언트를 계산하여 가중치를 업데이트하는 방법
- Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?   
  - [**[ref]**](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/)  
  - 이유
    - loss에서 차이를 보이더라도 metric은 유사할 수 있음
    - trainset과 test 혹은 inference 데이터의 양상은 다름
    - 파라미터가 많아질수록 손실함수는 고차원이 되는데 이때는 optimal local minima가 다수 존재할 수 있음 
  - GD가 Local Minima 문제를 피하는 방법은? 
	  - stochastic GD: 학습 데이터를 여러 개의 배치(batch)로 나누어(무작위 비복원추출) 모델 학습 및 가중치 업데이트 진행. single batch GD에 비해 매 batch마다 조금씩 다른 방향(gradient)로 
	  - 학습률(learning rate) 조정
  - 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?
    - 현실적으로 이러한 방법은 없음
    - 학습 데이터에서 global minimum일지라도 평가 데이터에서는 아닐 수 있음
- Training 세트와 Test 세트를 분리하는 이유는?
	- Validation 세트가 따로 있는 이유는?
	  - 가중치를 업데이트하는 과정에서 validation loss를 따로 파악하여 오버피팅을 방지하기 위함  
	- Test 세트가 오염되었다는 말의 뜻은?
	  - Test 세트를 두는 이유는 모델이 한번도 학습하지 못한 데이터로 보다 엄밀하게 모델의 성능을 파악하기 위함  
	  - 이러한 목적으로 분리된 Test 세트가 모델 학습에 어떠한 방식으로든 개입되었다면 이를 오염되었다고 함  
	- Regularization이란 무엇인가?
- Batch Normalization의 효과는?
	- Dropout의 효과는?
	  - 일부 노드를 무작위로 비활성화함으로써 노드 사이의 의존성을 완화함
	- BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?
	  - 은닉층 뒤에 사용
	  - 앞층 출력의 평균과 분산이 항상 같음. 분포의 변화 즉 covariate shift가 없음. 앞층 출력의 영향을 덜 받음. 
	  - 독립적, 빠르고 안정적인 학습
	- GAN에서 Generator 쪽에도 BN을 적용해도 될까?
- SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?
	- SGD에서 Stochastic의 의미는?
	  - 미니배치로 묶이는 데이터가 확률적 즉 무작위로 정해진다
	- 미니배치를 작게 할때의 장단점은?
	  - 장점: 가중치를 자주 수정할 수 있으므로 global minimum에 접근하는 데 유리
	  - 단점: 행렬 연산의 이점을 살리기 어렵다, 가중치 업데이트 시 진동이 심하다
	- 모멘텀의 수식을 적어 본다면?  
	  - `velocity = momentum * velocity - learning_rate * g` 
	  - `w = w + velocity`  
- 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?
	- 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
	- Back Propagation은 몇줄인가?
	- CNN으로 바꾼다면 얼마나 추가될까?
- 간단한 MNIST 분류기를 TF, Keras, PyTorch 등으로 작성하는데 몇시간이 필요한가?
	- CNN이 아닌 MLP로 해도 잘 될까?  
	  - MLP는 이미지의 위치정보를 보존하지 못하므로 일반적으로 성능이 떨어짐  
	- 마지막 레이어 부분에 대해서 설명 한다면?  
	  - 분류기: Fully connected layer = dense layer = linear layer  
	  - activation function: sigmoid or softmax  
	- 학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?  
	  - keras의 경우 compile 시 metrics에 `MeanSquaredError`를 추가  
	- 만약 한글 (인쇄물) OCR을 만든다면 데이터 수집은 어떻게 할 수 있을까?
- 딥러닝할 때 GPU를 쓰면 좋은 이유는?
	- 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?
	- GPU를 두개 다 쓰고 싶다. 방법은?
	- 학습시 필요한 GPU 메모리는 어떻게 계산하는가?
- TF, Keras, PyTorch 등을 사용할 때 디버깅 노하우는?
  - input과 output array의 dimensionality를 살핀다
- 뉴럴넷(neural network)의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가?  
  - 신경망을 깊이 쌓을수록 파라미터의 수가 급격히 증가하고 이를 학습시키기 위해서는 상응하는 양의 데이터가 필요함  
  - 그에 더해 데이터 불균형이 없어야 함  

##### [목차로 이동](#contents)

## 컴퓨터 비전
- OpenCV 라이브러리만을 사용해서 이미지 뷰어(Crop, 흑백화, Zoom 등의 기능 포함)를 만들어주세요
  - open image: `img = cv2.imread(fileName, cv2.IMREAD_COLOR)`
  - crop: `img_crop = img[h1:h2, w1:w2, :]`
  - gray scale: `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`  
  - zoom: `cv2.resize(img, dsize = None, fx, fy, interpolation = cv2.INTER_CUBIC)`  
  - save: `cv2.imwrite(fileName, img)`  
- 딥러닝 발달 이전에 사물을 Detect할 때 자주 사용하던 방법은 무엇인가요?  
  - 이미지에서 hand-crafted feature를 추출하여 객체를 검출. 수동으로 사전에 정의된 알고리즘, 전문가적 지식에 의존  
  - SIFT(Scale-Invariant Feature Transform), HOG(Histogram of Oriented Gradients) 등  
- Fatser R-CNN의 장점과 단점은 무엇인가요?  
  - 장점: region proposal 단계(물체의 대략적인 위치 식별)를 RPN으로 수행(R-CNN은 selective search 사용). 속도 개선, 객체 검출 전 단계를 신경망으로만 구성  
  - 단점: 여러 단계를 거쳐 학습이 수행되므로 시간이 오래 걸린다  
- dlib은 무엇인가요?  
  - 머신러닝 알고리즘 등이 포함된, C++ 툴킷. 여러 기능 가운데 HOG feature를 활용한 face detection 등이 유명함  
- YOLO의 장점과 단점은 무엇인가요?
  - one stage detector의 일종  
  - region proposal 단계를 건너뛰고 CNN 층에서 위치와 클래스를 모두 예측함  
  - 장점: 빠른 처리 속도  
    - 이미지를 여러 개의 격자로 나누고 각 격자에 대해 클래스를 분류  
    - 위 과정을 통해 여러 경계박스 후보 생성  
    - NMS로 최종 경계박스 추출    
  - 단점: 성능 하락  
- 제일 좋아하는 Object Detection 알고리즘에 대해 설명하고 그 알고리즘의 장단점에 대해 알려주세요  
  - 그 이후에 나온 더 좋은 알고리즘은 무엇인가요?
- Average Pooling과 Max Pooling의 차이점은?
  - Average Pooling: pool size 내의 값의 평균  
  - Max Pooling: pool size 내의 값의 최댓값  
- Deep한 네트워크가 좋은 것일까요? 언제까지 좋을까요?
  - 보다 많은 특성을 탐지 및 학습할 수 있다  
  - 단 네트워크가 깊어지면 gradient vanishing이 나타날 수 있으며 이는 모델의 학습을 방해한다  
- Residual Network는 왜 잘될까요? Ensemble과 관련되어 있을까요?  
  - ResNet에서 building blocks를 쌓는다는 것은 입력과 출력을 연결하는 O(2^n)개의 implicit paths를 설정함과 같음  
  - 이러한 paths들은 서로 강하게 의존하지 않고 결국 ensemble과 유사하게 행동함  
  - [**[paper]**](https://arxiv.org/abs/1605.06431)  
- CAM(Class Activation Map)은 무엇인가요?
  - 분류 과제를 위해서는 Flatten layer, Fully connection layer가 필요함  
  - 그러나 이를 거치면서 위치 정보가 소실됨  
  - CNN layer에 Global average pooling layer를 붙여 objects의 위치 정보를 보존할 수 있음  
  - 이를 활용하여 class specific activation map 추출. 즉 모델이 입력 이미지를 특정 class로 분류하는 데 영향을 미친 object를 heatmap 형태로 나타냄  
- Localization은 무엇일까요?
  - 이미지 속 object의 위치를 설정하는 알고리즘  
- 자율주행 자동차의 원리는 무엇일까요?  
  - Object detection: 전방 카메라에 잡힌 물체의 종류와 위치를 판단할 때 사용  
- Semantic Segmentation은 무엇인가요?
  - 이미지의 픽셀별로 클래스를 예측하는 알고리즘  
- Visual Q&A는 무엇인가요?
  - 이미지와 질문(자연어)이 주어지면 이미지의 내용을 바탕으로 질문에 대한 답(자연어)을 제시하는 모델  
  - [**[paper]**](https://arxiv.org/abs/1505.00468)
- Image Captioning은 무엇인가요?
  - 이미지를 설명하는 텍스트를 생성하는 알고리즘  
- Fully Connected Layer의 기능은 무엇인가요?
  - 합성곱층에서 추출된 특징을 토대로 이미지를 분류함  
- Neural Style Transfer는 어떻게 진행될까요?
  - 스타일 이미지의 스타일을 추출하여 콘텐트 이미지에 적용  
- CNN에 대해서 아는대로 얘기하라  
  - CNN이 MLP보다 좋은 이유는?  
    - MLP를 사용하면 (1) 2차원 이미지가 1차원 벡터로 바뀌며 소실되는 feature(공간적 특징)가 많다 (2) 이미지의 크기가 커질수록, 깊이 쌓을수록 파라미터의 수가 급격히 증가한다  
    - CNN: 2차원 구조의 정보를 잃지 않으면서도 MLP(FCL)에 비해 파라미터의 수가 적다  
  - 어떤 CNN의 파라미터 개수를 계산해 본다면?  
    - kernel params + bias params: `(kernel_h * kernel_w) * num_input_channels * num_filters + num_filters`  
  - 주어진 CNN과 똑같은 MLP를 만들 수 있나?  
  - 풀링시에 만약 Max를 사용한다면 그 이유는?  
    - pooling window 내에서 가장 특징적인 값을 통해 입력 이미지의 특성을 추출함  
  - 시퀀스 데이터에 CNN을 적용하는 것이 가능할까?
- CV models
  - AlexNet: 5 CNN layers & 3 FC layers. Relu, image augmentation, Batch Normalization, dropout, L2 regularization 
  - VGGNet: same hyper-parameters through the whole convolution and pooling layers
  - GoogLeNet: 다양한 크기의 커널을 동시에 사용하고 reduce layer가 적용된 inception module 사용  
  - ResNet: Gradient vanishing 문제 해결을 위해 Residual block(skip connection) 도입  
  - MobileNet: 모델 경량화를 위해 Depthwise separable convolution 도입  
  - EfficientNet: 신경망의 층수, 입력 이미지 크기, 합성곱층 필터의 수 등을 최적화  

##### [목차로 이동](#contents)

## 자연어 처리
- One Hot 인코딩에 대해 설명해주세요  
  - 한 특성의 unique value가 N개라고 할 때, 그 특성을 0과 1로 구성된 N차원의 벡터로 표현하는 방법  
- POS 태깅은 무엇인가요? 가장 간단하게 POS tagger를 만드는 방법은 무엇일까요?
  - part-of-speech 즉 품사 태깅  
  - POS tagger를 만드는 방법  
    - 데이터셋: tokenized된 문장과 각 토큰에 대응되는 품사로 구성된 데이터  
    - 모델: RNN, LSTM 등 
- 문장에서 "Apple"이란 단어가 과일인지 회사인지 식별하는 모델을 어떻게 훈련시킬 수 있을까요?
  - 개체명 인식 모델을 구축
- 뉴스 기사에 인용된 텍스트의 모든 항목을 어떻게 찾을까요?
- 음성 인식 시스템에서 생성된 텍스트를 자동으로 수정하는 시스템을 어떻게 구축할까요?
- 잠재론적, 의미론적 색인은 무엇이고 어떻게 적용할 수 있을까요?
- 영어 텍스트를 다른 언어로 번역할 시스템을 어떻게 구축해야 할까요?  
  - 알고리즘: 다대다 모델. seq2seq, transformer 등  
- 뉴스 기사를 주제별로 자동 분류하는 시스템을 어떻게 구축할까요?  
  - 토픽 모델링
  - BoW 기반: LSA, LDA 등  
  - BERT 기반: SBERT(문장 임베딩), CTM(Contextualized Topic Models), BERTopic 등  
- Stop Words는 무엇일까요? 이것을 왜 제거해야 하나요?  
  - 말과 말 사이의 관계, 문법적 관계 등을 나타내는 말  
  - 특정한 의도를 가진 문장에만 나타나지 않고 두루 등장. 한국어의 조사, 접속사 등  
  - 텍스크에 기초하여 특정 태스크를 수행하는 데 도움이 되지 않음. 그러나 자주 등장하므로 이를 제거하지 않으면 모델링 시 자원이 불필요하게 소모됨  
  - 불용어를 판단하는 절대적 기준은 없음. 태스크나 데이터의 특성에 맞게 설정해야 함  
- 영화 리뷰가 긍정적인지 부정적인지 예측하기 위해 모델을 어떻게 설계하시겠나요?
  - 긍정 혹은 부정으로 labeling된 자연어 문장 데이터셋 준비
  - 텍스트 전처리: tokenizing, encoding, stop words 제거 등
  - 학습 및 모델링 검증
- TF-IDF 점수는 무엇이며 어떤 경우 유용한가요?
  - term frequency - inverse document frequency
  - <img src="https://latex.codecogs.com/svg.latex?\normal\textstyle&space;\mathrm{tf}(t,d)=\frac{f_{t,d}}{\sum_{t'\in&space;d}f_{t',d}}"/>
  - <img src="https://latex.codecogs.com/svg.latex?\normal\textstyle&space;\mathrm{idf}(t,D)=\log{\frac{|D|}{|\left\{d\in&space;D:\;t\in&space;d\right\}|}"/>
  - <img src="https://latex.codecogs.com/svg.latex?\normal\textstyle&space;\mathrm{tfidf}(t,d,D)=\mathrm{tf}(t,d)\cdot\mathrm{idf}(t,D)"/>
  - 문서별 특성을 추출하거나 문서간 유사도를 계산하는 데 유용함. DTM 산출 시 각 단어의 중요도를 반영할 수 있음
- 한국어에서 많이 사용되는 사전은 무엇인가요?
- Regular grammar는 무엇인가요? regular expression과 무슨 차이가 있나요?
- RNN에 대해 설명해주세요
  - Recurrent Neural Network
  - 현시점의 결과를 예측하는 데 현시점의 입력값과 이전 시점의 은닉 상태를 함께 고려하는 모델
  - <img src="https://latex.codecogs.com/svg.latex?\normal\textstyle&space;h_t=tanh{W_x&space;x_t+W_h&space;h_{t-1}+b}"/>
  - Problem of long-term dependencies: 시점이 길어지면 초기 시점의 정보가 손실되는 문제 발생
  - 현시점을 예측하는 데 중요한 부분과 현시점 사이 거리가 길어지면 성능 하락
- LSTM은 왜 유용한가요?  
  - Long-Short Term Memory network
  - long-term dependencies를 학습하는 데 유효함
  - memory cell, input gate, forget gate, output gate 
  - [implementation](https://github.com/ywkim92/Paper-implementation/blob/main/neural_network/LSTM.ipynb)  
- Translate 과정 Flow에 대해 설명해주세요
- n-gram은 무엇일까요?
  - n개의 연속된 단어를 하나의 토큰으로 취급  
  - 언어 모델에서 문장의 다음 단어를 예측하는 데 해당 단어 직전 등장한 N개의 단어에 근거하는 방식. 문장 전체의 맥락을 고려하지 못함  
- PageRank 알고리즘은 어떻게 작동하나요?
- depedency parsing란 무엇인가요?
- Word2Vec의 원리는?
	- 원리: 주변(skip-gram) 혹은 중심(CBOW) 단어를 예측하는 인공신경망을 학습, 단어를 특정 차원의 실수 벡터로 표현. 단어의 의미가 수치화되며 단어 간 유사도를 비교할 수 있음  
	- 남자와 여자가 가까울까? 남자와 자동차가 가까울까? - 학습한 corpus에 따라 다를 것이다  
	- 번역을 Unsupervised로 할 수 있을까? - A언어의 단어와 B언어의 단어가 동일한 벡터 공간에 임베딩되어 있다면, 단어 간 유사도를 비교하는 방식으로 word-level 번역 가능  
- NLP models. 
  - Seq2Seq 계열: RNN, LSTM, GRU 등. 입력 도메인의 문장을 출력 도메인으로 바꿈. 장기 기억 문제, 병렬 처리 불가 등 문제가 존재  
  - BERT: transformer encoder 사용. base - 12 layers, 768 dims. 사전학습 MLM, NSP
  - Electra: 사전학습 Replaced token detection. generator & discriminator 네트워크 사용  
  - 

##### [목차로 이동](#contents)

## 추천 시스템
- 추천 시스템에서 사용할 수 있는 거리는 무엇이 있을까요?
  - cosine, jaccard, pearson, msd
- User 베이스 추천 시스템과 Item 베이스 추천 시스템 중 단기간에 빠른 효율을 낼 수 있는 것은 무엇일까요?
  - item-based rec sys.
  - 일반적으로 유저의 숫자는 아이템 숫자에 비해 매우 크고 그 특성 역시 많다. 더불어 생애 주기에 따라 특성이 변한다
- 성능 평가를 위해 어떤 지표를 사용할까요?
  - 정확도: mae, rmse
  - 랭킹: hite rate, ndcg
  - diversity
  - novelty
- Explicit Feedback과 Implicit Feedback은 무엇일까요? Impicit Feedback을 어떻게 Explicit하게 바꿀 수 있을까요?
  - explicit: 유저가 특정 아이템에 대한 호불호를 명시적으로 표현. 좋아요/싫어요, 평점 등
  - implicit: 유저와 아이템 간의 상호작용. 아이템 조회, 구매 등
  - 특정한 로직을 설정하고 그에 따라 implicit을 통해 explicit 추정
- Matrix Factorization은 무엇인가요? 해당 알고리즘의 장점과 단점은?
  - Utility matrix를 user latent matrix와 item latent matrix의 곱으로 표현
  - 장점: 비교적 높은 정확도, 확장성(scalable), sparse dataset에도 적용 가능
  - 단점: 아이템 혹은 유저의 특성을 반영할 수 없음, cold start
- SQL으로 조회 기반 Best, 구매 기반 Best, 카테고리별 Best를 구하는 쿼리를 작성해주세요
- 추천 시스템에서 KNN 알고리즘을 활용할 수 있을까요?
  - profile, latent factors 등을 이용해 유저나 아이템을 벡터 공간에 임베딩할 수 있다면 벡터 간 유사도(거리) 측정 가능
  - 이를 바탕으로 최근접 이웃을 추출
- 유저가 10만명, 아이템이 100만개 있습니다. 이 경우 추천 시스템을 어떻게 구성하시겠습니까?
- 딥러닝을 활용한 추천 시스템의 사례를 알려주세요
- 두 추천엔진간의 성능 비교는 어떤 지표와 방법으로 할 수 있을까요? 검색엔진에서 쓰던 방법을 그대로 쓰면 될까요? 안될까요?
- Collaborative Filtering에 대해 설명한다면?
  - 한 아이템 혹은 유저와 유사한 아이템/유저를 찾아 rating을 추정
  - 단점: cold start, sparsity, popularity bias, 최근접 이웃 추출 시 컴퓨팅 리소스 소모 큼
- Cold Start의 경우엔 어떻게 추천해줘야 할까요?
  - 신규 아이템: item profile 활용해 유사한 아이템 추출
  - 신규 유저: popular item 추천, 회원가입 시 설문조사
- 고객사들은 기존 추천서비스에 대한 의문이 있습니다. 주로 매출이 실제 오르는가 하는 것인데, 이를 검증하기 위한 방법에는 어떤 것이 있을까요? 위 관점에서 우리 서비스의 성능을 고객에게 명확하게 인지시키기 위한 방법을 생각해봅시다.

##### [목차로 이동](#contents)

## 데이터베이스
- PostgreSQL의 장점은 무엇일까요?
- 인덱스는 크게 Hash 인덱스와 B+Tree 인덱스가 있습니다. 이것은 무엇일까요?
- 인덱스 Scan 방식은 무엇이 있나요?
- 인덱스 설계시 NULL값은 고려되야 할까요? 
- Nested Loop 조인은 무엇일까요?
- Windows 함수는 무엇이고 어떻게 작성할까요?
- KNN 알고리즘을 쿼리로 구현할 수 있을까요?
- MySQL에서 대량의 데이터(500만개 이상)를 Insert해야하는 경우엔 어떻게 해야할까요?
- RDB의 char와 varchar의 차이는 무엇일까요?
- 구글의 BigQuery, AWS의 Redshift는 기존 RDB와 무슨 차이가 있을까요? 왜 빠를까요?
- 쿼리의 성능을 확인하기 위해 어떤 쿼리문을 작성해야 할까요?
- MySQL이 요새 느리다는 신고가 들어왔습니다. 첫번째로 무엇을 확인하시고 조정하시겠나요?
- 동작하는 MySQL에 Alter table을 하면 안되는 이유를 설명해주세요. 그리고 대안을 설명해주세요
- 빡세게 동작하고 있는 MySQL을 백업뜨기 위해서는 어떤 방법이 필요할까요?

##### [목차로 이동](#contents)

## 데이터 시각화
- 네트워크 관계를 시각화해야 할 경우 어떻게 해야할까요?
- Tableau같은 BI Tool은 어느 경우 도입하면 좋을까요?
- "신규/재방문자별 지역별(혹은 일별) 방문자수와 구매전환율"이나 "고객등급별 최근방문일별 고객수와 평균구매금액"와 같이 4가지 이상의 정보를 시각화하는 가장 좋은 방법을 추천해주세요
- 구매에 영향을 주는 요소의 발견을 위한 관점에서, 개인에 대한 쇼핑몰 웹 활동의 시계열 데이터를 효과적으로 시각화하기 위한 방법은 무엇일까요? 표현되어야 하는 정보(feature)는 어떤 것일까요? 실제시 어떤 것이 가장 고민될까요?
- 파이차트는 왜 구릴까요? 언제 구린가요? 안구릴때는 언제인가요?
- 히스토그램의 가장 큰 문제는 무엇인가요?
- 워드클라우드는 보기엔 예쁘지만 약점이 있습니다. 어떤 약점일까요?
- 어떤 1차원값이, 데이터가 몰려있어서 직선상에 표현했을 때 보기가 쉽지 않습니다. 어떻게 해야할까요?


##### [목차로 이동](#contents)

## 대 고객 사이드
- 고객이 궁금하다고 말하는 요소가 내가 생각하기에는 중요하지 않고 다른 부분이 더 중요해 보입니다. 어떤 식으로 대화를 풀어나가야 할까요?
- 현업 카운터 파트와 자주 만나며 실패한 분석까지 같이 공유하는 경우와, 시간을 두고 멋진 결과만 공유하는 케이스에서 무엇을 선택하시겠습니까?
- 고객이 질문지 리스트를 10개를 주었습니다. 어떤 기준으로 우선순위를 정해야 할까요?
- 오프라인 데이터가 결합이 되어야 해서, 데이터의 피드백 주기가 매우 느리고 정합성도 의심되는 상황입니다. 우리가 할 수 있는 액션이나 방향 수정은 무엇일까요?
- 동시에 여러개의 A/B테스트를 돌리기엔 모수가 부족한 상황입니다. 어떻게 해야할까요?
- 고객사가 과도하게 정보성 대시보드만을 요청할 경우, 어떻게 대처해야 할까요?
- 고객사에게 위클리 리포트를 제공하고 있었는데, 금주에는 별다른 내용이 없었습니다. 어떻게 할까요?
- 카페24, 메이크샵 같은 서비스에서 데이터를 어떻게 가져오면 좋을까요?
- 기존에 같은 목적의 업무를 수행하던 조직이 있습니다. 어떻게 관계 형성을 해 나가야 할까요. 혹은 일이 되게 하기 위해서는 어떤 부분이 해소되어야 할까요.
- 인터뷰나 강의에 활용하기 위한 백데이터는 어느 수준까지 일반화 해서 사용해야 할까요?
- 고객사가 우리와 일하고 싶은데 현재는 capa가 되지 않습니다. 어떻게 대처해야 할까요?

##### [목차로 이동](#contents)

## 개인정보
- 어떤 정보들이 개인정보에 해당할까요? ID는 개인정보에 해당할까요? 이를 어기지 않는 합법적 방법으로 식별하고 싶으면 어떻게 해야할까요?
- 국내 개인 정보 보호 현황에 대한 견해는 어떠한지요? 만약 사업을 진행하는데 장애요소로 작용한다면, 이에 대한 해결 방안은 어떤 것이 있을까요?
- 제3자 쿠키는 왜 문제가 되나요?

##### [목차로 이동](#contents)
 
## Reference
- 하용호님 자료
- [남세동님 자료](https://www.facebook.com/dgtgrade/posts/1679749038750622)
- [Data Science Interview Questions & Detailed Answers](https://rpubs.com/JDAHAN/172473?lipi=urn%3Ali%3Apage%3Ad_flagship3_pulse_read%3BgFdjeopHQ5C1%2BT367egIug%3D%3D)
- [Deep Learning Interview Questions and Answers](https://www.cpuheater.com/deep-learning/deep-learning-interview-questions-and-answers/)
- [Must know questions deeplearning : 객관식 딥러닝 문제](https://www.analyticsvidhya.com/blog/2017/01/must-know-questions-deep-learning/)
- [My deep learning job interview experience sharing](https://towardsdatascience.com/my-deep-learning-job-interview-experience-sharing-4f47dd77f57d)
- [Natural Language Processing Engineer Interview Questions](https://resources.workable.com/natural-language-processing-engineer-interview-questions)

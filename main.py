import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import matplotlib

font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False  # matplotlib 에 한국어 폰트 적용

# 데이터 불러오기
data = pd.read_csv('weather_data.csv', encoding='cp949')

# 변수간 상관관계 분석
data_corr = data.iloc[:, 5:14].corr()  # 간단히 상관관계 구하기

# 기본적인 상관관계 그래프
sns.clustermap(data_corr,
               annot=True,  # 실제 값 화면에 나타내기
               cmap='RdYlBu_r',  # Red, Yellow, Blue 로 색상 표시
               vmin=-1, vmax=1  # 컬러차트 -1 ~ 1 범위로 표시
               )

# 삼각형으로 상관관계 그리기
# 그림 사이즈 지정
fig, ax = plt.subplots(figsize=(10, 10))

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(data_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
sns.heatmap(data_corr,
            cmap='RdYlBu_r',
            annot=True,  # 실제 값을 표시한다
            mask=mask,  # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},  # 컬러바 크기 절반으로 줄이기
            vmin=-1, vmax=1  # 컬러바 범위 -1 ~ 1
            )
plt.show()

# 주성분 분석 해보기

# 정규화 시키기
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(data.iloc[:, 5:14])

# 주성분 분석 (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # PCA 객체 생성 (2개)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principalComponents1', 'principalComponents2'])  # 2개의 주성분이 나타내는 값

# 주성분 분석이나 요인분석은 python 보다는 R이더 직관적이고 사용이 편리한 것 같다.

# 만든주성분으로 회귀분석 해보기 이후 지역별 날씨 변화 그래프로 그려보기



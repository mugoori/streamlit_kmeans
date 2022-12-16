import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main () :
    st.title('K-Means 클러스터링')

    # 1. csv 파일을 업로드 할 수 있다.

    st.subheader('CSV 파일을 업로드 하세요.')
    file = st.file_uploader('',type=['CSV'])
    # 2. 업로드한 csv파일을 데이터프레임으로 읽고
    if file is not None :

        df=pd.read_csv(file,index_col=0)
        df=df.dropna()
        st.dataframe(df)
    
    # 3. KMeans 클러스터링을 하기 위해, X 로 사용할 컬럼을 설정 할 수 있다.
    column = df.columns
    selected_column = st.multiselect('컬럼을 선택하세요',column)
    X = df[selected_column]

    if len(selected_column) != 0 :
        
        st.dataframe(X)
        # 4. WCSS 를 확인하기 위한, 그룹의 갯수를 정할 수 있다.
        st.subheader('WCSS를 위한 클러스터링 갯수를 선택')
        max_number = st.number_input('최대 그룹 선택', 2, 10, value=10)

        wcss = []
        for k in np.arange(1,max_number+1) :
            kmeans=KMeans(n_clusters=k, random_state=5)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_ )

        # st.write(wcss)

        # 5. 엘보우 메소드 차트를 화면에 표시

        fig1 = plt.figure()
        x = np.arange(1,max_number+1)
        plt.plot(x,wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        st.pyplot(fig1)

        # 6. 그룹핑하고 싶은 갯수를 입력
        k = st.number_input('그룹 갯수 결정', 1, max_number)

        # 7. 위에서 입력한 그룹의 갯수로 클러스터링하여 결과를 보여준다.
        kmeans=KMeans(n_clusters= k, random_state=5)
        y_pred=kmeans.fit_predict(X)
        df['Group'] = y_pred
        st.dataframe(df.sort_values('Group'))
        a = st.selectbox('그룹 별 보기',df['Group'].unique())
        st.dataframe(df.loc[df['Group'] == a])

        df.to_csv('result.csv')



if __name__ == '__main__' :
    main()
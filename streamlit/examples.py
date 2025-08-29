# # # # import streamlit as st

# # # # # st.title("This is a title")

# # # # # title = st.text_input("Movie title")
# # # # # st.write("The current movie title is", title)

# # # # st.title("AI writer")

# # # # st.button("Reset", type="primary")
# # # # if st.button("Say hello"):
# # # #     st.write("Why hello there")
# # # # else:
# # # #     st.write("Goodbye")

# # # # if st.button("Aloha", type="tertiary"):
# # # #     st.write("Ciao")

# # # import streamlit as st

# # # st.title('Streamlit text')

# # # code = '''
# # # def sample_func():
# # #     print("Sample 함수")
# # # '''
# # # st.code(code, language="python")

# # # st.text('ChatGPT 개발 교육 과정입니다.')

# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.datasets import load_iris

# # # iris = load_iris()

# # # dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
# # # dataframe['target'] = iris.target

# # # st.dataframe(dataframe, use_container_width=True)
# # # st.table(dataframe)

# # # st.metric(label="생산량", value="54000개", delta="-150개")
# # # st.metric(label="영업이익률", value="18.2%", delta="1.4%")

# # # 영업1부, 영업2부 = st.columns(2)
# # # 영업1부.metric(label="수주잔고", value="3.8억", delta="-0.5억")
# # # 영업2부.metric(label="수주잔고", value="2.5억", delta="5000천만")


# # import streamlit as st
# # import pandas as pd
# # from datetime import datetime as dt
# # import datetime

# # st.write('버튼을 눌러보세요.')

# # button = st.button('버튼')

# # if button:
# #     st.write('버튼이 눌렸습니다')

# # human = st.checkbox('사람이면 체크해주세요.')

# # if human:
# #     st.write('당신은 사람이군요!')

# # religion = st.radio(
# #     index=None,
# #     label='당신의 종교는 무엇입니까?',
# #     options=('기독교', '천주교', '불교', '기타', '무교')
# # )
     
# # if religion:
# #     st.write("당신의 종교는 *" + religion + "* 이군요!")     

# # school = st.selectbox(
# #     index=None,
# #     label='당신의 최종학력은 무엇입니까?',
# #     options=('대학원졸', '대졸', '고졸')
# # )

# # if school:
# #     st.write("당신의 최종학력은 :sparkle:" + school + ":sparkle: 이군요!")     

# # foods = st.multiselect(
# #     '당신이 가장 좋아하는 음식은 뭔가요?',
# #     ['돼지갈비', '소갈비', '스테이크', '생선회', '삼겹살', '김치찌개']
# # )

# # if len(foods) > 0:
# #     st.write(f'당신이 가장 좋아하는 음식은 {foods}입니다.')

# # bp = st.slider('혈압 범위를 지정해주세요.',
# #     6.0, 200.0, (90.0, 130.0))
# # st.write('이완기 혈압:', bp[0])
# # st.write('수축기 혈압:', bp[1])

# # birthday_time = st.slider(
# #     "당신의 출생년월일 시각을 알려주세요",
# #     min_value=dt(1950, 1, 1, 0, 0), 
# #     max_value=dt(2024, 3, 11, 12, 0),
# #     step=datetime.timedelta(hours=1),
# #     format="MM/DD/YY - HH:mm")
# # st.write(f"당신의 생년월일시는 {birthday_time}입니다.")

# # name = st.text_input(
# #     label='이름', 
# #     placeholder='당신의 이름을 입력해주세요.'
# # )
# # if name:
# #     st.write(f'안녕하세요 {name}씨!')

# # employee = st.number_input(
# #     label='당신의 회사 인원 수를 알려주세요.', 
# #     min_value=1, 
# #     max_value=300, 
# #     value=30,
# #     step=5
# # )
# # st.write(f'당신 회사의 인원수는 {employee}명입니다.')

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_iris
# import seaborn as sns

# st.title('Titanic dataset 분석')

# df = sns.load_dataset('titanic')

# g = sns.barplot(x='sex', y='survived', data=df)
# st.pyplot(g.get_figure())

# g = sns.barplot(x='sex', y='fare', data=df)
# st.pyplot(g.get_figure())

# g =sns.barplot(x='sex', y='survived', hue = 'class', data=df)
# st.pyplot(g.get_figure())

# g = sns.barplot(x='sex', y='survived', hue = 'class', order = ['female', 'male'], data=df)
# st.pyplot(g.get_figure())

# g = sns.barplot(x='sex', y='survived', hue = 'class', order = ['female', 'male'], estimator = sum, data=df)
# st.pyplot(g.get_figure())

# g = sns.barplot(x='sex', y='survived', hue = 'class', order = ['female', 'male'], palette="Blues_d", data=df)
# st.pyplot(g.get_figure())

# g = sns.barplot(x = 'sex', y = 'survived', hue = 'class', data = df)
# g.set(xlabel='Gender', ylabel='Survival Rate')
# st.pyplot(g.get_figure())

# g = sns.barplot(x = 'sex', y = 'survived', hue = 'class', data = df)
# g.set_title("Gender vs Survival Rate in Males and Females") 
# st.pyplot(g.get_figure())

# g = sns.violinplot(x ="sex", y ="age", hue ="survived", data = df, split = True)
# st.pyplot(g.get_figure())

import streamlit as st

if "count" not in st.session_state:
    st.session_state["count"] = 0

st.write(f"카운터 = {st.session_state['count']}")

button = st.button("누르세요")

if button:
    st.session_state["count"] = st.session_state["count"] + 1
    st.rerun()

# https://docs.streamlit.io/library/api-reference/performance/st.cache
# https://docs.streamlit.io/library/api-reference/performance/st.cache_resource
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sqlalchemy import create_engine
from scipy.stats import pearsonr
import requests
import time
import os

import numpy as np
import scipy.sparse
from implicit.evaluation import  *
from implicit.als import AlternatingLeastSquares as ALS
from implicit.bpr import BayesianPersonalizedRanking as BPR
from scipy.sparse import csr_matrix

#스레드 제한> 속도 보고 아래 학습량을 줄일지 이걸 없앨지 결정해야함.
os.environ['OPENBLAS_NUM_THREADS'] = '1'


# Flask 앱 생성
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# DB 연결 설정

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL 환경 변수가 설정되지 않았습니다.")

API_KEY = os.getenv("B_API_KEY")
if not API_KEY:
    raise RuntimeError("B_API_KEY 환경 변수가 설정되지 않았습니다.")


engine = create_engine(DB_URL)



try:
    with engine.connect() as connection:
        pivot_df = pd.read_sql("SELECT * FROM pivot_df", connection)
        user_pd = pd.read_sql("SELECT * FROM user_pd", connection)
        main_character_dict = pd.read_sql("SELECT * FROM main_character_dict", connection).set_index('key')['value'].to_dict()
        num_class_dict = pd.read_sql("SELECT * FROM num_class_dict", connection).set_index('key')['value'].to_dict()
    print("Neon DB에서 데이터 로드 성공")
except Exception as e:
    print("Neon DB 연결 실패:", e)


existing_user_pd = user_pd.copy()
existing_pivot_df = pivot_df.copy()
existing_main_character_dict = main_character_dict.copy()

main_character=list(user_pd['main_character'].unique())
user_pd = user_pd.query("class_type not in ['스페셜리스트','헌터(남)','전사(남)','무도가(여)','무도가(남)','전사(여)','헌터(여)','마법사','데모닉','암살자']")
user_pd=user_pd.reset_index(drop=True)

user_pd_num=pd.DataFrame()
user_pd_num['main_character_num'], main_character_mapping = pd.factorize(user_pd['main_character'])
user_pd_num['class_num'], class_mapping = pd.factorize(user_pd['class_type'])
bins = [0, 1540, 1580, 1620, 1660, 1680, 1700, float('inf')]
labels = [1, 2, 3, 4, 5, 6, 7]
try:
    user_pd['level'] = user_pd['level'].str.replace(',', '').astype(float)
    user_pd=user_pd.reset_index(drop=True)
except:
   pass
   
user_pd_num['level_category'] = pd.cut(user_pd['level'], bins=bins, labels=labels, right=False)
user_pd_num['level_category']=user_pd_num['level_category'].astype(int)


def add_new_data_to_db(new_user_pd, new_pivot_df,new_main_character_dict):

    global existing_user_pd,existing_pivot_df,existing_main_character_dict

    additional_user_pd = new_user_pd[~new_user_pd['main_character'].isin(existing_user_pd['main_character'])]
    additional_pivot_df = new_pivot_df[~new_pivot_df.index.isin(existing_pivot_df.index)]
    additional_main_character_dict = {k: v for k, v in new_main_character_dict.items() if k not in existing_main_character_dict}


    if not additional_user_pd.empty:
        additional_user_pd.to_sql('user_pd', engine, if_exists='append', index=False)
        print(f"{len(additional_user_pd)}개의 새로운 user_pd 데이터가 추가되었습니다.")

    if not additional_pivot_df.empty:
        additional_pivot_df.to_sql('pivot_df', engine, if_exists='append', index=False)
        print(f"{len(additional_pivot_df)}개의 새로운 pivot_df 데이터가 추가되었습니다.")

    if additional_main_character_dict:
        additional_df = pd.DataFrame(list(additional_main_character_dict.items()), columns=['key', 'value'])
        additional_df.to_sql('main_character_dict', engine, if_exists='append', index=False)
        print(f"{len(additional_main_character_dict)}개의 새로운 딕셔너리 항목이 추가되었습니다.")
        
    existing_user_pd = new_user_pd.copy()
    existing_pivot_df= new_pivot_df.copy()
    existing_main_character_dict=new_main_character_dict.copy()




#피어슨 기반 직업추천
def reco_pear(user_name,api):
  
  global pivot_df, user_pd, main_character_dict, user_pd_num

  start_time = time.time()

  api_key=api
  headers={
  'accept' : 'application/json',
  'authorization' : f'Bearer {api_key}',
  }


  if user_name not in  main_character_dict:
    
    try:
        url=f'https://developer-lostark.game.onstove.com/characters/{user_name}/siblings'

        expedtion_data=requests.get(url,headers=headers).json()

        if len(expedtion_data)==0:
            return(f"{user_name}을 검색할 수 없거나, api key 를 다시 확인해주세요.")
    
        temp={
            'class_type':[],
            'level':[],
            'CharacterName':[]
            }
        for j in expedtion_data:
            temp['CharacterName'].append(j['CharacterName'])
            temp['class_type'].append(j['CharacterClassName'])
            temp['level'].append(j['ItemAvgLevel'])
    except:
        return(f"{user_name}을 검색할 수 없거나, api key를 다시 확인해주세용.")
    temp_user=pd.DataFrame(temp)
    temp_user['level'] = temp_user['level'].str.replace(',', '').astype(float)
    temp_user=temp_user.reset_index(drop=True)
    if temp_user['level'].max() < 1640:
        return "1640 하나는 찍고와라 애송이"
    temp_user['main_character']=temp_user['CharacterName'].iloc[temp_user['level'].idxmax()]
    del temp_user['CharacterName']
    user_name=temp_user['main_character'][0]
    if user_name not in  main_character_dict:
      user_pd=pd.concat([user_pd,temp_user],ignore_index=True)
      user_pd = user_pd.query("class_type not in ['스페셜리스트','헌터(남)','전사(남)','무도가(여)','무도가(남)','전사(여)','헌터(여)','마법사','데모닉','암살자']")
      user_pd=user_pd.reset_index(drop=True)
      main_character=list(user_pd['main_character'].unique())
      user_pd_num=pd.DataFrame()
      user_pd_num['main_character_num'], main_character_mapping = pd.factorize(user_pd['main_character'])
      user_pd_num['class_num'], class_mapping = pd.factorize(user_pd['class_type'])
      user_pd_num['level_category'] = pd.cut(user_pd['level'], bins=bins, labels=labels, right=False)
      user_pd_num['level_category']=user_pd_num['level_category'].astype(int)
      main_character_dict = {name: num for num, name in enumerate(main_character_mapping)}
      user_pd_num=user_pd_num[user_pd_num['level_category']>=4]
      pivot_df = user_pd_num.pivot_table(
        index='main_character_num',    # 유저 ID
        columns='class_num',           # 직업 번호
        values='level_category',       # 레벨 카테고리
        aggfunc='max',                # 중복값이 있을 경우 최댓값을 구함
        fill_value=0                   # 값이 없는 경우 0으로 채움
        )
      
      add_new_data_to_db(user_pd,pivot_df,main_character_dict)
      print("데이터가 갱신 되었습니다. 직업추천을 시작합니다.")

    else:
      print("기존 데이터에 존재하는 유저입니다. 직업 추천을 시작합니다1.")
  else:
    print("기존 데이터에 존재하는 유저입니다. 직업 추천을 시작합니다2.")

  my_character_id = main_character_dict[user_name]
  my_vector = pivot_df.loc[my_character_id]

  # 다른 유저들과 피어슨 유사도 계산
  similarities = {}

  for other_id in pivot_df.index:
      if other_id != my_character_id:
          other_vector = pivot_df.loc[other_id]
          correlation, _ = pearsonr(my_vector, other_vector)
          similarities[other_id] = correlation

  # 유사도 순으로 정렬
  sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

  # 나와 유사한 상위 N명의 유저 ID 추출
  top_n_similar_users = [user[0] for user in sorted_similarities[:13]]  # N명을 설정

  # 이 유저들이 주로 사용하는 직업 추출
  top_n_users_df = user_pd_num[user_pd_num['main_character_num'].isin(top_n_similar_users)]

  # 내가 아직 사용하지 않은 직업 필터링
  my_classes = user_pd_num[user_pd_num['main_character_num'] == my_character_id]['class_num'].unique()
  recommended_classes = top_n_users_df[~top_n_users_df['class_num'].isin(my_classes)]

  # 추천 직업의 가중치에 따른 평균 레벨 카테고리로 정렬
  recommendations = recommended_classes.groupby('class_num')['level_category'].mean().sort_values(ascending=False)

  end_time = time.time()

  print(f" {round(end_time - start_time,2)} 초 소요")

  return [num_class_dict[i] for i in recommendations.index]


#als 기반 직업추천
def reco_als(user_name,api):
  
  global pivot_df, user_pd, main_character_dict, user_pd_num

  start_time = time.time()

  api_key=api
  headers={
  'accept' : 'application/json',
  'authorization' : f'Bearer {api_key}',
  }


  if user_name not in  main_character_dict:
    
    try:
        url=f'https://developer-lostark.game.onstove.com/characters/{user_name}/siblings'

        expedtion_data=requests.get(url,headers=headers).json()

        if len(expedtion_data)==0:
            return(f"{user_name}을 검색할 수 없거나, api key 를 다시 확인해주세요.")
    
        temp={
            'class_type':[],
            'level':[],
            'CharacterName':[]
            }
        for j in expedtion_data:
            temp['CharacterName'].append(j['CharacterName'])
            temp['class_type'].append(j['CharacterClassName'])
            temp['level'].append(j['ItemAvgLevel'])
    except:
        return(f"{user_name}을 검색할 수 없거나, api key를 다시 확인해주세용.")
    temp_user=pd.DataFrame(temp)
    temp_user['level'] = temp_user['level'].str.replace(',', '').astype(float)
    temp_user=temp_user.reset_index(drop=True)
    if temp_user['level'].max() < 1640:
        return "1640 하나는 찍고와라 애송이"
    temp_user['main_character']=temp_user['CharacterName'].iloc[temp_user['level'].idxmax()]
    del temp_user['CharacterName']
    user_name=temp_user['main_character'][0]
    if user_name not in  main_character_dict:
      user_pd=pd.concat([user_pd,temp_user],ignore_index=True)
      user_pd = user_pd.query("class_type not in ['스페셜리스트','헌터(남)','전사(남)','무도가(여)','무도가(남)','전사(여)','헌터(여)','마법사','데모닉','암살자']")
      user_pd=user_pd.reset_index(drop=True)
      main_character=list(user_pd['main_character'].unique())
      user_pd_num=pd.DataFrame()
      user_pd_num['main_character_num'], main_character_mapping = pd.factorize(user_pd['main_character'])
      user_pd_num['class_num'], class_mapping = pd.factorize(user_pd['class_type'])
      user_pd_num['level_category'] = pd.cut(user_pd['level'], bins=bins, labels=labels, right=False)
      user_pd_num['level_category']=user_pd_num['level_category'].astype(int)
      main_character_dict = {name: num for num, name in enumerate(main_character_mapping)}
      user_pd_num=user_pd_num[user_pd_num['level_category']>=4]
      pivot_df = user_pd_num.pivot_table(
        index='main_character_num',    # 유저 ID
        columns='class_num',           # 직업 번호
        values='level_category',       # 레벨 카테고리
        aggfunc='max',                # 중복값이 있을 경우 최댓값을 구함
        fill_value=0                   # 값이 없는 경우 0으로 채움
        )
      
      add_new_data_to_db(user_pd,pivot_df,main_character_dict)
      print("데이터가 갱신 되었습니다. 직업추천을 시작합니다.")

    else:
      print("기존 데이터에 존재하는 유저입니다. 직업 추천을 시작합니다1.")
  else:
    print("기존 데이터에 존재하는 유저입니다. 직업 추천을 시작합니다2.")

  user_id=int(main_character_dict[user_name])

  #유저 정보
  a=pivot_df.iloc[user_id][pivot_df.iloc[user_id] != 0].sort_values(ascending=False).index.astype(int) 

  #if len(a)==1:
    #return reco_pear(user_name,api)


  rating_matrix = csr_matrix(pivot_df)
  als_model = ALS(factors=60, regularization=0.01, iterations = 20)
  als_model.fit(rating_matrix.T)
  als_model.user_factors
  als=np.dot(als_model.item_factors,als_model.user_factors.T)


  
  #추천정보
  b=np.argsort(als[user_id])[::-1][:20]

  end_time = time.time()

  print(f" {round(end_time - start_time,2)} 초 소요")

  return [num_class_dict[i] for i in b if i not in a]

def reco(user_name,api):
    time_now=int(str(time.time())[-1])%2==0
    if time_now%2==0:
        return reco_pear(user_name,api), 'pear'
    else:
        return reco_als(user_name,api), 'als'



@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_name = data.get('user_name')
        api_key = data.get('api_key','')

        if not api_key:
            api_key = os.getenv("B_API_KEY", "")  


        if not user_name:
            return jsonify({"error": "Missing 'user_name' "}), 400

        recommendations , algorithm_used = reco(user_name, api_key)
        

        if isinstance(recommendations, str):
            return jsonify({"message": recommendations }), 200     
        elif isinstance(recommendations, list):
            return jsonify({"recommendations": recommendations, "algorithm": algorithm_used}), 200
        else:
            return jsonify({"error": "Unexpected response from reco function"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

from sqlalchemy.sql import text


@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        feedback_type = data.get('feedback')  # 'good' 또는 'bad'
        algorithm = data.get('algorithm')    # 'pear' 또는 'als'

        if feedback_type not in ['good', 'bad']:
            return jsonify({"error": "Invalid feedback type"}), 400

        # 테이블 이름 결정
        table_name = f"feedback_{algorithm}"

        # 피드백 값 업데이트
        with engine.begin() as connection:
            query = text(f"UPDATE {table_name} SET {feedback_type} = {feedback_type} + 1")
            connection.execute(query)

        return jsonify({"message": f"{feedback_type.capitalize()} 피드백이 저장되었습니다."}), 200
    except Exception as e:
        print(f"오류 발생: {str(e)}")  # 오류 메시지 출력
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=False)

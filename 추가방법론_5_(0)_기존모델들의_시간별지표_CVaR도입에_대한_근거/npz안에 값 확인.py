import numpy as np

# 1. 파일 불러오기
data = np.load('proposed_ar_zhang_combined_diagnostics.npz')

# 2. 파일 안에 들어있는 배열들의 이름(키값) 확인하기
print(data.files)

# 3. 특정 배열 데이터 가져오기 (예: 키 이름이 'arr_0'인 경우)
#array_data = data['arr_0']
#print(array_data)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 엑셀 파일 경로
excel_file_path = './카드내역.xlsx'
# 사용자 선택
name = input("이름을 입력하세요 : ")

# 엑셀 파일 불러오기(입력:시간, 출력:대분류)
df = pd.read_excel(excel_file_path, sheet_name = name, usecols = ['시간', '대분류'])
# '열 이름'에서 12~20으로 시작하는 데이터를 필터링(12~21시 까지의 데이터만 사용)
dataset = df[df['시간'].astype(str).str.startswith(('12', '13', '14', '15', '16', '17', '18', '19', '20'))]
# 인덱스 새로 설정
dataset = dataset.reset_index(drop=True)
# '열 이름'에서 12~20으로 시작하는 데이터를 12~20으로 변경
dataset.loc[dataset['시간'].astype(str).str.startswith(('12', '13', '14', '15', '16', '17', '18', '19', '20')), '시간'] = dataset['시간'].astype(str).str[:2].astype(int)
print(dataset)

# 출력 카테고리(9개) => ['교통' '문화/여가' '뷰티/미용' '생활' '식비' '온라인쇼핑' '의료/건강' '카페/간식' '패션/쇼핑']
unique_values = dataset['대분류'].unique()
# 출력 카테고리 각각의 개수
value_counts = dataset['대분류'].value_counts()

print(value_counts)

# 데이터 분리
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.size, X_test.size)

# 모델 학습
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# 모델 검증(정확도)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
# 모델 검증(Confusion Matrix)
conf_mat = confusion_matrix(y_test, y_pred)

# 카테고리 종류 (9개)
print(logreg.classes_) 

# 시간별 카테고리 매칭
arr = [0] * 9
for i in range(12, 21):
    # print(str(i)+"시 : ", logreg.predict_proba([[i]]))
    index = 0
    for j in range (9):
        # print(logreg.predict_proba([[i]])[0][j])
        if logreg.predict_proba([[i]])[0][j] >= logreg.predict_proba([[i]])[0][index]:
            index = j
    arr[i-12] = index
print(arr)
# 시간별 카테고리 결과
for i in range (9):
    arr[i] = logreg.classes_[arr[i]]
print(arr)

print("-----------------------------------------")
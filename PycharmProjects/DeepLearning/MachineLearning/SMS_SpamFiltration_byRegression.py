import pickle
from sklearn.linear_model import LogisticRegression

with open("processed.pickle","rb") as file_handle:
    vocabulary,features,labels=pickle.load(file_handle)

# 학습 - 평가 데이터 나누기

total_number=len(labels)
middle_index=total_number//2

train_features=features[:middle_index,:]
train_labels=labels[:middle_index]

test_features=features[middle_index:,:]
test_labels=labels[middle_index:]

classifier=LogisticRegression()
classifier.fit(train_features,train_labels)

#LogisticRegression의 score함수로 정확도 측정
print("train accuracy : %4.4f"% classifier.score(train_features,train_labels))
print("test accuracy : %4.4f"% classifier.score(test_features,test_labels))

# 어떤 항목이 판별에 영향을 많이 줬는지 알아보기
weights=classifier.coef_[0,:] # coef_ : 계수값 [n_targets:n_features]
pairs=[]
for index, value in enumerate(weights):
    pairs.append((abs(value),vocabulary[index]))
    
# value값 기준으로 정렬
# lambda x: x[0] -> tuple(x[0],x[1]) x[0]은 value값, x[1]은 vocabulary 순
pairs.sort(key=lambda x:x[0], reverse=True)


# value값 상위 20개 출력
for pair in pairs[:20]:
    print('score %4.4f word : %s'% pair)

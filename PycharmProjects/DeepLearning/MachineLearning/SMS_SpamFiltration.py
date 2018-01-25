import numpy as np

vocabulary={}

with open('SMSSpamCollection',encoding="utf-8") as file_handle:
    for line in file_handle:
        #한줄씩 할당
        splits=line.split()
        label=splits[0]
        text=splits[1:]


        for word in text:
            lower=word.lower()
            if not lower in vocabulary:
                vocabulary[lower]=len(vocabulary)

print(vocabulary)


features=[]

with open("SMSSpamCollection", encoding="utf-8") as file_handle:
    for line in file_handle:
        splits=line.split()
        feature=np.zeros(len(vocabulary))
        text=splits[1:]

        for word in text:
            lower=word.lower()

            feature[vocabulary[lower]]+=1

        feature=feature/sum(feature)
        features.append(feature)
print(features)

# 대부분 배열값이 0으로
# sparse matrix (희박 행렬) 저장

labels=[]
with open("SMSSpamCollection",encoding="utf-8") as file_handle:
    for line in file_handle:
        splits=line.split()
        label=splits[0]

        if label=="spam":
            labels.append(1)
        else:
            labels.append(0)
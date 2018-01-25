import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

spam_header= "spam\t"
no_spam_header="ham\t"
documents=[]
labels=[]

with open("SMSSpamCollection") as file_handle:
    for line in file_handle:
        # 시작 부분에 spam/ham 여부가 있으므로 두가지로 나눔
        if line.startswith(spam_header):
            labels.append(1)
            # documents에 spam 글자를 제외한 내용부분 추가
            documents.append(line[len(spam_header):])
        elif line.startswith(no_spam_header):
            labels.append(0)
            # documents에 ham 글자를 제외한 내용부분 추가
            documents.append(line[len(no_spam_header):])

vectorizer=CountVectorizer()
term_counts=vectorizer.fit_transform(documents) # 단어 횟수 세기
vocabulary=vectorizer.get_feature_names()

# 단어 횟수 feature에서 단어 빈도 feature 로
# tf-idf에서 idf를 생성하지 않을 시 단어 빈도가 만들어짐.
tf_transformer=TfidfTransformer(use_idf=False).fit(term_counts)
features=tf_transformer.transform(term_counts)

# pickle을 통해 파일 저장
with open("processed.pickle","wb") as file_handle:
    pickle.dump((vocabulary,features,labels),file_handle)
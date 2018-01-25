from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

spam_header="spam\t"
no_spam_header='ham\t'
documents=[]

with open("SMSSPamCollection", encoding="UTf-8") as file_handle:
    for line in file_handle:
        if line.startswith(spam_header):
            documents.append(line[len(spam_header):])
        elif line.startswith(no_spam_header):
            documents.append(line[len(no_spam_header):])

# LDA는 단어 빈도가 아닌 단어의 갯수가 더 잘 동작하므로 CountVectorizer사용하고
# TfidfTransfomer사용 안함(단어 빈도 생성)

vectorizer=CountVectorizer(stop_words="english",max_features=2000)
term_count=vectorizer.fit_transform(documents)
vocabulary=vectorizer.get_feature_names()

# 토픽 모델 학습
# n_topics는 Deprecate 될 예정 -> n_components로 변경
# n_components:문서에서 몇개의 토픽을 뽑을지 결정
topic_model=LatentDirichletAllocation(n_components=10)
topic_model.fit(term_count)

# components_ : 토픽 모델 결과
topics=topic_model.components_

for topic_id, weights in enumerate(topics):
    print("topic %d"% topic_id, end=" : ")
    # topic_id : 토픽 인덱스

    pairs=[]

    for term_id,value in enumerate(weights):
        pairs.append((abs(value),vocabulary[term_id]))

    pairs.sort(key=lambda x:x[0], reverse=True)
    for pair in pairs[:10]:
        # pair[0] : value값 pair[1] : 단어
        print(pair[1],end=",")
    print()
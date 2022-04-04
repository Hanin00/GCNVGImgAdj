import sys


testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
label = label[:1000]  # 빈출 100 단어 만 사용

print(len(label))
print(label[:-20])
print(label[:20])
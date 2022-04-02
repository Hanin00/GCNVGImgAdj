import util as ut
import YEmbedding as yed

'''
    adj = cluster 값이 동일한 img인 경우 인접
    featurematrix : 각 이미지 별



'''




def print_hi(name):
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    startId = 1
    endId = 1000
    xlxspath = './data/image_regions.xlsx'
    #Y - image, cluser 몇 번인지~
    embedding_clustering = yed.YEmbedding(xlxspath)
    idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
    label = idCluster['cluster']  #<class 'pandas.core.series.Series'>

    print(label)

    ''' frequency obj - 1x100'''
    #freObj = ut.prequency_feature(1, 1000)
    testFile = open('freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
    readFile = testFile.readline()
    list = (readFile[1:-1].replace("'",'')).split(',')



    adjMatrix = ut.create_adjMatrix(clusterList=label[1])
    # featuremap =ut.featuremap(startId,endId,freObj)

print('freObj : ',freObj)
print('adjMatrix : ', adjMatrix)
# print('featuremap : ', featuremap)

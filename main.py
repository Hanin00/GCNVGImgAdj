import util as ut
import YEmbedding as yed


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
    label = idCluster['cluster']

    print(label)
    freObj = ut.prequency_feature(1, 1000)
    adjMatrix = ut.create_adjMatrix(clusterList=label[1])
    # featuremap =ut.featuremap(startId,endId,freObj)

print('freObj : ',freObj)
print('adjMatrix : ', adjMatrix)
# print('featuremap : ', featuremap)

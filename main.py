import util as ut



def print_hi(name):


    return 0



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    startId = 1
    endId = 1000
    idList = []

    for i in range(endId):
        idList.append(i + 1)


    freObj = ut.prequency_feature(1,1000)
    adjMatrix = ut.create_adjMatrix(idList)
    featuremap =ut.featuremap(startId,endId,freObj)

    print('freObj : ',freObj)
    print('adjMatrix : ', adjMatrix)
    print('featuremap : ', featuremap)
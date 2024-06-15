import cv2
import os

regulateMat = [
    [-11.41, 15.59, 26.59, 15.59, -12.41, -31.41, 15.59, 21.59, 31.59, -12.41, -7.41, 27.59, 12.59, 15.59, 15.59, 29.59, 35.59, 33.59, 31.59, 11.59, 30.59, 11.59, -24.41, 31.59, 11.59, 15.59, 15.59, -15.41, 11.59, 33.59, -9.41, 8.59, 7.59, 31.59, 14.59, 22.59, 33.59, 8.59, 31.59, 31.59, 32.59, 32.59, 31.59, 27.59, 32.59, 33.59, 28.59, 10.59, 8.59, 35.59, 31.59, 23.59, 14.59, 15.59, 11.59, 15.59, 31.59, 11.59, 35.59, 12.59, 13.59, 31.59, 31.59, 15.59, 35.59],
    [-0.41, 3.59, -3.41, -30.41, -31.41, -31.41, -10.41, 11.59, -14.41, -31.41, -31.41, 15.59, -31.41, -3.41, 14.59, 15.59, 9.59, 15.59, 15.59, 7.59, 15.59, 15.59, -31.41, -31.41, -0.41, 6.59, -31.41, -7.41, 15.59, -31.41, 5.59, 15.59, -31.41, -8.41, -31.41, 15.59, -12.41, -31.41, -18.41, 15.59, 7.59, 15.59, -31.41, -31.41, 15.59, -21.41, 15.59, 11.59, 15.59, -5.41, -31.41, -12.41, 14.59, 14.59, 13.59, -0.41, -24.41, -0.41, -16.41, -31.41, 16.59, -31.41, 5.59, -31.41, 15.59],
    [-15.41, -31.41, -1.41, 13.59, 12.59, -1.41, 10.59, -16.41, -16.41, 3.59, -31.41, -4.41, -28.41, -22.41, -12.41, -18.41, -5.41, 15.59, -5.41, 1.59, -8.41, -31.41, -24.41, -24.41, 15.59, -5.41, 2.59, -24.41, -9.41, -31.41, 15.59, 4.59, -31.41, 3.59, -30.41, -9.41, -28.41, 15.59, -26.41, -6.41, -8.41, 10.59, -0.41, 15.59, 17.59, -31.41, 13.59, 13.59, 0.59, 1.59, -31.41, 15.59, 8.59, 11.59, -31.41, 13.59, -31.41, 10.59, -0.41, 1.59, 14.59, 11.59, -26.41, -12.41, -30.41],
    [-20.41, -31.41, -3.41, -18.41, -31.41, -14.41, -16.41, -10.41, -6.41, -31.41, -28.41, -4.41, -3.41, -12.41, -10.41, -1.41, 7.59, 13.59, -16.41, -0.41, 1.59, -20.41, -12.41, 2.59, 19.59, -0.41, 6.59, -0.41, -9.41, -12.41, -18.41, -14.41, -5.41, 29.59, -8.41, 11.59, -0.41, 1.59, -6.41, -3.41, 23.59, -6.41, -4.41, 0.59, -5.41, 6.59, 5.59, -4.41, -12.41, 15.59, -13.41, 7.59, -4.41, -13.41, -22.41, -2.41, -16.41, 4.59, 0.59, -17.41, -20.41, -5.41, -16.41, 5.59, 8.59]
]  # domain x category, regulate numbers

def detect_edges(image_path):
    image = cv2.imread(image_path, 0)
    edges = cv2.Canny(image, 100, 200)
    return edges

def get_avg_numb(rootDir):
    sumOfFileNumb = 0
    domainList = os.listdir(rootDir)
    for domainIter in domainList:
        domainDir = os.path.join(rootDir, domainIter)
        categoryList = os.listdir(domainDir)
        for categoryIter in categoryList:
            categoryDir = os.path.join(domainDir, categoryIter)
            if not os.path.exists(categoryDir):
                    break
            sumOfFileNumb += len(os.listdir(categoryDir))
    return int(sumOfFileNumb/(65*4))

def gen_regulate_table(rootDir):
    avgNumb = get_avg_numb(rootDir)
    domainList = os.listdir(rootDir)
    for domainIdx, domainIter in enumerate(domainList):
        domainDir = os.path.join(rootDir, domainIter)
        categoryList = os.listdir(domainDir)
        for categoryIdx, categoryIter in enumerate(categoryList):
            categoryDir = os.path.join(domainDir, categoryIter)
            if not os.path.exists(categoryDir):
                    break
            regulateMat[domainIdx][categoryIdx] = avgNumb - len(os.listdir(categoryDir))
            
    return 

def remove_edge_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("edge_"):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)

def main(rootDir):
    gen_regulate_table(rootDir)
    domainList = os.listdir(rootDir)
    for domainIdx, domainIter in enumerate(domainList):
        domainDir = os.path.join(rootDir, domainIter)
        categoryList = os.listdir(domainDir)
        for categoryIdx, categoryIter in enumerate(categoryList):
            dummyNum = int(regulateMat[domainIdx][categoryIdx])
            if dummyNum > 0:
                categoryDir = os.path.join(domainDir, categoryIter)
                if not os.path.exists(categoryDir):
                    break
                preFileNumb = len(os.listdir(categoryDir))
                cnt = 0
                for fileIdx in range(min(dummyNum, preFileNumb)):
                    curFiles = os.listdir(categoryDir)
                    curImgDir = os.path.join(categoryDir, curFiles[fileIdx])
                    saveImgDir = os.path.join(categoryDir, 'edge_' + str(cnt) + '.jpg')
                    cnt += 1
                    cv2.imwrite(saveImgDir, detect_edges(curImgDir))
                
if __name__ == "__main__":
    rootDir = 'trainDataset'
    #remove_edge_files(rootDir)
    main(rootDir)

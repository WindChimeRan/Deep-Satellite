import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

TRAIN_X_ADDRESS = "/home/ryan/Desktop/sample.png"
TRAIN_Y_ADDRESS = "/home/ryan/Desktop/sam.png"
class makeData:

    def __init__(self):
        self.ix = -1
        self.iy = -1
        self.nodex = []
        self.nodey = []
        self.img = cv2.imread(TRAIN_X_ADDRESS)
        #self.window()
    
    def draw(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            self.nodex.append(x)
            self.nodey.append(y)
        if event==cv2.EVENT_RBUTTONDOWN:
            node = np.column_stack((self.nodex,self.nodey))
            self.nodex=[]
            self.nodey=[]
            cv2.fillPoly(self.img,[node],(0,0,0))

    def window(self):
        cv2.namedWindow('image',cv2.WINDOW_NORMAL) 
        cv2.setMouseCallback('image',self.draw)
        while(1): 
            cv2.imshow('image',self.img) 
            k = cv2.waitKey(0)&0xFF
            ## s to save
            if k==115:
                #cv2.imwrite("/home/ryan/Desktop/sam.png",img)
                GrayImage=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)  
                ret,thresh=cv2.threshold(GrayImage,0,255,cv2.THRESH_BINARY)
                cv2.imwrite(TRAIN_Y_ADDRESS,thresh)
            ## Esc to quit
            if k==27:
                break
        cv2.destroyAllWindows()
    def to01(self):
        im=Image.open(TRAIN_Y_ADDRESS)
        temp = np.array(im)
        mask = temp == 0
        return np.ones(temp.shape)*mask
if __name__ == "__main__":
    a = makeData()
    a.window()
    a.to01()
    


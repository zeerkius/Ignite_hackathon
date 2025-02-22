import pandas

import pandas as pd
import sys

# laoding data and creating regions

data_frame = pd.read_csv(r"C:\Users\agboo\Downloads\Ignite_Chev.csv")

south = "TX ,OK, AR, LA, KY, TN, MS, AL, WV, VA, NC, SC, GA, FL, MD, DE , FG"

west = "WA, OR, CA, NV, ID, MT, WY, UT, CO, AK, HI , FP"

midwest = "ND, SD, NE, KS, MN, IA, MO, WI, IL, IN, MI, OH"

northeast = "ME, NH, VT, MA, RI, CT, NY, PA, NJ"


southarrnan = []

southarr = []

#/////

westarrnan = []

westarr = []

#////

midwestarrnan = []

midwestarr = []


#///


northeastarrnan = []

northeastarr = []



#//////

# preprocessing

prod_to_class = [
[12,7],
[12,8],[13,8.5],
[15,9],
[20,10],
[25,11],
[30,12],
[40,13],
[50,14],
[100,14.5],
[100,15],
[200,16],
[400,17],
[800,18],
[1600,19],
[3200,20],
[6400,21],
[12800,22],[14000,23]]


clsnums = [row[1] for row in prod_to_class]
prods = [row[0] for row in prod_to_class]

test_data = []


for val in data_frame.ClassNumber:
    if val <= min(clsnums):
        test_data.append(12)
    else:
        if val in clsnums:
            test_data.append(prods[clsnums.index(val)])


testsouth = []
testwest = []
testne = []
testmidwest = []

index = 0

for val in data_frame.State:
    if val in south:
        southarr.append(data_frame.iloc[index][1:5].tolist()) # oil
        southarrnan.append(data_frame.iloc[index][1:3].tolist() + data_frame.iloc[0][5:7].tolist()) # natural
        testsouth.append(test_data[index])       
    if val in west:
        westarr.append(data_frame.iloc[index][1:5].tolist())
        westarrnan.append(data_frame.iloc[index][1:3].tolist() + data_frame.iloc[0][5:7].tolist())
        testwest.append(test_data[index])
    if val in midwest:
        midwestarr.append(data_frame.iloc[index][1:5].tolist())
        midwestarrnan.append(data_frame.iloc[index][1:3].tolist() + data_frame.iloc[0][5:7].tolist())
        testne.append(test_data[index]) 
    if val in northeast:
        northeastarr.append(data_frame.iloc[index][1:5].tolist())
        northeastarrnan.append(data_frame.iloc[index][1:3].tolist() + data_frame.iloc[0][5:7].tolist())
        testmidwest.append(test_data[index]) 
    index += 1

# Creating Model

class LinearRegressor:
    def __init__(self,fit_on,w1 = 0.5,w2 = 0.5,w3 = 0.5 , w4 = 0.5,data = list()):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.data = data
        self.alpha = 0.00001
        self.fit_on = fit_on
    def guess(self,x1,x2,x3,x4):
        return (x1 * self.w1) + (x2 * self.w2) + (x3 * self.w3) + (x4 * self.w4)       
    def SSE(self,y,yprime):
        error = (y - yprime) ** 2
        #print(" Current Error Squared " + str(error))
        return error
    def update(self,w1,w2,w3,w4):
        p = self.fit_on
        self.__init__(fit_on = p ,w1 = w1,w2 = w2 , w3 = w3 , w4 = w4) 
    def deltagradw1(self,y,x1,x2,x3,x4):
        transpose = (x1 * self.w1) + (x2 * self.w2) + (x3 * self.w3) + (x4 * self.w4)
        neg = -1
        var = x1
        delta = ((y - transpose) * (neg) * var)
        return delta
                        
    def deltagradw2(self,y,x1,x2,x3,x4):
        transpose = (x1 * self.w1) + (x2 * self.w2) + (x3 * self.w3) + (x4 * self.w4)
        neg = -1
        var = x2
        delta = ((y - transpose) * (neg) * var)
        return delta
        
    def deltagradw3(self,y,x1,x2,x3,x4):
        transpose = (x1 * self.w1) + (x2 * self.w2) + (x3 * self.w3) + (x4 * self.w4)
        neg = -1
        var = x3
        delta = ((y - transpose) * (neg) * var)
        return delta

    def deltagradw4(self,y,x1,x2,x3,x4):
        transpose = (x1 * self.w1) + (x2 * self.w2) + (x3 * self.w3) + (x4 * self.w4)
        neg = -1
        var = x4
        delta = ((y - transpose) * (neg) * var)
        return delta

    def fit(self): # 20000 , 1000 epochs
        sumdw1 = 0
        sumdw2 = 0
        sumdw3 = 0
        sumdw4 = 0
        counter = 0
        to20 = 0 
        for x1 , x2 , x3 , x4 in self.data:
            print(sumdw1)
            ground_truth = self.fit_on[counter]
            prediction = self.guess(x1 = x1,x2 = x2,x3 = x3,x4 = x4)
            if self.SSE(ground_truth,prediction) <= 4:
                print([self.w1,self.w2,self.w3,self.w4])          
            elif to20 == 20:
                to20 = 0
                neww1 = self.w1 + self.alpha * sumdw1
                neww2 = self.w2 + self.alpha * sumdw2
                neww3 = self.w3 + self.alpha * sumdw3
                neww4 = self.w4 + self.alpha * sumdw4
                self.update(w1 = neww1, w2 = neww2,w3 = neww3,w4 = neww4)
                sumdw1 = 0
                sumdw2 = 0
                sumdw3 = 0
                sumdw4 = 0   
            else:                
                sumdw1 += self.deltagradw1(test_data[counter],x1,x2,x3,x4)
                sumdw2 += self.deltagradw2(test_data[counter],x1,x2,x3,x4)
                sumdw3 += self.deltagradw3(test_data[counter],x1,x2,x3,x4)
                sumdw4 += self.deltagradw4(test_data[counter],x1,x2,x3,x4)
          
            to20 += 1
            counter += 1
            

model_south = LinearRegressor(data = southarr , fit_on = testsouth)

print(model_south.fit())



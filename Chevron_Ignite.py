

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
        southarr.append(data_frame.iloc[index][2:5].tolist()) # oil
        southarrnan.append(data_frame.iloc[index][2:3].tolist() + data_frame.iloc[index][5:7].tolist()) # natural
        testsouth.append(test_data[index])     #   
    if val in west:
        westarr.append(data_frame.iloc[index][2:5].tolist())
        westarrnan.append(data_frame.iloc[index][2:3].tolist() + data_frame.iloc[index][5:7].tolist())
        testwest.append(test_data[index]) #
    if val in midwest:
        midwestarr.append(data_frame.iloc[index][2:5].tolist())
        midwestarrnan.append(data_frame.iloc[index][2:3].tolist() + data_frame.iloc[index][5:7].tolist())
        testmidwest.append(test_data[index])  # 
    if val in northeast:
        northeastarr.append(data_frame.iloc[index][2:5].tolist())
        northeastarrnan.append(data_frame.iloc[index][2:3].tolist() + data_frame.iloc[index][5:7].tolist())
        testne.append(test_data[index])  #
    index += 1

# Creating Model



class LinearRegressor:
    # linear regresssion model using SSE for loss
    # batchsize intialized to 20 , however user customizable
    # learning rate is alos customizable 
    def __init__(self, X = None , Y = None , batch_size = 20):
    
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
          
    def SSE(self,y , groundtruth):
        error = (y - groundtruth) ** 2
        return error
    def SSE_repr(self,y , groundtruth):
        error = (y - groundtruth) ** 2
        print("Current Model Error " + str(error))
        
    def delta_bias(self,x1,w1,x2,w2,x3,w3,y,b):
        y_prime = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
        delta = (y - y_prime) * (-1)
        return delta
        
    
    def deltagradw1(self,x1,w1,x2,w2,x3,w3,y,b):
        y_prime = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
        diff = y - y_prime
        delta =  diff * (-1) * (x1)
        return delta 
    
    def deltagradw2(self,x1,w1,x2,w2,x3,w3,y,b):
        y_prime = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
        diff = y - y_prime
        delta =  diff * (-1) * (x2)
        return delta 
    
    def deltagradw3(self,x1,w1,x2,w2,x3,w3,y,b):
        y_prime = (x1 * w1) + (x2 * w2) + (x3 * w3) + b
        diff = y - y_prime
        delta =  diff * (-1) * (x3)
        return delta
    
    
    def alpha_optimizer(self, tau , alpha):
        c = 10
        k = c / (c + tau)
        return alpha * k
  
    def fit(self, learning_rate = 0.000005): # error rate really matters , however finally was able to stabalize weights , #0.000005 ~ alpha
        weight1 = 0.5
        weight2 = 0.5
        weight3 = 0.5
        
        bias = 0.67
        
        sum1 = []
        sum2 = []
        sum3 = []
        sumbias = []
        
        rho = learning_rate # rolling alpga
        
        test_index = 0
        
        times = 0
        
        err = list()
        
        for x1 , x2 , x3 ,in self.X:
            y_prime = (weight1 * x1) + (weight2 * x2) + (weight3 * x3) + bias
            if self.SSE(self.Y[test_index],y_prime) > 0:
                sum1.append(self.deltagradw1(x1,weight1,x2,weight2,x3,weight3,self.Y[test_index],bias))
                sum2.append(self.deltagradw2(x1,weight1,x2,weight2,x3,weight3,self.Y[test_index],bias))
                sum3.append(self.deltagradw3(x1,weight1,x2,weight2,x3,weight3,self.Y[test_index],bias))
                sumbias.append(self.delta_bias(x1,weight1,x2,weight2,x3,weight3,self.Y[test_index],bias))
                times += 1
                if times % self.batch_size == 0:
                    weight1 -= sum(sum1) * self.alpha_optimizer(test_index,rho)
                    weight2 -= sum(sum2) * self.alpha_optimizer(test_index,rho)
                    weight3 -= sum(sum3) * self.alpha_optimizer(test_index,rho)
                    bias -= sum(sumbias) * self.alpha_optimizer(test_index,rho)
                    rho =  self.alpha_optimizer(test_index,rho)
            err.append((self.SSE(self.Y[test_index],y_prime),[bias,weight1,weight2,weight3]))
             
            
            
            test_index += 1
        global_min = min(err)
        print(global_min)
        err.clear()
        
        return global_min
    
     

# South Oil & Natural Gas Models
                               
model_south = LinearRegressor(X = southarr,Y = testsouth)     # checks out ~~

model_south.fit()

model_south_nan = LinearRegressor(X = southarrnan , Y = testsouth)

model_south_nan.fit(learning_rate=0.0000006)

# West Oil & Natural Gas Models

model_west = LinearRegressor(X = westarr , Y = testwest)

model_west.fit()

model_west_nan = LinearRegressor(X = westarrnan , Y = testwest)

model_west_nan.fit(learning_rate = 0.0000009)

# North East Oil & Natural Gas Models

model_ne = LinearRegressor(X = northeastarr , Y = testne)

model_ne.fit()

model_ne_nan = LinearRegressor( X = northeastarrnan , Y = testne)

model_ne_nan.fit(learning_rate = 0.0000008)


# MidWest Oil & Natural Gas Models

model_mw = LinearRegressor(X = midwestarr , Y = testmidwest )

model_mw.fit(learning_rate = 0.000000009) 


model_mw_nan = LinearRegressor(X = midwestarrnan , Y = testmidwest , batch_size = 700)

model_mw_nan.fit(learning_rate = 0.00004)


## creating flask app
import flask

host_file = __name__

app = flask.Flask(host_file)

@app.route("/hello")
def chev():
    ## so here we then start either using a template and start making request
    
app.run()
                    
                
            
                    
            
            
                
                
                
            
            
            
            
  

        

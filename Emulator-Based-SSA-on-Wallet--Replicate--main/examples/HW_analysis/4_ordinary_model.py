#!/usr/bin/env python3
from rainbow.devices.stm32 import rainbow_stm32f215 as rainbow_stm32
import numpy as np
import pandas as pd
import random
import pickle
import os
from rainbow import TraceConfig, HammingWeight
from lascar import TraceBatchContainer, Session, NicvEngine
from rainbow.utils.plot import viewer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def containsPin(e, pin_attempt, stored_pin):
    """ Handle calling the pin comparison function using the emulator """
    e.reset()

    stor_pin = 0x08008110 + 0x189  # address of the storagePin->rom
    e[stor_pin] = bytes(stored_pin + "\x00", "ascii")

    input_pin_addr = 0xcafecafe
    e[input_pin_addr] = bytes(pin_attempt + "\x00", "ascii")

    e['r0'] = input_pin_addr
    e['lr'] = 0xaaaaaaaa

    e.start(e.functions['storage_containsPin'], 0xaaaaaaaa)
    
    
    
print("Setting up emulator")
e = rainbow_stm32(trace_config=TraceConfig(register=HammingWeight(), instruction=True))
e.load("trezor.elf")
e.setup()

m = 15000 # we do anaylsis on first 15000 traces of CSV file
df = pd.read_csv("traces.csv", nrows = m)
df = np.array(df,dtype=float)

X = df[:m,1:-2] #X contain entire csv file apart from last two column which is entered_pin and stored_pin
Y_1 = [[],[],[],[]]  
Y_2 = [[],[],[],[]]

# Y_1 = np.array(Y_1)
# print(X.shape)
# print(Y_1.shape)

Y1 = df[:m,-1] #it contain last column which have stored_pin
Y2 = df[:m,-2] #it contain second last column which have entered_pin

# print("\nvalue of Y1 :")
# print(Y1)
# print("\nvalue of Y2 :")
# print(Y2)


for i in range(4):
    Y_1[i] = Y1%10 #- Y2%10
    Y_2[i] = Y2%10
    Y1 = Y1//10
    Y2 = Y2//10
# print("\nvalue of Y_1 :")
# print(Y_1)
# print("\nvalue of Y_2 :")
# print(Y_2)


mean = []
std = []
print(X.shape[1])
for i in range(X.shape[1]):
    mean.append(np.mean(X[:,i]))
    std.append(np.std(X[:,i]))
    
mean = np.array(mean) # contains means of each column(total column=46)
std = np.array(std)
# print("\nvalue of mean :")
# print(mean)
# print("\nvalue of std :")
# print(std)

index = [] # it keeps track of the column indices that have at least one non-matching element with first row's element
count = [0 for i in range(X.shape[1])] #it keeps track of the number of non-matching elements with mean of that column 

for i in range(X.shape[1]): #X.shape[1] is total column in X, here which is 46 
    for j in range(X.shape[0]): #X.shape[0] is total row in X, here which is 1500 
        if(mean[i]!= X[0,i]):
            count[i] = count[i]+1
    if(count[i]!=0):
        index.append(i)

# print(X.shape[0])
# print(X.shape[1])
# print(len(index))
# print(index)
X = X[:,index]
mean = mean[index]
std = std[index]
X = (X - mean)/std # do normalization on each element
# print("size of index : "+len(index))
# print(index)
def get_clf(f_in):
    clf = []

    for i in range(4):
        f = f_in + f"{i}.sav"
        if os.path.exists(f):
            print("Loading", i)
            clf.append(pickle.load(open(f, 'rb')))

        else:
            print("Training", i)
            clf.append(SVC(kernel='linear', probability=True))
            clf[i] = clf[i].fit(X, Y_1[i] - Y_2[i])
            pickle.dump(clf[i], open(f, 'wb'))

    return clf


N = 10    #represent the number of iterations
trials = 15 #the number of trials per iteration 
pin_len = 4 #length of the PIN 

def run(clf, strat):
    ranks = np.zeros((4, trials, N)) #3D numpy array
    for i in range(N):
        options = "123456789"
        STORED_PIN = "".join(random.choices(options, k = pin_len))

        traces = []
        res = []
        preds = []
        inps = []
        flags = [0, 0, 0, 0]
        if strat == 1: #choosen pin strategy
            pred = "5555"
        if strat == 2:
            pred = "1111" #iterative PIN Strategy
        
        for j in range(trials):

            # print("Trace test", j)
            if strat == 0:
                input_pin = "".join(random.choices(options, k = pin_len))  
            else:
                input_pin = pred

            pred = ""

            containsPin(e, input_pin, STORED_PIN)
            # print("Index:")
            # print(index)
            # traces.append(np.array([event["register"] for event in e.trace if "register" in event]))
            traces.append(np.array([event["register"] for event in e.trace if "register" in event]))
            # traces.append(np.array([event for event in e.trace]))
            # print("traces:")
            # print(e.trace)
            traces[j] = (traces[j][index] - mean)/std #normalization
            traces[j] = traces[j].reshape(1, -1)
            # print(traces)

            # print(input_pin)
            for k in range(4):
                val = clf[3 - k].predict_proba(traces[j])[0] # val contains probability of each digit, total class = 17
                # print(f"\nk :{k}")
                # print(f"val size: {val.size}")

                # print(f"val: {val}")
                in_dig = int(input_pin[k])
                # print(f"in_dig: {in_dig}")
                idxes = [i - in_dig + 8 for i in range(1, 10)]
                # print(f"idex: {idxes}")
                # print(in_dig, idxes)
                sum = np.sum(val[idxes])
                # print(f"sum: {sum}")
                if len(res) == k:
                    res.append(val[idxes] / sum)
                else:
                    # print("hellllllllllllllllllllllllllllo")
                    res[k] += val[idxes]/sum 
                # print(f"res of {k}:")
                # print(res[k])
                idx = np.argmax(res[k]) + 1 #return index of res[k] such that value of that index is max out of all value in res[k]
                # print(f"idx {i}:")
                # print(idx)
                ranks[k][j][i] = 9 - np.searchsorted(np.sort(res[k]), res[k][int(STORED_PIN[k]) - 1])
                pred += str(idx)
                if np.argmax(val) == 8:
                    flags[k] = 1

            preds.append(pred)
            inps.append(input_pin)

            if strat == 2:
                temp = ""
                for k in range(4):
                    if flags[k] != 1:
                        temp+= str(int(input_pin[k]) + 1)
                    else:
                        temp += input_pin[k]
                pred = temp
        print("\nOutput of iteration :", i)
        print("Input Pins", inps)
        print("Predicted", preds)
        print("Actual", STORED_PIN)

    # print(np.mean([len(preds[i]) for i in range(len(preds))]))

    return np.mean(ranks, axis = 2)




# means = run(get_clf("model_"), 0)

# for i in range(len(means)):
#     plt.plot(means[i], label = "digit" + str(i + 1))
    
# #X-axis reprsent number of tries for each iteration   
# #Y-axos represnt mean rank 

# plt.legend()
# plt.title("Mean Rank Progression using Ordinary Strategy, trained on 15000 samples")
# plt.savefig("./temppppppp mean_rank_Ordinary.png")
# plt.show()




# means = run(get_clf("model_"), 1)

# for i in range(len(means)):
#     plt.plot(means[i], label = "digit " + str(i + 1))

# plt.legend()
# plt.title("Mean Rank Progression using Chosen Pin Strategy, trained on 1500 samples")
# plt.savefig("./temppppppppppp mean_rank_Choosen_Pin.png")
# plt.show()





means = run(get_clf("model_"), 2)

for i in range(len(means)):
    plt.plot(means[i], label = "digit" + str(i + 1))

plt.legend()
plt.title("Mean Rank Progression using Iterative Pin Strategy, trained on 1500 samples")
plt.savefig("./tempppppppp mean_rank_Iterative.png")
plt.show()


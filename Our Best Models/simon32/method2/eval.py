import simon as si
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model



diff1 = (0x0,0x40)

def evaluate(net,X,Y):
    
    batch_size = 5000 
    Z = net.predict(X,batch_size=batch_size).flatten();
    Zbin = (Z >= 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    # mreal = np.median(Z[Y==1]);
    # high_random = np.sum(Z[Y==0] > mreal) / n0;
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    # print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);



def eval(types,num_rounds,pairs):
    if pairs == 1:        
        diff2 = (0x0,0x2000)
        diff_str = "0x40_0x2000"
    elif pairs == 2:        
        diff2 = (0x0,0x8000)
        diff_str = "0x40_0x8000"
    elif pairs == 4:        
        diff2 = (0x0,0x8000)
        diff_str = "0x40_0x8000"
        
    print(f"type: Trained using difference {diff_str} with {10**7} samples, num_rounds: {num_rounds},pairs: {pairs}")
    model_path = f"./{diff_str}/simon_model_{num_rounds}r_depth1_num_epochs30_pairs{pairs}.h5"
    X, Y = si.make_train_data_twodiff(diff1=diff1,diff2=diff2,n=10**6, nr=num_rounds, pairs=pairs, diff=diff1,)
    # print(X.shape)
    net= load_model(model_path)
    evaluate(net,X,Y)



types = "twodiffs"
num_rounds = 9
pairs = 1
eval(types , num_rounds, pairs)
pairs = 2
eval(types , num_rounds, pairs)
pairs = 4
eval(types , num_rounds, pairs)




'''
type: Trained using difference 0x40_0x2000 with 10000000 samples, num_rounds: 9,pairs: 1
Accuracy:  0.730577 TPR:  0.7375554208988535 TNR:  0.7235975462184202 MSE: 0.17373557
type: Trained using difference 0x40_0x8000 with 10000000 samples, num_rounds: 9,pairs: 2
Accuracy:  0.804709 TPR:  0.801729319090625 TNR:  0.8076786384582825 MSE: 0.13262516
type: Trained using difference 0x40_0x8000 with 10000000 samples, num_rounds: 9,pairs: 4
Accuracy:  0.885566 TPR:  0.8848900148566424 TNR:  0.8862422907577112 MSE: 0.08157913
'''
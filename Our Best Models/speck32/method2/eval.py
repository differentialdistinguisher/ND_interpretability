import speck as sp
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model


diff1 = (0x40,0)
diff2 = (0x0,0x8000)



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



def eval(types,num_rounds,pairs,model_path):
    
    X,Y = sp.make_train_data_twodiff(diff1,diff2,10**6,num_rounds, pairs)    
      
    net= load_model(model_path)
    evaluate(net,X,Y)



types = "twodiffs"
diff_str = "0x40000_0x8000"
num_rounds = 7
pairs = 1
print(f"type: Trained using difference {diff_str} with {10**7} samples, num_rounds: {num_rounds},pairs: {pairs}")
eval(types , num_rounds, pairs, f"./model_{num_rounds}r_depth5_num_epochs40_pairs{pairs}.h5")
pairs = 2
print(f"type: Trained using difference {diff_str} with {10**7} samples, num_rounds: {num_rounds},pairs: {pairs}")
eval(types , num_rounds, pairs, f"./model_{num_rounds}r_depth5_num_epochs40_pairs{pairs}.h5")
pairs = 4
print(f"type: Trained using difference {diff_str} with {10**7} samples, num_rounds: {num_rounds},pairs: {pairs}")
eval(types , num_rounds, pairs, f"./model_{num_rounds}r_depth5_num_epochs40_pairs{pairs}.h5")



'''
type: Trained using difference 0x40000_0x8000 with 10000000 samples, num_rounds: 7,pairs: 1
Accuracy:  0.633916 TPR:  0.5627691281893624 TNR:  0.7053260318548039 MSE: 0.2229455
type: Trained using difference 0x40000_0x8000 with 10000000 samples, num_rounds: 7,pairs: 2
Accuracy:  0.691289 TPR:  0.659741137602643 TNR:  0.7228411531984175 MSE: 0.19871598
type: Trained using difference 0x40000_0x8000 with 10000000 samples, num_rounds: 7,pairs: 4
Accuracy:  0.76394 TPR:  0.7471058629872088 TNR:  0.7808133050604803 MSE: 0.16031879
'''
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
    X, Y = sp.make_dataset_with_group_size(diff1=diff1,diff2=diff2,n=10**6, nr=num_rounds, diff=diff1, group_size=pairs,r_start=num_rounds,types=types)
    
    net= load_model(model_path)
    evaluate(net,X,Y)



types = "twodiffs"
diff_str = "0x40000_0x8000"
num_rounds = 7
pairs = 1
print(f"type: Trained using difference {diff_str} with {10**7} ciphertext pairs, num_rounds: {num_rounds},pairs: {pairs}")
eval(types , num_rounds, pairs, f"./{num_rounds}_{pairs}_mc_distinguisher.h5")
pairs = 2
print(f"type: Trained using difference {diff_str} with {10**7} ciphertext pairs, num_rounds: {num_rounds},pairs: {pairs}")
eval(types , num_rounds, pairs, f"./{num_rounds}_{pairs}_mc_distinguisher.h5")
pairs = 4
print(f"type: Trained using difference {diff_str} with {10**7} ciphertext pairs, num_rounds: {num_rounds},pairs: {pairs}")
eval(types , num_rounds, pairs, f"./{num_rounds}_{pairs}_mc_distinguisher.h5")



'''
type: Trained using difference 0x40000_0x8000 with 10000000 ciphertext pairs, num_rounds: 7,pairs: 1
Accuracy:  0.626563 TPR:  0.571268 TNR:  0.681858 MSE: 0.22553148497275816
type: Trained using difference 0x40000_0x8000 with 10000000 ciphertext pairs, num_rounds: 7,pairs: 2
Accuracy:  0.667 TPR:  0.636044 TNR:  0.697956 MSE: 0.2101076086075904
type: Trained using difference 0x40000_0x8000 with 10000000 ciphertext pairs, num_rounds: 7,pairs: 4
Accuracy:  0.712636 TPR:  0.701608 TNR:  0.723664 MSE: 0.1900805673814708
'''
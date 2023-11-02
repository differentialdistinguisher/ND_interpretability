# Investigating and Enhancing the Neural Distinguisher
## 1. The folder **Figures** contains the results of studies on different neural distinguishers in the article.

- The folder **cd** contains the results of the study of the neural distinguishers using ciphertext differentials.

- The folder **cp** contains the results of the study of the neural distinguisher using ciphertext pairs.

## 2 The folder **Accuracy List** gives the accuracy of different neural distinguishers

- speck: The 7-round neural distinguishers trained with $k$ ciphertext pairs for Speck32
- simon: The 9-round neural distinguishers trained with $k$ ciphertext pairs for Simon32


## 3. The folder **Our Best Models** gives our enhanced neural distinguishers for the 7-round Speck32 and the 9-round Simon32.

- method1: Neural distinguishers trained with ciphertext pairs only.

- method2: Neural distinguishers trained using data format in [1].

-   If you want to evaluate our neural distinguishers, please go to the corresponding folder and execute the following code
```bash
python eval.py
```

## 4. References
```bash
[1] Zhang L, Wang Z. Improving Differential-Neural Cryptanalysis with Inception[J]. Cryptology ePrint Archive, 2022.
```

# Week1 Summary

 <details><summary><font size=4>DeepMove</font></summary>


1. Done 
    1. modify and run codes with python 3.8 and without cuda
        1. delete cuda part, disable use-cuda
        2. c_Pickle is deprecated, use pickle instead
        3. replace os.mkdir with os.makedir because the dir may exist
        4. correct some wrong code
    2. implement two [RnnNumpy Class](../codes/DeepMove/codes/RnnNp.py), respectly RNNNumpy1 and RNNNumpy2 to replace torch Rnn and verify the correctness using 
        1. is to instanitiate 
            - input: parameters for rnn
            - output: a rnn isntance 

2. To be done
    1. extend RnnNumpy to LSTMNumpy and GRUNumpy Class
    2. Now I just test simple mode. I should work on different model_modes.
    ```python
        model_modes = ['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long']
    ```
    3. use cupy to speed up the calculation

</p>
</details>


<details><summary><font size=4>MobilityPrediction</font></summary>
haven't started yet
</details>

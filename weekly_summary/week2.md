# Week2 Summary
 ##  DeepMove
### Components
Here I implement three *.py file, namely:

- [model_edit.py](../codes/DeepMove/codes/model_edit.py)
- [train_edit.py](../codes/DeepMove/codes/train_edit.py)
- [main_edit.py](../codes/DeepMove/codes/main_edit.py)  

    
1\. **model_edit.py** is to implement different models for training and test. There are differnt model may contain different parameters, such as loc, tim, history...
 ```python
    model_modes = ['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long']
 ```  
    
2\. **train_edit.py** create some functions to ouput the training set and test set according to our need.
Besides, it defines a training function.

3\. **main_edit.py** is to train the specified model from model_edit.py with variable parameters. If we are satisfied with the model, we could store it as a pretrained-model.  

### The whole logic

The code starts from ```run(args)``` in **main_edit.py**, first initilizes parameters(dict) for model and trainning. Then according to parameters.model_modes, we choose corresponding model.   

Afterwards, we use ```generate_input_history``` from **train_edit.py** . ```generate_input_history``` takes ```(parameters.data_neural, 'train' or 'test', parameters.history_mode, candidate)``` as inputs. 
- The first parameters.data_neural is whole data we get, it's actually a nested dict. And we need to extract useful data from it. 
- About the second parameter,  if we are going to get training set, we use ```'train'``` as the second parameter, vice versa for test set. 
- The last parameter candidate is the keys of parameters.data_neural. candidate is actually a list, which indicates the ids of users.  

The output is data_train(or data_test), train_idx(or test_idx)
- The structrue of data_train(or data_test) is like this: ```data_train[user][id] = trace```. trace is user's id-th trace, which is composed of information about location, time...
- id(dict) for training or test. ```id.keys()``` is more important than ```id.values()```.
  
Then we use these to generate speficifed training set and test set using ```generate_input_list```. As the name indicates, the outputs of this function  are two lists.  

Since we get training set, it's better to split it into two sets. One is real training set, another is validation set. the ratio of t:v is 3:1.  

About the model, at present I just implement ```TrajPreSimple1``` class corresponding to ```'simple'``` mode. It contains ```forward_propagation``` to perform forward propagation. It takes single training or test array as input and gives us hidden states and single computed_target as output. ```calculate_loss``` takes whole training or test set as input , and then perform forward propagation on each indivdual array rather than the whole set, then get computed_target and finally get loss and accuracy by computed_target and input_target. ```bptt``` is to compute and update the gradient.

Got the above done, we can start training a model using ```train_simple```. It takes training set or test set as main input, calulates loss and accuracy. If we train a model, it will also return the updated model.  

For one epoch, we run ```train_simple``` once, so train the model one time and save the temporary model. We have two criterions when changing learning rate. One is  we can decrease learning rate after indicated number of epoches, while another is we can set a patiance step, which refers to the numbers of steps after no improvement we decrease learning rate.

Every time decreasing learning rate, we load the termporary best model into code and work on it. When the iteration is over, we choose the one with highest accuray and save related parameters. 

### Need to implement further  
1\. rich **model_edit.py** with more model and modes(```'LSTM'```, ```'GRU'```)  
2\. maybe replace numpy with cupy to accelerate

### Potential challenges
1\. build LSTM and GRU network since I'am unsure I could compute the gradient manually.


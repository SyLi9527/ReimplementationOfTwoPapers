# Week3 Summary
 ##  DeepMove
 -----------------------
### Components

Here I implement three *.py file, namely:

- [model_edit.py](../codes/DeepMove/codes/model_edit.py)
- [train_edit.py](../codes/DeepMove/codes/train_edit.py)
- [main_edit.py](../codes/DeepMove/codes/main_edit.py)  

    
1\. **model_edit.py** is to implement different models for training and test. There are differnt model may contain different parameters, such as loc, tim, history...
 ```python
    model_modes = ['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long']
 ```  
    
2\. [train_edit.py](../codes/DeepMove/codes/train_edit.py)create some functions to ouput the training set and test set according to our need.
Besides, it defines a training function.

3\. **main_edit.py** is to train the specified model from model_edit.py with variable parameters. If we are satisfied with the model, we could store it as a pkt file.


--------------------------------------



### The whole logic about the program

The code starts from <font color=#8bd99d>run</font>`(args)` which is located in **main_edit.py**. args is a dictionary in which we  initilizes parameters for model and training. In function <font color=#8bd99d>run</font>`(args)`, we call several classes and function to perform train and test. We use `parameters` to instantiate class <font color=#8bd99d>RnnParameterData</font> with `args`. Now `parameters` is class instance contains all infomation we need.

Then according to parameters.model_modes, we are able to choose corresponding model. 
There are four models we could choose, which are all imported from [model_edit.py](../codes/DeepMove/codes/model_edit.py). We initilize some parameters for models, such as parameters.hidden_size and so on. We will introduce how to use the model later.

Afterwards, we call function <font color=#8bd99d>generate_input_history</font> from [train_edit.py](../codes/DeepMove/codes/train_edit.py). It takes tuple `(parameters.data_neural, 'train' or 'test', parameters.history_mode, candidate)` as inputs. 
- The first `parameters.data_neural` is data we get from [foursquare.pk](../codes/DeepMove/data/foursquare.pk), it's actually a nested dictionary. And we need to extract useful data from it. 
  
  - The keys of this dictionary indicate user ids which are from 1 to 886. 
  - And value corresponding to a key contains train_id, test_id and sessions of the user, which are lists. The sample is as follows.
      ```python
      sessions = parameters.data_neural[u]['sessions']
      # sessions = [[1,3,45,6,2,3], [3,4,6,2,34,86,5,6], ...]
      train_id = parameters.data_neural[u]['train']
      # train_id = [1,2,3,4,5,6,7,99,123, ...]
      test_id = parameters.data_neural[u]['test']
      # test_id = [6,7,8,12, ...] 

      # the ratio of the number of train_id and test_id is fixed but according to model_mode. For model_mode simple , the ratio is about 4:1 , however, in model_mode it is 2:1
      ```

- About the second parameter,  if we are going to get training set, we use `'train'` as the second parameter, vice versa for test set. 
- The last parameter candidate is the keys of parameters.data_neural, which means:  
   ```python
   candidate = parameters.data_neural.keys()
   # candidate = [1,2,4,5,6,7,8,9,10, ...,886] 
   # fixed-size
   ```
  candidate is actually a list, which indicates the ids of users.  

- The output is data_train(or data_test), train_idx(or test_idx)
  - The structrue of data_train(or data_test) is like this: 
      ```python
      data_train[user][id] = trace  
      # trace = [2, 4, 5, 6, 7, 2, 100, 2, 4]
      # fixed-size for each trace(session)
      # 10 for each in model_mode simple
      ```
   Trace is user's id-th session, which is composed of information about location, time...
  - train_id or test_id is for training or test, they are all lists. For example,
      ```python
      train_id = [[0, user1id], [0, user1id], [0, user1id]...]
      test_id = [[user1id], [user2id], [user3id] ...]
      # userxid is in candidate
      # userxid is a number from 1 to 886
      ```
  
Then we use `(mode, run_idx, data, model_mode)` as input to generate speficifed training set and test set by using function <font color=#8bd99d>generate_input_list</font>. As the name indicates, the outputs of this function  are two lists.   
- As for input, `mode` is for `train` or  `test`. `run_idx` indicates `train_id` or `train_id`. . `data` indicates `data_train` or `data_test`. And `model_mode` is parameters.model_mode. These paras are mainly from the output of function <font color=#8bd99d>generate_input_history</font>.
- As for output, regarding different `model_mode`, we can get several return information. For `model_mode` = `simple`, we can get `loc_list, tim_list, target_list`. Take target as example. Other return items are simliar.
  ```python
  target = list()
  for u, i in xxx:
      target_list.append(data[u][i]['target']) 
  # target is list containing 'target' item ('target' is also a list) in all users' sessions
  # target is like this: target = [[2,3,4,2,4543,4,8,6,3],[3,5,6,2,1,7,1,1],[2,6,3,7,21,6,8,3,1]...]
  # fixed-size for one input, will change according to input 
  ```
And the output of function <font color=#8bd99d>generate_input_list</font> 
`loc_list, tim_list, target_list` would be our training set or test set according to model_mode.

Since we get training set, it's better to split it into two sets. One is for real training set, another is validation set. We set the ratio of t:v to 3:1.  

About the model, at present I just implement <font color=#8bd99d>TrajPreSimple1</font> class corresponding to `'simple'` mode. It contains `forward_propagation` to perform forward propagation. It takes single training or test array as input and gives us hidden states and single computed_target as output. `calculate_loss` takes whole training or test set as input , and then perform forward propagation on each indivdual array rather than the whole set, then get computed_target and finally get loss and accuracy by computed_target and input_target. `bptt` is to compute and update the gradient.

To be more concrete, we will cover inputs and outputs of all class function in <font color=#8bd99d>TrajPreSimple1</font> class.
- calculate_total_loss(self, loc, tim, target): ... return L, sumA
  - inputs: loc_list, tim_list and target_list are from output of <font color=#8bd99d>generate_input_list</font>. We know each item in loc and tim is also a list. And then we iterate the list loc and tim, in each iteration, we call function forward_propagation. And set each item of loc and tim as the inputs of it. Then we feed function predict with output of forward_propagation and item of target. What we get finally is the correct location we compute.
  - outputs: L, sumA are total correct location we compute and total loaction we see respectively. They are all numbers(int).
- forward_propagation(self, loc, tim):  ... return [o, s]
   - inputs: loc and tim are one item of loc_list and tim_list from output of <font color=#8bd99d>generate_input_list</font>.
   - outputs: s represents rnn hidden states, which is matrix (size:(T, self.hidden_dim)). T is the number of location in one item of location_size. o is output we compute, which is the same size as corresponding item of target.
- predict(self, o, target): ... return sum(o_index[i] == target[i] for i in range(len(target)))
   - inputs: o is one of output of function <font color=#8bd99d>generate_input_list</font>. target is the item of target_list.
   - outputs: the number of matching location
- sgd_step(self, loc, tim, target, learning_rate):
  - inputs: item of loc_list, tim_list, target_list, learning_rate. The function is for updating gradient using bptt.
- bptt(self, loc, tim, target): ... return gradients
  - inputs: item of loc_list, tim_list, target_list, learning_rate. The function is to calculate and return gradient.
  - outputs: gradients
  
Got the above done, we can start training a model using `train_simple`. It takes training set or test set as main input, calulates loss and accuracy. If we train a model, it will also return the updated model.  



For one epoch, we run `train_simple` once, so train the model one time and save the temporary model. We have two criterions when changing learning rate. One is  we can decrease learning rate after indicated number of epoches, while another is we can set a patiance step, which refers to the numbers of steps after no improvement we decrease learning rate.

Every time decreasing learning rate, we load the termporary best model into code and work on it. When the iteration is over, we choose the one with highest accuray and save related parameters. 

-------------------------
### Need to implement further  

1\. rich **model_edit.py** with more model and modes(`'LSTM'`, `'GRU'`)  
2\. maybe replace numpy with cupy to accelerate

### Potential difficulty
1\. build LSTM and GRU network since I'am unsure I could compute the gradient manually.


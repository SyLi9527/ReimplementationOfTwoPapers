# DeepMove
## Requirements
- python 3.8.5
- tensorflow 2.4.1
- numpy 1.19.5
- json 2.0.9
- pickle 4.0
- matplotlib 3.4.2
## Structure
already in DeepMove dir

- /codes
    - main_tf.py
    - model_tf.py # define models
    - train_tf.py # define tools for train the model

- /data # preprocessed foursquare sample data (pickle file)
- /results # the default save path when training the mode
## Usage
The codes contain four network model (simple, simple_long, attn_avg_long_user, attn_local_long) and a baseline model (Markov). The parameter settings for these model can refer to  [result.md](./result.md) file.

- Train a new model:

    `python3 main_tf.py --model_mode=attn_local_long --pretrain=0`

    Other parameters (refer to main_tf.py):

    - for training:
        - learning_rate, lr_step, lr_decay, L2, clip, epoch_max, dropout_p
    - model definition:
        - loc_emb_size, uid_emb_size, tim_emb_size, hidden_size, rnn_type, attn_type
        - history_mode: avg, avg, whole
- Plot user's predicted and real trajectory:

    `python3 main_tf.py --model_mode=xxx --plot_user_traj=user` # user is valid from 0 to 885

    Then in each epoch, the user's trajtectory is saved as a png in /codes dir. The png would update automatically after each epoch.

- Apply Geolife datasets to Deepmove algorithm:

    `python3 main_tf.py --use_geolife_data=True`

    First, we need to get the dataset. You can find the detail from last item in Usage of MobilityPredictabilityUpperBounds. Then copy .npz file to /data dir. Afterwards, specify the filename in line52 of main_tf.py. Finally, run this command to get cross dataset accuracy.

# MobilityPredictabilityUpperBounds
## Requirements
- python 3.8.5
- numpy 1.20.2
- scipy 1.6.3
- healpy 1.14.0 # can't be installed on Windows OS
- apsw 3.9.2.post1 # other versions can also work
## Structure
already in MobilityPredictabilityUpperBounds dir
- /ResultsLoP_replication # the dir which saves Heatmap.csv and Heatmap.pdf
- GeolifeEntropyCalc.py # calucate entropy and build Heatmap.csv
- GeolifeSymbolisation.py # convert Geolife data to symbolized positions
- Graphing.py # plot Heatmap
- Utils.py  
- ...
## Usage

- Pre-build the database cache files by running `python3 GeolifeSymbolisation.py`. This will take a significant amount of time as it builds the trajectory database from downloaded Geolife data set and pre-computes all required trajectory quantisations. If this is not done then it will be done on first run of the main code.
- Run `python3 GeolifeEntropyCalc.py`. This computes the upper bounds for all spatiotemporal quantisations investigated in the paper and saves them to a CSV.
- Run `python3 Graphing.py`. This plots the results as heatmaps.
- Run `python3 GeolifeSymbolisation.py --get_geolife_data=True --s=xx --t=xx`. s denotes SpatialRes, and t denotes TemporalRes. Say we want to get part of Geolife dataset with SpatialRes=1000, TemporalRes = 1:00:00, we should run GeolifeSymbolisation.py --get_geolife_data=True --s=1000 --t=1:00:00. Then we can get a corresponding .npz file as "1000|1:00:00.npz". With this file, we could apply geolife datasets to DeepMove algorithm.
- Run `python3 GeolifeEntropyCalc.py --use_deepmove_dataset=True`. This command will calculate upperbound probability for DeepMove dataset ranther than Geolife dataset.



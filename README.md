# Deep Variational Metric Learning
This is the implementation of the paper Deep Variation Metric Learning (http://openaccess.thecvf.com/content_ECCV_2018/papers/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.pdf) by Xudong Lin et al. 
The paper provides us an innovative approach for modelling a deep metric learning algorithm, which significantly boosts its performance over them. The keypoint responsible for its marked performance can be listed as follows:
 - Takes into account that in the central latent space, the distribution of variance within classes is actually independent on classes itself.
 -  The core distinction is that it disentangles intra-class variance and class centers.
 -  Utilizes variational inference through which it forces the conditional distribution of intra-class variance, given a certain image sample, to be isotropic multivariate Gaussian.
 - To the best of knowledge, this is the first work that utilizes variational inference to disentangle intra-class variance and leverages the distribution to generate discriminative samples to improve robustness.
 - It is the combination of a discriminative model and a generative model, where the former outputs class centers and the latter its intra-class variance.
 -  This framework is also applicable to hard negative mining methods.

### Running Instruction:
Make sure you have all the requirements installed in your pc before execution. To do so you may use pip and execute the following:
```sh
$ pip3 install -r requirements.txt
```
##### Data Generation Part
You will have to generate a HDF5 file structure for storing the training data. To do so make sure you have a `cars_train` folder containing all the training images and `cars_train_annos.mat` file which contains the annotation, in the  `datasets/data/cars196` folder. Now execute the `cars196_converter.py` file in the datasets folder as follows:
```sh
$ cd datasets
$ python3 cars196_converter.py
```
You will then find a `cars197.hdf5` file inside the `datasets/data/cars196` folder

##### Training Part
Make sure you have generated the hdf5 file. Now you will need to change the model_dir path in `GoogleNet_Model.py` file, such that it points to the `tf_ckpt_from_caffe.mat` file in the repo. Now you are all set to execute the model.
You can set all the hyperparameters related to your training by editing respective fields in the `parameters.py` file. Then execute the code as follows:
```sh
$ python3 main.py
```
The result of the training is stored in the `tensorboard_log/cars196` directory, wherein you could analyze your result using tensorboard.

### Results
We have only trained our model for the Cars196 dataset for a batch size of 32 and Embedding size of 64 which acheives the best Recall@32 Rate of 0.866. However to acheive best of results it is ideal that we set batch size to 128 and Embedding size to 512.
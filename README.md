### Dataset

[baidu cloud](https://pan.baidu.com/s/1iDClwqtODylPI0Rj6yfAEQ)  pwd:tt23

## !!!! In order to obtain more typical data, we sampled part of the data in ECW as new data, which contains richer context information (such as gestures, interaction information)

[baidu cloud](https://pan.baidu.com/s/13AZca4cAnyH70e5GkkG83w)  pwd:l7wu

<img src="./data/saved_res/refined_data.png" style="zoom:50%;" />

we assume that:

Interaction : The information that other people who interact with the annotated person can give, mainly the face information of others

body : Posture information of the annotated person

other: other background info except for the above two cases.



Since the original data has certain noise (landmarks are marked on different individuals, the duration is too short a clip), and the data needs to be converted into a standard format, so here is a preprocessing sh



the  Class Dataset_Config 's method :Get_path, change the name 'ours' to your path which you put the ECW data.

and then run :

```
python dataloader/Unify_Dataloader.py
```

or  you just need run 

```
sh genderate_data.sh
```

as a result , You will get the split data in the save path you set in config.py



### Train

Set the model options you want in the get_common_config class of config.py, and then set the model you want to use in Unify-Trainer.py  line 52. 



```
python Unify-Trainer.py
```

it will work.



After training, you will get the following results

1. the confusion matrix in Result/Confusion_matrix dir
2. the train log and tensorboard in  ./log



More standardized code will be released after some time



### Res

acc of refined dataset:

<img src="./data/saved_res/res.png" style="zoom:50%;" />

confusion matrix (refined data):

<img src="./data/saved_res/cm.png" style="zoom:30%;" />

<img src="./data/saved_res/train.png" style="zoom:50%;" />


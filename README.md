# Top1 Solution of CheXpert

## What is Chexpert?
CheXpert is a large dataset of chest X-rays and competition for automated chest x-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets.
## Why Chexpert?
Chest radiography is the most common imaging examination globally, critical for screening, diagnosis, and management of many life threatening diseases. Automated chest radiograph interpretation at the level of practicing radiologists could provide substantial benefit in many medical settings, from improved workflow prioritization and clinical decision support to large-scale screening and global population health initiatives. For progress in both development and validation of automated algorithms, we realized there was a need for a labeled dataset that (1) was large, (2) had strong reference standards, and (3) provided expert human performance metrics for comparison.
## How to take part in?
CheXpert uses a hidden test set for official evaluation of models. Teams submit their executable code on Codalab, which is then run on a test set that is not publicly readable. Such a setup preserves the integrity of the test results.

Here's a tutorial walking you through official evaluation of your model. Once your model has been evaluated officially, your scores will be added to the leaderboard.**Please refer to the**[https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
## What the code include?
* If you want to train yourself from scratch, we provide training and test the footwork code. In addition, we provide complete training courses
* If you want to use our model in your method, we provide **a best single network pre-training model,** and you can get the network code in the code

### train the model by yourself

* Data preparation
> We gave you the example file, which is in the folder 'config/train.csv'
> You can follow it and write its path to 'config/example.json'

* if you want to train the model,please run the command.(We use 4 1080Ti for training, so larger than 4 gpus is recommended）:
> `pip install -r requirements.txt`
> 
> `python Chexpert/bin/train.py Chexpert/config/example.json logdir --num_workers 8 --device_ids "0,1,2,3"`

* if you want to test your model,please run the command:
> `cd logdir/`

* cuz we set "save_top_k": 3 in the config/example.json, so we may have got 3 models for ensemble here. So you should do as below:
> `cp best1.ckpt best.ckpt`
> 
> `python classification/bin/test.py`

* if you want to plot the roc figure and get the AUC, please run the command
> `python classification/bin/roc.py plotname`

 * *How to drink a cup of coffee?*
> you can run the command like this. Then you can have a cup of caffe.(log will be written down on the disk)
`python Chexpert/bin/train.py Chexpert/config/example.json logdir --num_workers 8 --device_ids "0,1,2,3" --logtofile True &`

### train the model with pre-trained weights
* we provide one pre-trained model here: `config/pre_train.pth`
we test it on 200 patients dataset, got the **AUC** as below:

|Cardiomegaly|Edema|Consolidation|Atelectasis|Pleural_Effusion|
|---------|-----|---|----|-----|
|0.8703|0.9436|0.9334|0.9029|0.9166|

* you can train the model with pre-trained weights,run the command as below:

> `python Chexpert/bin/train.py Chexpert/config/example.json logdir --num_workers 8 --device_ids "0,1,2,3" --pre_train "Chexpert/config/pre_train.pth" `

### Contact
* If you have any quesions, please post it on github issues or email at coolver@sina.com

### Reference
* [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
* [http://www.jfhealthcare.com/](http://www.jfhealthcare.com/)









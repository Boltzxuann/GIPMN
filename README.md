# GIPMN
Codes for "A Structured Graph Neural Network for Improving the Numerical Weather Prediction of Rainfall"

## Environmental Settings
a) anaconda & jupyter notebook : https://www.anaconda.com/products/individual 
<br>
b)PyTorch : https://pytorch.org/
<br>
c) deep graph libary : https://www.dgl.ai/
<br>
d) meteva : https://www.showdoc.com.cn/meteva/3975600102516402
<br>
## How to Use
Just run `train_main.py` , a single GIPMN model will begin training. There is a tiny dataset with 10 examples used for demonstrate the training process.
<br>
If you want to train a model for a specific rainfall level, for example, 10 mm, run `train_main.py -- level_threshold 10 -- level_width 3`, where the parameter "level_threshold" is the lower bound of the rainfall level, "level_width" is the rainfall range corresponding to probability of 0.1 ~ 0.9 .

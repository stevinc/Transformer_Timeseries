# 🐐 Time Series Forecasting
Time series forecasting models

## Set-up

- Install the required packages (pip or conda)
    - `pip install -r requirements.txt`
    - `conda env create -f env.yml`

- Download data
    - https://drive.google.com/file/d/1Na7e2yJy1Oix8-HcKQS97u1VZodpZ-OZ/view?usp=sharing
   
- Train and Test on electricity dataset
    - `python ./main.py --exp_name electricity --conf_file_path ./conf/electricity.yaml`
    
    Plot prediction on Test set
    - `python ./main.py --exp_name electricity --conf_file_path ./conf/electricity.yaml --inference=True`

    
 ## Models
 
 -  Temporal fusion transformer<br>
    https://arxiv.org/pdf/1912.09363.pdf
    
    Usage:<br>
        - `model: tf_transformer`
 
 -  Transformer<br>
    https://arxiv.org/pdf/1706.03762.pdf<br>
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    
    Usage:<br>
        - `model: transformer`
    
 -  GRN-Tranformer <br>
    Use GRN block after multi-head attention to encode static variables
    
    Usage:<br>
        - `model: grn_transformer`

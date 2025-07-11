## Fork from "RETFound - A foundation model for retinal imaging"


Adaptation of the RETFound pretrain/finetune pipelines for FAME2 dataset. 

The goal of the project is to evaluate perfomances of pretrained Vision Transformers on classification of cardiac event prediction. 


<img src=images/Pretrain_scheme.png alt="Example Image" width="500">


### 📝Key features

- Different models (ViT on ImNet, ResNet) in models_vit.py 
- RETFound-FAME2 has been validated on 3-FOLD dataset consisting of XCA images with binary label (cardiac envent within 2 years or not)



### 🔧Install environment

1. Create environment with conda:

```
conda create -n retfound python=3.7.5 -y
conda activate retfound
```

2. Install dependencies

```
git clone https://github.com/rmaphoh/RETFound_MAE/
cd RETFound_MAE
pip install -r requirement.txt
```


### 🌱Fine-tuning with RETFound weights

To fine tune RETFound on your own data, follow these steps:

1. Download the RETFound pre-trained weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_cfp-XCA-FAME2 pretraining</td>
<td align="center"><a href="https://drive.google.com/file/d/1cUucmFR_24gg0rmtKyVClHGsGRKcqFqD/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left"> ViT_ImNet FAME2 pretraing</td>
<td align="center"><a href="https://drive.google.com/file/d/1rMg_U6mA3SBb-y_SvO0WHowFp6KkmzOQ/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

2. Organise your data into this directory structure 

```
├── data folder
    ├──train
        ├──VOCE_0
        ├──VOCE_1
    ├──val
        ├──VOCE_0
        ├──VOCE_1
    ├──test
        ├──VOCE_0
        ├──VOCE_1
``` 

3. Start fine-tuning. Here is an example of the finetune of the Fusion Model on FOLD_1


```
python main_finetune.py \
    --model vit_large_patch16 \
    --aa rand-m9-mstd0.5-inc10 \
    --finetune "/path/to/checkpoints/fame2/FOLD_1/Retfound_fame2_finetuned FOLD_1-best-f1.pth" \
    --task "Retfound_fame2_finetuned FOLD_1 Fusion" \
    --data_path "/path/to/FAME2/FOLD_1" \
    --use_metadata 1 \
    --freeze_backbone 1 \
    --device cuda:1 \
    --seed 2 \
    --resume "" \
    --blr 5e-3

```


4. For evaluation only


```
python main_finetune.py \
    --model vit_large_patch16 \
    --aa rand-m9-mstd0.5-inc10 \
    --finetune "/path/to/checkpoints/fame2/FOLD_1/Retfound_fame2_finetuned FOLD_1-best-f1.pth" \
    --task "Retfound_fame2_finetuned FOLD_1 Fusion" \
    --data_path "/path/to/FAME2/FOLD_1" \
    --use_metadata 1 \
    --freeze_backbone 1 \
    --device cuda:1 \
    --seed 2 \
    --resume "/path/to/checkpoints/fame2/FOLD_1/Retfound_fame2_finetuned FOLD_1-best-f1.pth" \
    --blr 5e-3 \
    --eval

```


### Load the model and weights (if you want to call the model in your code)

```python
import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

# call the model
model = models_vit.__dict__['vit_large_patch16'](
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True,
)

# interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# manually initialize fc layer
trunc_normal_(model.head.weight, std=2e-5)

print("Model = %s" % str(model))
```


## Reproducing Results

All the command lines required to reproduce the results are available here:  

[📥 Download Command Lines](https://drive.google.com/drive/folders/15RVi4gV-ZK_Ab0hj6Ice4WLKAJT6lIEn?usp=drive_link)





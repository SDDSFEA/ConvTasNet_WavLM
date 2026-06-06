# 数据集
1. libri2mix官方脚本生成8k,max
2. Generate scp file using script file of create_scp.py
3. 目前主要用的是train-100的clean,8k,max
4. 数据集加载
    `train.py`里面`from DataLoaders import make_dataloader`
    - convTasNet原论文4s一截断`DataLoaders.py`


# Training Command
   ```python
  python train.py --opt ./option/train/train.yml
   ```
   
   在TCN block里面的dwconv后面融合
   train.py里面改为`from Conv_TasNet_wavlm_dwconvFuse import ConvTasNet`
   ```python
  python train.py --opt ./options/train/train_clean100_WavLM_dwconvFuse.yml
   ```
   目前.yml里面的融合方式设的是直接拼接 fuse: concat
   
   在TCN block里面的Scconv后面融合
   train.py里面改为`from Conv_TasNet_wavlm_ScconvFuse import ConvTasNet`
   ```python
  python train.py --opt ./options/train/train_clean100_WavLM_ScconvFuse.yml
   ```
   目前.yml里面的融合方式设的是直接拼接

# Inference this model
- Inference Command (Use this command if you need to test a **large number** of audio files.)
   ```python
  python Separation.py -mix_scp 1.scp -yaml ./config/train/train.yml -model best.pt -gpuid [0,1,2,3,4,5,6,7] -save_path ./checkpoint
   ```
- Inference Command (Use this command if you need to test a **single** audio files.)

   ```python
  python Separation_wav.py -mix_wav 1.wav -yaml ./config/train/train.yml -model best.pt -gpuid [0,1,2,3,4,5,6,7] -save_path ./result
   ```

# evaluate 
经过inference得到分离后的spk1和spk2之后
```python
python SI_SNR_eval.py
```


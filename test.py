import torch
checkpoint=torch.load('D:\ML\BNN_Sequence\outputs\mnist_MLPBinaryConnect_M1_STE_lr0.01_2020_09_17_22_03_13_id0\saved_models\model_0_checkpoint_best.ckpt')
print(checkpoint['epoch'])

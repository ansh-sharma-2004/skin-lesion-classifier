import torch
import torch.nn as nn
import timm

NUM_CLASSES = 7

def get_model(dropout=0.4):
    model = timm.create_model(
        'efficientnet_b3',
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=dropout        # added dropout
    )
    return model

def get_class_weights(df):
    class_counts = df['label'].value_counts().sort_index()
    total        = len(df)
    weights      = [total / (NUM_CLASSES * count) for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32)
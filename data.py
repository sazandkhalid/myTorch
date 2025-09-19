import pandas as pd
from PIL import Image
import numpy as np
import os
import re
from pathlib import Path
import json


def load_tabular_data(path, target_col: int | str | None = -1, sep=None, **kwargs):
    if sep is None and str(path).lower().endswith(".tsv"):
        sep = "\t"
    df = pd.read_csv(path, sep=sep, **kwargs)
    if target_col is None:
        return df
    if isinstance(target_col, str):
        y = df[target_col].to_numpy()
        X = df.drop(columns=[target_col]).to_numpy()
    else:
        y = df.iloc[:, target_col].to_numpy()
        X = df.drop(df.columns[target_col], axis=1).to_numpy()
    return X, y

def load_image_data(folder_path, grayscale=True):
    images = []
    for filename in sorted(os.listdir(folder_path)): 
        if filename.lower().endswith(('.png', '.jpeg', '.jpg')):
            img = Image.open(os.path.join(folder_path, filename))
            img = img.convert("L") if grayscale else img.convert("RGB")
            images.append(np.array(img))
    if not images:
        raise FileNotFoundError("No images found.")
    return np.stack(images)  

def preprocess_image(images):
    #normalize the pixel values between 0 and 1 
    return images.astype("float32") / 255.0

def load_text_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text 


def preprocess_text(text, one_hot=False, vocab=None):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    if not one_hot:
        return tokens
    if vocab is None:
        vocab = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
    T, V = len(tokens), len(vocab)
    oh = np.zeros((T, V), dtype=np.float32)
    for t, tok in enumerate(tokens):
        idx = vocab.get(tok)
        if idx is not None:
            oh[t, idx] = 1.0
    return oh, vocab

def train_val_test_split(X,y,ratios=(0.7,0.15,0.15), seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_end = int(ratios[0] * len(X))
    val_end = train_end + int(ratios[1] * len(X))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def save_data(path, data, **kwargs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if isinstance(data, np.ndarray):
        if ext not in ('.npy', '.npz'):
            p = p.with_suffix('.npy')
        np.save(p, data)
        return str(p)

    if isinstance(data, pd.DataFrame):
        if ext == '.parquet':
            data.to_parquet(p, **kwargs)
        else:
            p = p.with_suffix('.csv') if ext != '.csv' else p
            data.to_csv(p, index=False, **kwargs)
        return str(p)

    if isinstance(data, str):
        if ext != '.txt':
            p = p.with_suffix('.txt')
        with open(p, 'w', encoding=kwargs.get('encoding', 'utf-8')) as f:
            f.write(data)
        return str(p)

    if isinstance(data, (list, dict)):
        if ext != '.json':
            p = p.with_suffix('.json')
        with open(p, 'w', encoding=kwargs.get('encoding', 'utf-8')) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(p)

    raise TypeError(f"Unsupported type for saving: {type(data)}")



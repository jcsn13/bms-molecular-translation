import os
import cv2
import torch
import albumentations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functions import Tokenizer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from functions import get_transforms, Tokenizer, split_form, split_form2
from constants import CFG
from torch.nn.utils.rnn import pad_sequence

tqdm.pandas()


def download_data():
    download_link = os.getenv("download_link")

    os.system(
        "gdown --fuzzy 'https://drive.google.com/file/d/1cXqZPlQz7MZ3zn6NbYg4hnzc_t7ko8CY/view?usp=sharing'"
    )
    os.system(
        "mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    )
    os.system("kaggle competitions download -c bms-molecular-translation")
    os.system(
        "mkdir data && mkdir data/processed && mv bms-molecular-translation.zip data"
    )
    os.system("cd data && unzip bms-molecular-translation.zip")
    os.system("rm -f bms-molecular-translation.zip")


def get_file_path(image_id, mode):
    return "data/{}/{}/{}/{}/{}.png".format(
        mode, image_id[0], image_id[1], image_id[2], image_id
    )


def plot_example(df):
    img = cv2.imread(df.sample(n=1)["file_path"].item())
    plt.imshow(img)
    plt.show()


class ModelDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, train=True):
        super(ModelDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df["file_path"].values
        self.transform = transform
        self.train = train
        if self.train:
            self.labels = df["InChI_text"].values
        else:
            self.fix_transform = albumentations.Compose(
                [albumentations.Transpose(p=1), albumentations.VerticalFlip(p=1)]
            )

    def __getitem__(self, idx):
        if self.train:
            file_path = self.file_paths[idx]
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            label = self.labels[idx]
            label = self.tokenizer.text_to_sequence(label)
            return image, torch.LongTensor(label)
        else:
            file_path = self.file_paths[idx]
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            h, w, _ = image.shape
            if h > w:
                image = self.fix_transform(image=image)["image"]
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image

    def __len__(self):
        return len(self.df)


def preprocess_training_data(logger):
    df_train = pd.read_csv("data/train_labels.csv")

    df_train["InChI_1"] = df_train["InChI"].progress_apply(lambda x: x.split("/")[1])
    df_train["InChI_text"] = (
        df_train["InChI_1"].progress_apply(split_form)
        + " "
        + df_train["InChI"]
        .apply(lambda x: "/".join(x.split("/")[2:]))
        .progress_apply(split_form2)
        .values
    )

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train["InChI_text"].values)
    torch.save(tokenizer, "data/processed/tokenizer.pth")
    logger.info("Saved tokenizer")

    lengths = []
    tk0 = tqdm(df_train["InChI_text"].values, total=len(df_train))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    df_train["InChI_length"] = lengths
    df_train.to_pickle("data/processed/train.pkl")
    logger.info("Saved preprocessed train.pkl")


def read_training_data():
    tokenizer = torch.load("data/processed/tokenizer.pth")
    df_train = pd.read_pickle("data/processed/train.pkl")
    df_train["file_path"] = df_train["image_id"].apply(
        lambda x: get_file_path(x, "train")
    )

    train_dataset = ModelDataset(
        df_train, tokenizer, transform=get_transforms(data="train")
    )

    return train_dataset, tokenizer


def read_test_data():
    tokenizer = torch.load("data/processed/tokenizer.pth")
    df_test = pd.read_csv("data/sample_submission.csv")
    df_test["file_path"] = df_test["image_id"].apply(lambda x: get_file_path(x, "test"))

    test_dataset = ModelDataset(
        df_test, tokenizer, transform=get_transforms(data="test"), train=False
    )

    return test_dataset, tokenizer


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_data_loader(dataset, mode="train"):
    tokenizer = torch.load("data/processed/tokenizer.pth")
    if mode == "train":
        loader = DataLoader(
            dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=Collate(tokenizer.stoi["<pad>"]),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

    return loader


if __name__ == "__main__":
    download_data()

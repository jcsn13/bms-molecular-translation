import torch
import sys
import logging
import torch.nn as nn
import torch.optim as optim

from constants import CFG
from functions import seed_torch, get_score

from tqdm import tqdm
from models import ImgCaptionModel
from data import get_data_loader, read_training_data, preprocess_training_data

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def train():
    seed_torch(CFG.seed)

    logger.info("STARTED reading Training Data")

    preprocess_training_data(logger)
    train_data, tokenizer = read_training_data()
    train_loader = get_data_loader(train_data)

    logger.info("FINISHED reading Training Data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using {device} device")

    model = ImgCaptionModel(
        embed_size=CFG.embed_size,
        hidden_size=CFG.hidden_size,
        vocab_size=len(tokenizer),
        n_layers=CFG.num_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)

    model.train()

    logger.info("STARTED Traning Neural Network")

    for epoch in range(CFG.n_epochs):
        for idx, (imgs, inchis) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            inchis = inchis.to(device)

            outputs = model(imgs, inchis[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), inchis.reshape(-1))
            score = get_score(inchis, outputs)
            
            logger.info(f"Epoch {epoch + 1} - Training loss: {loss.item()}")
            logger.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

    logger.info("FINISHED Traning Neural Network")

    logger.info("Saving the model on file")
    torch.save(model.state_dict(), "ImgCaptionModel.pth")


if __name__ == "__main__":
    train()

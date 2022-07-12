import torch
import numpy as np

from constants import CFG
from tqdm import tqdm
from data import read_test_data, get_data_loader
from models import ImgCaptionModel
from torch.utils.data import DataLoader


def inference(test_loader, model, tokenizer, device):
    model.eval()
    text_preds = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            predictions = model.predict(images, tokenizer, CFG.max_len)
        _text_preds = tokenizer.predict_captions(predictions)
        text_preds.append(_text_preds)
    text_preds = np.concatenate(text_preds)
    return text_preds


def get_text(prediction):
    text = ""

    for char in prediction[1:]:
        if char == "<pad>" or char == "<eos>":
            break
        text += char
    return text


def test():
    test_data, tokenizer = read_test_data()
    test_loader = get_data_loader(test_data, mode="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImgCaptionModel(
        embed_size=CFG.embed_size,
        hidden_size=CFG.hidden_size,
        vocab_size=len(tokenizer),
        n_layers=CFG.num_layers,
    ).to(device)

    state_dict = torch.load("ImgCaptionModel.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)

    preds = inference(test_loader, model, tokenizer, device)

    test_data.df["InChI"] = [f"InChI=1S/{text}" for text in preds]
    test_data.df[["image_id", "InChI"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    test()

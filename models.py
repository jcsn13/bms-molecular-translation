import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from constants import CFG


class Encoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(Encoder, self).__init__()
        self.train_CNN = train_CNN
        self.CNN = models.resnet18(pretrained=True)
        self.CNN.fc = nn.Linear(self.CNN.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(CFG.dropout)

    def forward(self, x):
        x = self.CNN(x)
        return self.dropout(self.relu(x))


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, n_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(CFG.dropout)

    def forward(self, x, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((x.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class ImgCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, n_layers):
        super(ImgCaptionModel, self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, n_layers)

    def forward(self, x, captions):
        x = self.encoder(x)
        outputs = self.decoder(x, captions)
        return outputs

    def predict(self, x, vocabulary, max_length=CFG.max_len):
        result = torch.zeros([x.size(0), max_length])

        with torch.no_grad():
            x = self.encoder(x).unsqueeze(0)
            states = None

            for t in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result[:, t] = predicted

                if np.argmax(output.detach().cpu().numpy()) == vocabulary.stoi["<eos>"]:
                    break
                    
                x = self.decoder.embed(predicted).unsqueeze(0)

        return result

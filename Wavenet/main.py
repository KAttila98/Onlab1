import text_data
import networks
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Data preprocessing and model compiling
corpus = text_data.Corpus("data/wikitext")

n_tokens = len(corpus.dictionary)

model = networks.WaveNet(layer_size = 5, stack_size = 2, n_tokens = n_tokens, in_channels = 256, res_channels = 512)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


def train(inputs, targets):
    """
    Train 1 time
    :param inputs: Tensor[batch, timestep, channels]
    :param targets: Torch tensor [batch, timestep, channels]
    :return: float loss
    """
    preds, logits = model(inputs)

    loss = criterion(logits.view(-1, n_tokens),
                     targets.long().view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

train_data = text_data.TextDataset(corpus.train, model.receptive_fields, sample_size = 100)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = False)

# Training
epochs = 2

torch.cuda.empty_cache()

model.train()
for e in range(epochs):
    for b in tqdm(train_loader):
        inp, out = b
        loss = train(inp, out)

    print(f'[{e + 1}/{epochs}] loss: {loss}')

# Evaluating
sentence = corpus.test[0:63].unsqueeze(0).cuda()
generated = sentence
decoder = corpus.dictionary.idx2word

print("Generating text with seed:")
print(' '.join([decoder[i] for i in generated.tolist()[0]]))

sample_size = 63
for i in range(50): # Generating 50 consecutive words
    y_hats, _ = model(sentence)
    preds = torch.argmax(y_hats, dim = -1)
    generated = torch.cat((generated, preds), dim=1)
    sentence = generated[:,-sample_size:]

l_gen = generated.tolist()[0]
gen_text = ' '.join([decoder[i] for i in l_gen])
print(gen_text)
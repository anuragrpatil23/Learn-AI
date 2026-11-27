import os
import pathlib
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Hyperparameters
BATCH_SIZE = 4  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 8  # what is the maximum context length for predictions?
SPLIT = "train"

MAX_ITERS = 5000
LEARNING_RATE = 1e-3
N_EMBD = 32
N_HEAD = 4

EVAL_INTERVAL = 500
EVALUATION_ITERATIONS = 200

DEVICE = "mkl" if torch.backends.mkl.is_available() else "cpu"

# read it in to inspect it
with open("./dataset/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)


# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    itos[i] for i in l
)  # decoder: take a list of integers, output a string


# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)


# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# Now we are gonna take arbitary chucks of this data and train it. This chuck is called block size. We need to do this cause its computationally prohibitive to take and train all the data at once.
# lets look at the train data.
BLOCK_SIZE = 8
train_data[: BLOCK_SIZE + 1]


# Data loading
def get_batch(split=SPLIT):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - BLOCK_SIZE, (BATCH_SIZE,)
    )  # generate batch size number of random numbers between 0 and len(data)-BLOCK_SIZE

    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device=DEVICE), y.to(device=DEVICE)
    return x, y


@torch.no_grad()  # this tells pytorch that we will no call .backward on. And so pytorch can be a lot more memory efficient.
def estimate_loss():
    out = {}
    model.eval()  # this tells model to behaviour in eval mode. some models have different behaviour in train and val
    for split in ["train", "val"]:
        losses = torch.zeros(EVALUATION_ITERATIONS)
        for k in range(EVALUATION_ITERATIONS):
            X, Y = get_batch(split=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


# Bigram Model defination
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        # embedding is a thin wrapper around the tensor which creates vocab_size, vocab_size tensor.

        # Creating an embedding layer for 10 unique items, each represented by 5-dimensional embeddings
        # embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=5)

        # In the above example, you could indeed use a regular tensor of size (10, 5) to represent embeddings for 10 unique items, and there would not be a functional difference in this specific scenario. However, using the nn.Embedding layer has certain advantages and is more common in practice when working with embeddings for categorical data. Let's explore the reasons:
        # Memory Efficiency: When dealing with large datasets or vocabularies, using an nn.Embedding layer can be more memory-efficient. The embedding layer only stores the embeddings for the unique items (based on the number of unique indices), while a regular tensor of size (10, 5) would allocate memory for all elements, including those that might not correspond to any actual item.
        # Computational Efficiency: The nn.Embedding layer has optimized implementations for efficient indexing and lookup operations. When you pass a batch of input indices to the embedding layer, it efficiently retrieves the corresponding embeddings. This optimized implementation is especially beneficial when working with large-scale models and datasets.
        # Flexibility: The nn.Embedding layer is designed to be integrated seamlessly with other layers in PyTorch's neural network modules (nn.Module). It allows you to easily train and update the embeddings during the learning process, making it convenient for end-to-end training.
        # Integration with Embedding Lookup: When using nn.Embedding, you can utilize the torch.nn.functional.embedding function, which provides efficient embedding lookup capabilities. This function is particularly useful when you need to perform lookups across multiple indices simultaneously, such as in recurrent neural networks (RNNs) or transformer models.
        # However, if your use case involves a small number of unique items, and memory or computational efficiency is not a concern, you can still represent embeddings using regular tensors. The decision between using a regular tensor and an nn.Embedding layer depends on the specific requirements of your model, the size of your vocabulary, and your computational resources. For larger-scale applications, nn.Embedding is generally preferred due to its efficiency and convenience.
        # bigram model is just learning this table. given a word what the probablity of the next word. No context is considered.
        # Now instead of embedding layer directly, we fix the size cause vocab is gonna be bigger. And to get the logits now it needs to go through a linear layer

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # In bigram model the probability of the next word is captured in the table. Hence just simply look up that value with the index of next word.
        # C = Channel = embedding size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,N_EMBD) N_EMBD = C
        # logits are the name given to the representation of the set of predicted possible characters for the next character in the 65x65 table.
        # logits usually refer to the raw output of a neural network's final layer before the application of an activation function to produce probabilities.

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # cross entropy expects it in different shape.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # cross entropy is just log likelihood loss.
            loss = F.cross_entropy(logits, targets)
            # internal converts logits to probabilities for the next character and applies the loss function.
            # Next, the loss is computed as the negative log-likelihood loss. For each token in the batch, the cross-entropy loss measures the difference between the predicted probabilities and the true next word's one-hot encoded representation (the target label).
            # The negative log-likelihood loss is mathematically equivalent to multiplying the one-hot encoded target label with the logarithm of the corresponding probability for the true next word.
            # By minimizing the cross-entropy loss, the model learns to improve its predictions and generate more accurate sequences of words.

            # In summary, the provided code snippet reshapes the logits and targets to match the expected input shapes for the F.cross_entropy function. Inside the function, the logits are converted to probabilities using softmax, and the cross-entropy loss is calculated as the negative log-likelihood between the predicted probabilities and the true one-hot encoded target labels.

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Job is to given the batch and some T generate the time dimension for the specified set of time (tokens)
        """
        # idx is (B, T) arrary of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(idx, targets=None)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # smaple from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# Transformer Based Model Defination
class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()

        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        # why is pytorch syntax like this here? I think this is how you declare a lower triangular matrix

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # compute the attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # prevents communication of past to future
        wei = F.softmax(wei, dim=-1)  # convert logits to probabilities

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.multi_head = nn.ModuleList([Head(N_EMBD // n_head) for i in range(n_head)])

    def forward(self, x):
        out = [head.forward(x) for _, head in enumerate(self.multi_head)]
        out = torch.cat(out, dim=-1)
        return out


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        # we are encoding not just the identity of these token but also their position. hence a position layer.
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.sa_head = MultiHeadSelfAttention(N_HEAD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

        # embedding is a thin wrapper around the tensor which creates vocab_size, vocab_size tensor.

        # Creating an embedding layer for 10 unique items, each represented by 5-dimensional embeddings
        # embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=5)

        # In the above example, you could indeed use a regular tensor of size (10, 5) to represent embeddings for 10 unique items, and there would not be a functional difference in this specific scenario. However, using the nn.Embedding layer has certain advantages and is more common in practice when working with embeddings for categorical data. Let's explore the reasons:
        # Memory Efficiency: When dealing with large datasets or vocabularies, using an nn.Embedding layer can be more memory-efficient. The embedding layer only stores the embeddings for the unique items (based on the number of unique indices), while a regular tensor of size (10, 5) would allocate memory for all elements, including those that might not correspond to any actual item.
        # Computational Efficiency: The nn.Embedding layer has optimized implementations for efficient indexing and lookup operations. When you pass a batch of input indices to the embedding layer, it efficiently retrieves the corresponding embeddings. This optimized implementation is especially beneficial when working with large-scale models and datasets.
        # Flexibility: The nn.Embedding layer is designed to be integrated seamlessly with other layers in PyTorch's neural network modules (nn.Module). It allows you to easily train and update the embeddings during the learning process, making it convenient for end-to-end training.
        # Integration with Embedding Lookup: When using nn.Embedding, you can utilize the torch.nn.functional.embedding function, which provides efficient embedding lookup capabilities. This function is particularly useful when you need to perform lookups across multiple indices simultaneously, such as in recurrent neural networks (RNNs) or transformer models.
        # However, if your use case involves a small number of unique items, and memory or computational efficiency is not a concern, you can still represent embeddings using regular tensors. The decision between using a regular tensor and an nn.Embedding layer depends on the specific requirements of your model, the size of your vocabulary, and your computational resources. For larger-scale applications, nn.Embedding is generally preferred due to its efficiency and convenience.
        # bigram model is just learning this table. given a word what the probablity of the next word. No context is considered.
        # Now instead of embedding layer directly, we fix the size cause vocab is gonna be bigger. And to get the logits now it needs to go through a linear layer

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # In bigram model the probability of the next word is captured in the table. Hence just simply look up that value with the index of next word.
        # C = Channel = embedding size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,N_EMBD) N_EMBD = C
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T,C)

        x = (
            tok_emb + pos_emb
        )  # (B, T, C) holds not just token identity but also the position.
        x = self.sa_head.forward(x)  # apply one head of attention (B, T, C)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)

        # logits are the name given to the representation of the set of predicted possible characters for the next character in the 65x65 table.
        # logits usually refer to the raw output of a neural network's final layer before the application of an activation function to produce probabilities.

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # cross entropy expects it in different shape.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # cross entropy is just log likelihood loss.
            loss = F.cross_entropy(logits, targets)
            # internal converts logits to probabilities for the next character and applies the loss function.
            # Next, the loss is computed as the negative log-likelihood loss. For each token in the batch, the cross-entropy loss measures the difference between the predicted probabilities and the true next word's one-hot encoded representation (the target label).
            # The negative log-likelihood loss is mathematically equivalent to multiplying the one-hot encoded target label with the logarithm of the corresponding probability for the true next word.
            # By minimizing the cross-entropy loss, the model learns to improve its predictions and generate more accurate sequences of words.

            # In summary, the provided code snippet reshapes the logits and targets to match the expected input shapes for the F.cross_entropy function. Inside the function, the logits are converted to probabilities using softmax, and the cross-entropy loss is calculated as the negative log-likelihood between the predicted probabilities and the true one-hot encoded target labels.

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Job is to given the batch and some T generate the time dimension for the specified set of time (tokens)
        """
        # idx is (B, T) arrary of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]

            # get the predictions
            logits, loss = self.forward(idx_cond, targets=None)

            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # smaple from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


bigram_model = BigramLanguageModel()
single_head_self_attention = LanguageModel()

model = single_head_self_attention
# move model parameters to device
m = model.to(device=DEVICE)

# create a Pytorch optimizer
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)


def main():
    # Train loop
    for iter in range(MAX_ITERS):
        # every once in a while evaluate the loss on train and val sets
        if iter % EVALUATION_ITERATIONS == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f} : val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = m.forward(idx=xb, targets=yb)
        optimizer.zero_grad(
            set_to_none=True
        )  # we are zeroing out all the gradients from the previous step
        loss.backward()  # getting the gradients for all the parameters
        optimizer.step()  # using those gradients to update our parameters

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode((m.generate(idx=context, max_new_tokens=500).tolist()[0])))


if __name__ == "__main__":
    main()

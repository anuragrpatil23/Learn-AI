from dataclasses import dataclass, asdict
import blobfile as bf

import torch.nn as nn
import torch.nn.functional as F
import torch as torch


@dataclass
class Config:
    enc_input_size: int
    enc_hidden_size: int
    dec_hidden_size: int
    dec_input_size: int
    activation: str
    top_k: int
    gpt2_layer_location: str
    gpt2_layer_index: int


class Encoder(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.linear = nn.Linear(config["enc_input_size"], config["enc_hidden_size"], bias=False)
        self.b_pre = nn.Parameter(torch.zeros(config["enc_input_size"]))  # b_pre as a learnable parameter
        self.b_enc = nn.Parameter(torch.zeros(config["enc_hidden_size"]))  # b_enc as a seperate learnable parameter instead of bias=True in self.linear, cause later on with topk activation this term is set to zero. hence for easy handling keeping it as a seperate parameter.

    def forward(self, x):
        x_adj = x - self.b_pre
        x = self.linear(x_adj) + self.b_enc
        return x
    

# Define a mapping from strings to activation functions
activation_map = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
    'softmax': nn.Softmax,
    'none': nn.Identity,  # Use nn.Identity for no activation
}

class Activation(nn.Module):
    
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config

            assert config["activation"] in activation_map, f"Activation function {config['activation']} not found in activation_map"

            self.postact_fn = activation_map[config["activation"]]()
    
        def forward(self, x):
            x = self.postact_fn(x)
            
            top_activations = torch.topk(
                x, largest=True, sorted=True, k=self.config["top_k"]
            )
            top_act_idx = top_activations.indices
            top_act_values = top_activations.values

            x_zeroed = torch.zeros_like(x)
            x_zeroed.scatter_(1, top_act_idx, top_act_values)

            
            return x, x_zeroed
        

class Decoder(nn.Module):

    def __init__(self, config, b_pre=None) -> None:
        super().__init__()
        self.config = config

        self.linear = nn.Linear(config["dec_input_size"],config["dec_hidden_size"], bias=False)

        if b_pre is not None:
            # Ensure that b_pre is padded to match the size of dec_hidden_size
            padding_size = config["dec_input_size"] - b_pre.size(0)

            assert config["dec_input_size"]>b_pre, "Autoencoder currently supports only latent dimension greater than input latent dimension"
            
            if padding_size > 0:
                # Pad with zeros on the right side (end of the tensor)
                self.b_pre = torch.cat([b_pre, torch.zeros(padding_size)], dim=0)        
        else:
            self.b_pre = torch.zeros(config["dec_input_size"])

    def forward(self, x):  
        x_adj = x - self.b_pre
     
        x = self.linear(x_adj)
        return x


class SparseAutoencoder(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict({
            "encoder": Encoder(config),
            "activation": Activation(config),
            "decoder": Decoder(config)  
        })

        padding_size = config["dec_input_size"] - self.model["encoder"].b_pre.size(0)

        self.model["decoder"].b_pre = torch.cat([self.model["encoder"].b_pre, torch.zeros(padding_size)], dim=0) #share the bias between encoder and decoder
        self.model["decoder"].b_pre



    def forward(self, x):
        x = self.model["encoder"](x)
        x, x_zeroed = self.model["activation"](x)
        x = self.model["decoder"](x_zeroed)
        return x    

    def encode(self, x):
        x = self.model["encoder"](x)
        x, x_zeroed = self.model["activation"](x)
        return x_zeroed
    
    def decode(self, x):
        x = self.model["decoder"](x)
        return x    
    

    @classmethod
    def load_from_pretrained(cls, config):
        
        # load openai sparese autoencoder weights
        import sparse_autoencoder

        with bf.BlobFile(sparse_autoencoder.paths.v5_32k(config["gpt2_layer_location"], config["gpt2_layer_index"]), mode="rb") as f:
            openai_sd = torch.load(f)

            # Initialize the model
            model = SparseAutoencoder(config)
            sd = model.state_dict()

            with torch.no_grad():
                sd['model.encoder.linear.weight'].copy_(openai_sd["encoder.weight"])
                sd['model.decoder.linear.weight'].copy_(openai_sd["decoder.weight"])
                sd['model.encoder.b_pre'].copy_(openai_sd["pre_bias"])
                # padding_size = config["dec_input_size"] - openai_sd["pre_bias"].size(0)
                # sd['model.decoder.b_pre'].copy_(torch.cat([openai_sd["pre_bias"], torch.zeros(padding_size)], dim=0))
            
        return model
            



# For now get activations with transformer lens
# Eventually we will use nanogpt for this.
 
#attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"  
device = "cpu"
print("Using device: ", device)

import transformer_lens
# Extract neuron activations with transformer_lens
model_ht = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
model_ht.to(device)

sd_ht = model_ht.state_dict()
for k, v in sd_ht.items():
    print(k, v.shape)

print(model_ht)
print(next(model_ht.parameters()))

# Get activations for a an example prompt
prompt = "Hi, I am a language model, "
tokens = model_ht.to_tokens(prompt)  # (1, n_tokens)
tokens = tokens.to(device)

with torch.no_grad():
    logits, activation_cache = model_ht.run_with_cache(tokens, remove_batch_dim=True)

print(logits.size())

config = Config(
    enc_input_size=768,
    enc_hidden_size=32768,
    dec_input_size=32768,
    dec_hidden_size=768, 
    activation="relu",
    top_k=32,
    gpt2_layer_location="resid_post_mlp",
    gpt2_layer_index=6
)

layer_index = config.gpt2_layer_index
location = config.gpt2_layer_location

transformer_lens_loc = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}[location]

print("transformer_lens_loc :", transformer_lens_loc)

# pass the activations to the sparse autoencoder model

model = SparseAutoencoder(asdict(config))
model = SparseAutoencoder.load_from_pretrained(asdict(config))
model.to(device)
print(model)
#print the model state dict
sd_oa = model.state_dict()

for k, v in sd_oa.items():
    if type(v) == torch.Tensor:
        print(k, v.shape)
    else:
        print(k, v)


input_tensor = activation_cache[transformer_lens_loc].to(device)

print("input tensor ", input_tensor.shape)

with torch.no_grad():
    latent_activations = model.encode(input_tensor)
    print("latent_activations: ", latent_activations)
    print("info: ")
    reconstructed_activations = model.decode(latent_activations)
    print("reconstructed_activations: ", reconstructed_activations)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)

print(location, normalized_mse)


# passing through the open ai model for reference values. 
print("passing through the open ai model for reference values")
import sparse_autoencoder

with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)

with torch.no_grad():
    latent_activations, info = autoencoder.encode(input_tensor)
    print("latent_activations: ", latent_activations)
    print("info: ",info)
    reconstructed_activations = autoencoder.decode(latent_activations, info)
    print("reconstructed_activations: ", reconstructed_activations)

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
print(location, normalized_mse)
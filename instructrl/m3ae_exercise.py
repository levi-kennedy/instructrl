import sys

sys.path.append('/content/instructrl/instructrl/models')
sys.path.append('/content/instructrl/instructrl/models/m3ae')

from functools import partial
import pickle
from PIL import Image    
from numpy import asarray
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import transformers
import einops
from models.m3ae.model import MaskedMultimodalAutoencoder, merge_patches
from models.m3ae.utils import set_random_seed
from models.m3ae.jax_utils import JaxRNG
from jax import random


#from .utils import get_1d_sincos_pos_embed


# Function to load pretrained model and parameters into the m3ae model and encode an image and text
# instruction into an image-text embedding vector and then decode the image-text embedding vector back
# into an image and text instruction.
def encode_decode_image_text(img_path, text, num_timestep=1):    

    # Load image and convert to JAX numpy array
    image = asarray(Image.open(img_path))
    image = jnp.array(image)
    image = (image / 255.0).astype(np.float32)
    image = jnp.reshape(
        image, (-1,) + image.shape[-3:]
    )

    patch_dim = 16
    patchify = lambda x: einops.rearrange(
    x,
    "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
    p1=patch_dim,
    p2=patch_dim,
    )

    patch = patchify(image)

    # emb_dim = 64
    # image_text_input = nn.Dense(emb_dim)

    # transfer_type = 'm3ae_vit_b16'
    # model_type    = 'vit_base'

    # model_name = transfer_type.split("_", 1)[1]
    text_vocab_size = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased").vocab_size

    

    #tokenized_caption = jnp.tile(text, (patch.shape[0], 1))

    tokenizer = partial(
        transformers.BertTokenizer.from_pretrained("bert-base-uncased"),
        truncation=True,
        return_tensors="np",
        add_special_tokens=False,
    )
    tokenized_text = tokenizer(text)["input_ids"].astype(np.long)
    
    file = open("/content/drive/MyDrive/research/m3ae/m3ae_base.pkl", "rb")
    data = pickle.load(file)
    pt_params = data["state"].params

    text_padding_mask = jnp.ones_like(tokenized_text)
    
    

    config_m3ae = MaskedMultimodalAutoencoder.get_default_config()
    pt_model = MaskedMultimodalAutoencoder(
        config_m3ae,
        text_vocab_size=text_vocab_size
    )

    image_output, text_output, image_mask, text_mask = pt_model.apply(
    pt_params,
    patch,
    tokenized_text,
    text_padding_mask,
    rngs={"noise": jax.random.PRNGKey(24)},
    deterministic=True,
    )

    

    return image_output, text_output



if __name__ == "__main__":
    img_path = "/content/drive/MyDrive/research/boat_img.jpg"
    text = "A beautiful day to go water skiing"

    image_output, text_output = encode_decode_image_text(img_path, text, num_timestep=1)

    image_output = merge_patches(image_output, 16)
    lk = Image.fromarray(image_output[1:].astype(np))

    # Convert tokenized text back to string
    text_output = tokenizer.decode(text_output[0])
    
    # write the image to disk
    
    image_output.save("/content/drive/MyDrive/research/boat_img_out.jpg")

    # get the last three dimensions of the image output
    


    

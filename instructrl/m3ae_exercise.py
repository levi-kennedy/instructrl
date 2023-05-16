import sys


sys.path.append('/content/instructrl/instructrl/models')
sys.path.append('/content/instructrl/instructrl/models/m3ae')
import pickle
from PIL import Image    
from numpy import asarray
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import transformers
import einops
from m3ae.model import MaskedMultimodalAutoencoder
#from .utils import get_1d_sincos_pos_embed


# Function to load pretrained model and parameters into the m3ae model and encode an image and text
# instruction into an image-text embedding vector and then decode the image-text embedding vector back
# into an image and text instruction.
def EncodeDecodeImageText(img_path, text, num_timestep=1):

    

    image = asarray(Image.open(img_path))
    image = jnp.array(image)
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

    emb_dim = 64
    image_text_input = nn.Dense(emb_dim)

    transfer_type = 'm3ae_vit_b16'
    # model_type    = 'vit_base'

    model_name = transfer_type.split("_", 1)[1]
    text_vocab_size = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased").vocab_size

    pt_model = MaskedMultimodalAutoencoder(
        text_vocab_size=text_vocab_size
    )

    tokenized_caption = jnp.tile(text, (patch.shape[0], 1))


    file = open("/content/drive/MyDrive/research/m3ae/m3ae_small.pkl", "rb")
    data = pickle.load(file)
    pt_params = data["state"].params

    text_padding_mask = jnp.ones((1, 1, 1, 1))
    
    image_output, text_output, image_mask, text_mask = pt_model.apply(
    pt_params,
    patch,
    tokenized_caption,
    deterministic=True,
    )

    # #image_text_emb = concat_multiple_image_emb(image_text_emb)
    # image_text_emb = jax.lax.stop_gradient(image_text_emb)

    # image_text_emb = nn.tanh(image_text_input(image_text_emb, axis=-1))
    # image_text_emb = image_text_emb + get_1d_sincos_pos_embed(
    # image_text_emb.shape[-1], num_timestep
    # )

    return image_output, text_output



if __name__ == "__main__":
    img_path = "/content/drive/MyDrive/research/boat_img.jpg"
    text = "A beautiful day to go water skiing"
    image_output, text_output = EncodeDecodeImageText(img_path, text, num_timestep=1)





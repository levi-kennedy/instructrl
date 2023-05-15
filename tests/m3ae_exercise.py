from PIL import Image    
from numpy import asarray
import jax
import jax.numpy as jnp
import numpy as np
from flax import nn
import transformers
import einops

from instructrl.models.m3ae.model import MaskedMultimodalAutoencoder, load_m3ae_model_vars
from instructrl.utils import get_1d_sincos_pos_embed


# Function to load pretrained model and parameters into the m3ae model and encode an image and text
# instruction into an image-text embedding vector and then decode the image-text embedding vector back
# into an image and text instruction.
def EncodeDecodeImageText(img_path, text, num_timestep=1):

    image = asarray(Image.open(img_path))
    image = jnp.array(image)
    image = (image / 255.0).astype(np.float32)

    num_image  = 1
    batch_size = 1


   

    transfer_type = 'm3ae_vit_b16'
    model_type    = 'vit_base'

    model_name = transfer_type.split("_", 1)[1]
    text_vocab_size = transformers.BertTokenizer.from_pretrained(
    "bert-base-uncased").vocab_size

    pt_model = MaskedMultimodalAutoencoder(
    text_vocab_size=text_vocab_size
    )

    emb_dim = 64
    pt_params = load_m3ae_model_vars(model_name)
    image_text_input = nn.Dense(emb_dim)

    patch_dim = 16
    patchify = lambda x: einops.rearrange(
    x,
    "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
    p1=patch_dim,
    p2=patch_dim,
    )

    tokenized_caption = jnp.tile(text, (patch.shape[0], 1))
    patch = patchify(image)

    text_padding_mask = jnp.ones_like(tokenized_caption)
    
    image_text_emb = pt_model.apply(
    pt_params,
    patch,
    tokenized_caption,
    text_padding_mask,
    method=pt_model.forward_representation,
    deterministic=True,
    )

    #image_text_emb = concat_multiple_image_emb(image_text_emb)
    image_text_emb = jax.lax.stop_gradient(image_text_emb)

    image_text_emb = nn.tanh(image_text_input(image_text_emb, axis=-1))
    image_text_emb = image_text_emb + get_1d_sincos_pos_embed(
    image_text_emb.shape[-1], num_timestep
    )

    return image_proc, text_proc, image_text_emb



if __name__ == "__main__":
    img_path = "/content/drive/MyDrive/research/boat_img.jpg"
    text = "A beatiful day to go water skiing"
    image_proc, text_proc, image_text_emb = EncodeDecodeImageText(img_path, text, num_timestep=1)





import flax
from flax import linen as nn
from flax.training import checkpoints
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import clip
from moviepy.editor import ImageSequenceClip
import pickle
import torch

class ResNetBlock(nn.Module):
  """ResNet pre-Activation block. https://arxiv.org/pdf/1603.05027.pdf"""
  features: int
  stride: int = 1

  def setup(self):
    self.conv0 = nn.Conv(self.features // 4, (1, 1), (self.stride, self.stride))
    self.conv1 = nn.Conv(self.features // 4, (3, 3))
    self.conv2 = nn.Conv(self.features, (1, 1))
    self.conv3 = nn.Conv(self.features, (1, 1), (self.stride, self.stride))

  def __call__(self, x):
    y = self.conv0(nn.relu(x))
    y = self.conv1(nn.relu(y))
    y = self.conv2(nn.relu(y))
    if x.shape != y.shape:
      x = self.conv3(nn.relu(x))
    return x + y


class UpSample(nn.Module):
  """Simple 2D 2x bilinear upsample."""

  def __call__(self, x):
    B, H, W, C = x.shape
    new_shape = (B, H * 2, W * 2, C)
    return jax.image.resize(x, new_shape, 'bilinear')


class ResNet(nn.Module):
  """Hourglass 53-layer ResNet with 8-stride."""
  out_dim: int

  def setup(self):
    self.dense0 = nn.Dense(8)

    self.conv0 = nn.Conv(64, (3, 3), (1, 1))
    self.block0 = ResNetBlock(64)
    self.block1 = ResNetBlock(64)
    self.block2 = ResNetBlock(128, stride=2)
    self.block3 = ResNetBlock(128)
    self.block4 = ResNetBlock(256, stride=2)
    self.block5 = ResNetBlock(256)
    self.block6 = ResNetBlock(512, stride=2)
    self.block7 = ResNetBlock(512)

    self.block8 = ResNetBlock(256)
    self.block9 = ResNetBlock(256)
    self.upsample0 = UpSample()
    self.block10 = ResNetBlock(128)
    self.block11 = ResNetBlock(128)
    self.upsample1 = UpSample()
    self.block12 = ResNetBlock(64)
    self.block13 = ResNetBlock(64)
    self.upsample2 = UpSample()
    self.block14 = ResNetBlock(16)
    self.block15 = ResNetBlock(16)
    self.conv1 = nn.Conv(self.out_dim, (3, 3), (1, 1))

  def __call__(self, x, text):

    # # Project and concatenate CLIP features (early fusion).
    # text = self.dense0(text)
    # text = jnp.expand_dims(text, axis=(1, 2))
    # text = jnp.broadcast_to(text, x.shape[:3] + (8,))
    # x = jnp.concatenate((x, text), axis=-1)

    x = self.conv0(x)
    x = self.block0(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)

    # Concatenate CLIP features (mid-fusion).
    text = jnp.expand_dims(text, axis=(1, 2))
    text = jnp.broadcast_to(text, x.shape)
    x = jnp.concatenate((x, text), axis=-1)

    x = self.block8(x)
    x = self.block9(x)
    x = self.upsample0(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.upsample1(x)
    x = self.block12(x)
    x = self.block13(x)
    x = self.upsample2(x)
    x = self.block14(x)
    x = self.block15(x)
    x = self.conv1(x)
    return x


class TransporterNets(nn.Module):
  """TransporterNet with 3 ResNets (translation only)."""

  def setup(self):
    # Picking affordances.
    self.pick_net = ResNet(1)

    # Pick-conditioned placing affordances.
    self.q_net = ResNet(3)  # Query (crop around pick location).
    self.k_net = ResNet(3)  # Key (place features).
    self.crop_size = 64
    self.crop_conv = nn.Conv(features=1, kernel_size=(self.crop_size, self.crop_size), use_bias=False, dtype=jnp.float32, padding='SAME')

  def __call__(self, x, text, p=None, train=True):
    B, H, W, C = x.shape
    pick_out = self.pick_net(x, text)  # (B, H, W, 1)

    # Get key features.
    k = self.k_net(x, text)

    # Add 0-padding before cropping.
    h = self.crop_size // 2
    x_crop = jnp.pad(x, [(0, 0), (h, h), (h, h), (0, 0)], 'maximum')

    # Get query features and convolve them over key features.
    place_out = jnp.zeros((0, H, W, 1), jnp.float32)
    for b in range(B):

      # Get coordinates at center of crop.
      if p is None:
        pick_out_b = pick_out[b, ...]  # (H, W, 1)
        pick_out_b = pick_out_b.flatten()  # (H * W,)
        amax_i = jnp.argmax(pick_out_b)
        v, u = jnp.unravel_index(amax_i, (H, W))
      else:
        v, u = p[b, :]

      # Get query crop.
      x_crop_b = jax.lax.dynamic_slice(x_crop, (b, v, u, 0), (1, self.crop_size, self.crop_size, x_crop.shape[3]))
      # x_crop_b = x_crop[b:b+1, v:(v + self.crop_size), u:(u + self.crop_size), ...]

      # Convolve q (query) across k (key).
      q = self.q_net(x_crop_b, text[b:b+1, :])  # (1, H, W, 3)
      q = jnp.transpose(q, (1, 2, 3, 0))  # (H, W, 3, 1)
      place_out_b = self.crop_conv.apply({'params': {'kernel': q}}, k[b:b+1, ...])  # (1, H, W, 1)
      scale = 1 / (self.crop_size * self.crop_size)  # For higher softmax temperatures.
      place_out_b *= scale
      place_out = jnp.concatenate((place_out, place_out_b), axis=0)

    return pick_out, place_out


def n_params(params):
  return jnp.sum(jnp.int32([n_params(v) if isinstance(v, dict) or isinstance(v, flax.core.frozen_dict.FrozenDict) else np.prod(v.shape) for v in params.values()]))

@jax.jit
def train_step(optimizer, batch):
  def loss_fn(params):
    batch_size = batch['img'].shape[0]
    pick_logits, place_logits = TransporterNets().apply({'params': params}, batch['img'], batch['text'], batch['pick_yx'])

    # InfoNCE pick loss.
    pick_logits = pick_logits.reshape(batch_size, -1)
    pick_onehot = batch['pick_onehot'].reshape(batch_size, -1)
    pick_loss = jnp.mean(optax.softmax_cross_entropy(logits=pick_logits, labels=pick_onehot), axis=0)

    # InfoNCE place loss.
    place_logits = place_logits.reshape(batch_size, -1)
    place_onehot = batch['place_onehot'].reshape(batch_size, -1)
    place_loss = jnp.mean(optax.softmax_cross_entropy(logits=place_logits, labels=place_onehot), axis=0)
    
    loss = pick_loss + place_loss
    return loss, (pick_logits, place_logits)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, grad, logits

@jax.jit
def eval_step(params, batch):
  pick_logits, place_logits = TransporterNets().apply({'params': params}, batch['img'], batch['text'])
  return pick_logits, place_logits

class Cliport:
  def __init__(self, clip_model):
    self.clip_model = clip_model
    self.clip_model.cuda().eval()
    # Coordinate map (i.e. position encoding).
    coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
    self.coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
    init_text = jnp.ones((4, 512), jnp.float32)
    init_pix = jnp.zeros((4, 2), np.int32)
    init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
    print(f'Model parameters: {n_params(init_params):,}')
    optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)
    ckpt_path = f'ckpt_{40000}'
    self.optim = checkpoints.restore_checkpoint(ckpt_path, optim)
    print('Loaded:', ckpt_path)
  
  def run_cliport(self, obs, text, env):
    before = env.get_camera_image()
    prev_obs = obs['image'].copy()

    # Tokenize text and get CLIP features.
    text_tokens = clip.tokenize(text).cuda()
    with torch.no_grad():
      text_feats = self.clip_model.encode_text(text_tokens).float()
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    text_feats = np.float32(text_feats.cpu())

    # Normalize image and add batch dimension.
    img = obs['image'][None, ...] / 255
    img = np.concatenate((img, self.coords[None, ...]), axis=3)

    # Run Transporter Nets to get pick and place heatmaps.
    batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
    pick_map, place_map = eval_step(self.optim.target, batch)
    pick_map, place_map = np.float32(pick_map), np.float32(place_map)

    # Get pick position.
    pick_max = np.argmax(np.float32(pick_map)).squeeze()
    pick_yx = (pick_max // 224, pick_max % 224)
    pick_yx = np.clip(pick_yx, 20, 204)
    pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

    # Get place position.
    place_max = np.argmax(np.float32(place_map)).squeeze()
    place_yx = (place_max // 224, place_max % 224)
    place_yx = np.clip(place_yx, 20, 204)
    place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

    # Step environment.
    act = {'pick': pick_xyz, 'place': place_xyz}
    obs, _, _, _ = env.step(act)

    # Show pick and place action.
    plt.title(text)
    plt.imshow(prev_obs)
    plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
    plt.show()

    # Show debug plots.
    plt.subplot(1, 2, 1)
    plt.title('Pick Heatmap')
    plt.imshow(pick_map.reshape(224, 224))
    plt.subplot(1, 2, 2)
    plt.title('Place Heatmap')
    plt.imshow(place_map.reshape(224, 224))
    plt.show()

    # Show video of environment rollout.
    debug_clip = ImageSequenceClip(env.cache_video, fps=25)
    debug_clip.write_videofile("output.mp4", fps=25) 

    # Show camera image after pick and place.
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    plt.title('After')
    after = env.get_camera_image()
    plt.imshow(after)
    plt.show()

    # return pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx
    return obs

if __name__ == "__main__":
  from scene import PickPlaceEnv
  dataset = pickle.load(open('dataset-9999.pkl', 'rb'))  # ~10K samples.
  dataset_size = len(dataset['text'])
  
  config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}
  np.random.seed(42)
  env = PickPlaceEnv()
  obs = env.reset(config)
  img = env.get_camera_image()
  plt.imshow(img)
  plt.show()
  
  user_input = 'Pick the yellow block and place it on the blue bowl.'
  clip_model, _ = clip.load("ViT-B/32")
  cliport = Cliport(clip_model=clip_model) 
  obs = cliport.run_cliport(obs, user_input, env)


# Keras implementation of CycleGAN

[CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf)  
[Reference implementation](https://junyanz.github.io/CycleGAN/)

Fork of https://github.com/simontomaskarlsson/CycleGAN-Keras.  
Changes after forking:

- Migrated from `keras` and `keras_contrib` to TensorFlow 2
  (`tensorflow.keras` and `tensorflow_addons`).
- Changed config, IO, control, module structure.
- Added the possibility to first train an autoencoder and then
  train CycleGAN on the latent space in order to have the chance to learn a more
  abstract concept transfer rather than pure style transfer.

TODO:
- Add autoencoder README.
- Add model architecture loading for image generation.
- Add resume training.
- Add TensorBoard callbacks.


## Setup project

```
# Install `virtualenv` to manage environment.
pip3 install virtualenv

# Create and activate environment.
virtualenv <ENV_NAME> --python /path/to/python>=3.7
source <ENV_NAME>/bin/activate
# Windows: <ENV_NAME>\Scripts\activate

# Install libraries.
pip install -r requirements.txt

# Install project modules as libraries in order to make them
# available in modules which do not reside in the project root folder.
pip install -e .

# Optional: install graphviz for model inspection, e.g. on Debian/Ubuntu
# (see https://graphviz.gitlab.io for a different OS)
sudo apt-get install graphviz

# Optional: install environment as kernel for Jupyter (lab).
ipython kernel install --name <KERNEL_NAME> --user

# Deactivate environment in order to use system python or work on a different
# project. Activate it again as above in case you want to continue working
# on this project.
deactivate
```
  
  
## Setup data and model configuration

Place your images in  
`data/source_images/<DATASET_NAME>/{train_A, train_B, test_A, test_B}`.

Place a config in `data/configs/<CONFIG_NAME>.json`.  
Below is an example with comments, which you should remove because they are not valid JSON.
This example reflects the hyperparameters used in the CycleGAN paper.
```
{
    # The subfolder of your dataset, as indicated above.
    "dataset_name": "<DATASET_NAME>",
    # Image height, width, number of channels.
    "image_shape": [256, 256, 3],
    # Use a generator for loading data, rather than loading all images into RAM.
    "use_data_generator": false,
    "epochs": 200,
    "batch_size": 1,
    # Save examples each nth epoch
    "save_interval_examples": 1,
    # Save temporary examples each nth batch
    "save_interval_temporary_examples": 20,
    # Save model each nth epoch
    "save_interval_model": 5,
    # Normalization layer, currently only "instance_normalization".
    "normalization": "instance_normalization",
    # Cyclic loss weight A -> B -> A
    "lambda_ABA": 10.0,
    # Cyclic loss weight B -> A -> B
    "lambda_BAB": 10.0,
    # Weight for loss from discriminator guess on synthetic images
    "lambda_D": 1.0,
    "learning_rate_D": 2e-4,
    "learning_rate_G": 2e-4,
    # Training iterations in each training loop (on a single batch)
    "generator_iterations": 1,
    "discriminator_iterations": 1,
    "adam_beta_1": 0.5,
    "adam_beta_2": 0.999,
    # Track this many image examples during training (per fold and domain)
    "num_examples_to_track": 3,
    # Size of image pools used to update the discriminators
    "synthetic_pool_size": 50,
    # Linear decay of learning rate, for both discriminators and generators
    "use_linear_lr_decay": true,
    "linear_lr_decay_epoch_start": 101,
    # Identity learning: teach G_A2B to be close to the identity for B images (and v.v.)
    # Helps to preserve color of inputs.
    "use_identity_learning": false,
    # Identity learning for each nth batch
    "identity_learning_modulus": 10,
    # If False the discriminator learning rate should be decreased
    "use_patchgan_discriminator": true,
    # If True the generators have an extra encoding/decoding step to match
    # discriminator information access
    "use_multiscale_discriminator": false,
    # Resize convolution - instead of transpose convolution in deconvolution
    # layers (uk) - can reduce checkerboard artifacts but the blurring might
    # affect the cycle-consistency
    "use_resize_convolution": false,
    # Add MAE between B input and G_A2B (and v.v.) to training loss
    "use_supervised_learning": false,
    "supervised_learning_weight": 10.0,
    # Use e.g. 0.9 to avoid training the discriminators to zero loss
    "real_discriminator_label": 1.0,
    # Use only the first n images from the dataset (use all if None)
    "num_train_A_images": null,
    "num_train_B_images": null,
    "num_test_A_images": null,
    "num_test_B_images": null
}
```
  
  
## Train CycleGAN

Train a model with  
```
python models/cyclegan/train.py --config-path <CONFIG_NAME>.json
```
See `python models/cyclegan/train.py --help` for additional flags.

Results of model training are stored at various intervals in `data/results/<MODEL_KEY>`  
where `<MODEL_KEY>` is the timestamp at the start of training.  
The results stored are
- training configuration and metadata in `data/results/<MODEL_KEY>/meta_data.json`
- training losses in `data/results/<MODEL_KEY>/losses_history.pickle`
- model architectures and weights in `data/results/<MODEL_KEY>/saved_models/`
- generated examples in `data/results/<MODEL_KEY>/examples_history/`

Because the model results are stored at regular intervals, you can interrupt  
model training with a `KeyboardInterrupt` if you are satisfied with the intermediate  
results.

Training results can be inspected with `evaluate/evaluate.ipynb`.  
  
  
## Generate images with a trained model

You can use a model with reference `<MODEL_KEY>` (the last model stored under this key)  
to generate images which will be stored in `data/results/<MODEL_KEY>/generated_synthetic_images/`.  

The image generation needs a config in the same format as the model training.  
You can use the same config as the one used in training or a different one (as long  
as the model architecture is the same).  

The generation script generates mapped images for all images in  
`data/source_images/<DATASET_NAME>/{test_A, test_B}`.  
A possible use case for a different config for image generation is using a different dataset.

After (optionally, see above) preparing the dataset and config, run
```
python models/cyclegan/generate.py \
    --config-path <CONFIG_NAME>.json \
    --model-key <MODEL_KEY>
```
See `python models/cyclegan/generate.py --help` for additional flags.

# Keras implementation of CycleGAN

[CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf)  
[Reference implementation](https://junyanz.github.io/CycleGAN/)

Fork of https://github.com/simontomaskarlsson/CycleGAN-Keras.  
Changes after forking:

- Migrated from `keras` and `keras_contrib` to TensorFlow 2 (`tensorflow.keras` and `tensorflow_addons`).
- Changed config, IO, control, module structure.

TODO:
- Refactor into submodules.
- Add autoencoder.
- Expose all configurable parameters in config.
- Implement TensorFlow tuning options.  
  

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
Example with comments, which you should remove because they are not valid JSON:  
```
{
    # The subfolder of your dataset, as indicated above.
    "source_images": "<DATASET_NAME>",

    # Image height, width, number of channels.
    "image_shape": [256, 256, 3],

    # Use a generator for loading data, rather than loading all images into RAM.
    "use_data_generator": false,

    "batch_size": 1
}
```
  
  
## Train CycleGAN

Train a model with  
```
python model.py --config-path <CONFIG_NAME>.json
```
See `python model.py --help` for additional flags.

Results of model training are stored at various intervals in `data/results/<MODEL_KEY>`  
where `<MODEL_KEY>` is the timestamp at the start of training.  
The results stored are
- training configuration and metadata in `data/results/<MODEL_KEY>/meta_data.json`
- training losses in `data/results/<MODEL_KEY>/loss_output.csv`
- model architectures and weights in `data/results/<MODEL_KEY>/saved_models/`
- generated examples in `data/results/<MODEL_KEY>/output_history_samples/`

Because the model results are stored at regular intervals, you can interrupt  
model training with a `KeyboardInterrupt` if you are satisfied with the intermediate  
results.

Training results can be inspected with `evaluate/evaluate.ipynb`.  
  
  
## Generate images with a trained model

You can use a model with reference `<MODEL_KEY>` (the last model stored under this key)  
to generate images which will be stored in `data/results/<MODEL_KEY>/generated_synthetic_images/`.  

The image generation needs a config in the same format as the model training.  
You can use the same config as the one used in training or a different one (as long  
as the image shape is the same).  

The generation script generates mapped images for all images in  
`data/source_images/<DATASET_NAME>/{test_A, test_B}`.  
A possible use case for a different config for image generation is using a different dataset.

After (optionally, see above) preparing the dataset and config, run
```
python model.py \
    --config-path <CONFIG_NAME>.json \
    --model-key <MODEL_KEY> \
    --generate-synthetic-images
```
See `python model.py --help` for additional flags.

import sys
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Assumed external utility, left as-is
from lesion_segmentation_utils import read_data

# Imports from our refactored modules
from DL_model import dense_model
from DL_metrics import dice_coef
from DL_losses import tversky_loss

def get_filepaths(data_path: str, split_num: int) -> dict:
    """
    Generates file paths for training and validation data for a given split.
    """
    base_path = f"{data_path}/{split_num:02d}"
    return {
        "train_images": sorted(glob.iglob(f"{base_path}/train/*brain.nii.gz")),
        "train_masks": sorted(glob.iglob(f"{base_path}/train/*mask.nii.gz")),
        "valid_images": sorted(glob.iglob(f"{base_path}/valid/*brain.nii.gz")),
        "valid_masks": sorted(glob.iglob(f"{base_path}/valid/*mask.nii.gz")),
    }

def determine_shape(base_img_size: tuple, slice_direction: int) -> tuple:
    """
    Determines the 2D image shape based on the slice direction.
    """
    if slice_direction == 0:
        return (base_img_size[1], base_img_size[2])  # Axial
    elif slice_direction == 1:
        return (base_img_size[0], base_img_size[2])  # Coronal
    elif slice_direction == 2:
        return (base_img_size[0], base_img_size[1])  # Sagittal
    else:
        raise ValueError(f"Invalid slice direction: {slice_direction}")

def build_model(input_shape: tuple) -> tf.keras.Model:
    """
    Builds and compiles the Keras model.
    """
    input_img = Input((*input_shape, 1))
    model = dense_model(input_img)
    
    model.compile(
        optimizer=RMSprop(momentum=0.95), 
        loss=[tversky_loss], 
        metrics=[dice_coef]
    )
    return model

def main(args):
    """
    Main training pipeline.
    """
    base_img_size = tuple(map(int, args.img_size.split(',')))
    
    print(f"--- Starting Training ---")
    print(f"Data Path: {args.data_path}")
    print(f"Base Image Size: {base_img_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")

    for sn in range(args.split_num):
        split = sn + 1
        print(f"\nProcessing Split {split}/{args.split_num}")
        
        paths = get_filepaths(args.data_path, split)
        
        # Train one model for each slice direction
        for sl_dir in range(3):
            print(f"\nTraining for slice direction: {sl_dir}")
            
            try:
                img_shape = determine_shape(base_img_size, sl_dir)
            except ValueError as e:
                print(e)
                continue

            print("Loading training data...")
            X_train, y_train = read_data(
                paths["train_images"], paths["train_masks"], base_img_size, sl_dir
            )
            print("Loading validation data...")
            X_valid, y_valid = read_data(
                paths["valid_images"], paths["valid_masks"], base_img_size, sl_dir
            )

            if X_train is None or X_valid is None:
                print(f"Failed to load data for slice direction {sl_dir}. Skipping.")
                continue

            print(f"Input Shape: {img_shape}")
            model = build_model(img_shape)
            if sl_dir == 0:  # Only print summary for the first model
                model.summary()

            model_name = f"{args.model_name_prefix}_split{split}_dir{sl_dir}.h5"
            
            callbacks = [
                EarlyStopping(
                    patience=args.patience,
                    verbose=1,
                    monitor='val_loss', # Use 'val_loss' as tversky_loss is the loss
                    mode='min',
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    model_name,
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=False,
                    monitor='val_loss',
                    mode='min'
                ),
            ]

            print(f"Starting model training for: {model_name}")
            val_idxs = np.random.permutation(X_valid.shape[0])

            model.fit(
                X_train, y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=callbacks,
                validation_data=(X_valid[val_idxs], y_valid[val_idxs])
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseLes 2D Slice Training")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to the root data directory."
    )
    parser.add_argument(
        "--model_name_prefix", 
        type=str, 
        default="denseles_model", 
        help="Prefix for saving model files."
    )
    parser.add_argument(
        "--img_size", 
        type=str, 
        default="144,256,160", 
        help="Base 3D image dimensions (H,W,D) as a comma-separated string."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=10, 
        help="Training batch size."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1000, 
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10, 
        help="Early stopping patience."
    )
    parser.add_argument(
        "--split_num", 
        type=int, 
        default=1, 
        help="Number of data splits to process."
    )

    args = parser.parse_args()
    main(args)

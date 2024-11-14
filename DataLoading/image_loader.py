import os
import tensorflow as tf
import pandas as pd

class ImageDataLoader:
    def __init__(self, root, batch_size=128, shuffle_count=1024, transform=None):
        self.root = root
        self.batch_size = batch_size
        self.shuffle_count = shuffle_count
        self.transform = transform  # Add transform parameter
    
    def __call__(self):
        dataset = self.load_dataset_from_directory(self.root)
        dataset = dataset.shuffle(self.shuffle_count).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_image_and_label(self, image_path, label):
        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

        # Apply the transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Convert label to tensor
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        return image, label

    def load_dataset_from_directory(self, root_directory):
        image_paths = []
        labels = []

        for subdir in os.listdir(root_directory):
            subdir_path = os.path.join(root_directory, subdir)
            
            if os.path.isdir(subdir_path):
                label_csv_path = os.path.join(subdir_path, self._get_csv_filename(subdir))
                label_df = pd.read_csv(label_csv_path, header=None, skiprows=1)
                
                for index, row in label_df.iterrows():
                    frame_name = row[0]
                    label = row[1:8].tolist()

                    image_path = os.path.join(subdir_path, frame_name)
                    image_path = os.path.abspath(image_path)
                    
                    image_paths.append(image_path)
                    labels.append(label)

        image_paths = tf.constant(image_paths)
        labels = tf.constant(labels, dtype=tf.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def _get_csv_filename(self, dirname: str) -> str:
        return dirname + '.csv'

    @staticmethod
    def center_crop_224x224(image):
        """
        Center crops the image to 224x224.
        """
        return tf.image.resize_with_crop_or_pad(image, target_height=224, target_width=224)

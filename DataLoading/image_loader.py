import os
import tensorflow as tf
import pandas as pd

class ImageDataLoader:
    def __init__(self, root, batch_size=32, shuffle_count=1024):
        self.root = root
        self.batch_size = batch_size
        self.shuffle_count = shuffle_count
    
    def __call__(self):
        dataset = self.load_dataset_from_directory(self.root)
        dataset = dataset.shuffle(self.shuffle_count).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset        

    def load_image_and_label(self, image_path, label):
        # Load the image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0   # FIXME: do we need it? Normalize

        # Convert label to tensor
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        return image, label

    def load_dataset_from_directory(self, root_directory):
        image_paths = []
        labels = []
        
        # Traverse each subdirectory in the root_directory
        for subdir in os.listdir(root_directory):
            subdir_path = os.path.join(root_directory, subdir)
            
            if os.path.isdir(subdir_path):
                # Read the CSV file without a header
                #FIXME: proper CSV file name from directory name
                label_csv_path = os.path.join(subdir_path, 'labels.csv')
                label_df = pd.read_csv(label_csv_path, header=None)
                
                # Go through each row in the CSV
                for index, row in label_df.iterrows():
                    frame_name = row[0]           # First column is the frame name
                    label = row[1:8].tolist()     # Columns 1 to 7 are the labels
                    
                    # Full path to the frame
                    image_path = os.path.join(subdir_path, frame_name)
                    image_path = os.path.abspath(image_path)
                    
                    # Append image path and label
                    image_paths.append(image_path)
                    labels.append(label)

        # Convert lists to tensors
        image_paths = tf.constant(image_paths)
        labels = tf.constant(labels, dtype=tf.float32)
        
        # Create a tf.data.Dataset from image paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Map the function to load images and labels
        dataset = dataset.map(self.load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

# Load the dataset
root_directory = "path_to_your_root_directory"

dataset = ImageDataLoader(root_directory)()


# # EXAMPLE
# # Define your model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(7)  # Seven outputs for the seven numerical labels
# ])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # Train the model
# model.fit(
#     dataset,
#     epochs=10
# )
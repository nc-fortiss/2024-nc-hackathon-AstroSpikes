import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

from src.dataloaders.spades import create_dataset
from src.utils.kdsnt import denormalize_pixel_coordinates

if __name__ == '__main__':
    config_path = "/home/arunkumar/dev-python/2024-nc-hackathon-AstroSpikes/configs/heatmap_dsnt.yaml"
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print("Error loading YAML:", e)

    df = pd.read_csv(cfg.paths.train_data)
    data_dict = {col: df[col].values for col in df.columns}

    vis_dataset = create_dataset(data_dict,
                                 batch_size=4,  # Let's visualize 4 images
                                 input_size=cfg.data.input_size[:2],
                                 is_training=False,
                                 heatmap=True,
                                 cache_dir=None)

    # --- Visualization Script ---
    print("\nVisualizing a batch of data to verify correctness...")

    # Get one batch of data
    for images, keypoints_batch in vis_dataset.take(1):
        # Convert tensors to numpy arrays for plotting
        images_np = images.numpy()
        keypoints_np = keypoints_batch.numpy()
        batch_size = images_np.shape[0]
        fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 5))
        if batch_size == 1:  # Ensure axes is always an array
            axes = [axes]

        print(f"Displaying {batch_size} images from the batch...")

        for i in range(batch_size):
            image = images_np[i]
            keypoints_normalized = keypoints_np[i]
            height, width, _ = image.shape
            keypoints_pixel = denormalize_pixel_coordinates(keypoints_normalized, height, width)
            ax = axes[i]
            ax.imshow(image)
            # Plot keypoints as red 'x' markers
            ax.scatter(keypoints_pixel[:, 0], keypoints_pixel[:, 1], c='b', marker='x', s=50)
            ax.set_title(f'Sample {i + 1}')
            ax.axis('off')

        plt.tight_layout()
        plt.show()

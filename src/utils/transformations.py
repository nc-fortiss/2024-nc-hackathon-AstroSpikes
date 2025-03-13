import numpy as np
from omegaconf import DictConfig
from tonic import transforms


class Transformations:
    def __init__(self, cfg: DictConfig):
        self.config = cfg
        self.img_size = cfg.data.input_size[0]

    def three_c_representation(self, events):  # working
        print(transforms)
        transform = transforms.Compose([
            transforms.MergePolarities(),
            transforms.CenterCrop(sensor_size=(1280, 720, 1), size=(720, 720)),
            transforms.Downsample(spatial_factor=self.img_size / 720),
            transforms.ToTimesurface(sensor_size=(self.img_size, self.img_size, 1), dt=333, tau=200)
        ])
        transformed_events = transform(events)
        # create frames from the transformed events
        ret = []
        for frame in range(0, (len(transformed_events) // 3) * 3, 3):
            # empty frame
            rgb_frame = np.zeros(self.config.input_shape, dtype=np.uint8)
            # stack 3 frames into 3 channels
            # scale [0,1] to [0,255]
            rgb_frame[:, :, 0] = transformed_events[frame] * 255
            rgb_frame[:, :, 1] = transformed_events[frame + 1] * 255
            rgb_frame[:, :, 2] = transformed_events[frame + 2] * 255
            ret.append(rgb_frame)
        return ret

    def to_voxel_grid(self, events):  # working
        transform = transforms.Compose([
            transforms.MergePolarities(),
            transforms.CenterCrop(sensor_size=(1280, 720, 1), size=(720, 720)),
            transforms.Downsample(spatial_factor=self.img_size / 720),
            transforms.ToVoxelGrid(sensor_size=(self.img_size, self.img_size, 2), n_time_bins=600)
        ])
        t_events = transform(events)
        ret = []
        for frame in range(0, len(t_events)):
            # empty frame
            rgb_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            # scale [0,1] to [0,255]
            rgb_frame[:, :, 0] = t_events[frame] * 255
            ret.append(rgb_frame)
        return ret

    def two_polarity_time_surface(self, events):  # working
        transform = transforms.Compose([
            transforms.MergePolarities(),
            transforms.CenterCrop(sensor_size=(1280, 720, 1), size=(720, 720)),
            transforms.Downsample(spatial_factor=self.img_size / 720),
            transforms.ToTimesurface(dt=1000, tau=200, sensor_size=(self.img_size, self.img_size, 1))
        ])

        events_positive = events[events['p'] == 1]
        events_negative = events[events['p'] == 0]
        t_positive_events = transform(events_positive)
        t_negative_events = transform(events_negative)
        ret = []

        for frame in range(0, min(len(t_negative_events), len(t_positive_events))):
            # empty frame
            rgb_frame = np.zeros(self.config.input_shape, dtype=np.uint8)
            # stack 3 frames into 3 channels
            # scale [0,1] to [0,255]
            rgb_frame[:, :, 0] = t_positive_events[frame] * 255
            rgb_frame[:, :, 1] = t_negative_events[frame] * 255
            ret.append(rgb_frame)
        return ret

    def two_d_histogram(self, events):  # working
        transform = transforms.Compose([
            transforms.MergePolarities(),
            transforms.CenterCrop(sensor_size=(1280, 720, 1), size=(720, 720)),
            transforms.Downsample(spatial_factor=self.img_size / 720),
            transforms.ToFrame(sensor_size=(self.img_size, self.img_size, 1), n_time_bins=600)
        ])
        t_events = transform(events)
        ret = []
        for frame in range(0, len(t_events)):
            # empty frame
            rgb_frame = np.zeros(self.config.input_shape, dtype=np.uint8)
            # scale [0,1] to [0,255]
            rgb_frame[:, :, 0] = t_events[frame] * 255
            ret.append(rgb_frame)
        return ret

    def lnes(self, events, intervalLength=1000):
        # Define transformation pipeline
        transform = transforms.Compose([
            transforms.CenterCrop(sensor_size=(1280, 720, 1), size=(720, 720)),
            transforms.Downsample(spatial_factor=self.img_size / 720),
        ])

        # Transform and sort events by timestamp
        t_events = transform(events)
        t_events = t_events[t_events['t'].argsort()]  # Sort events by time

        # Initialize parameters
        t_start = t_events[0]['t']  # Global start time
        t_end = t_events[-1]['t']  # Global end time
        n_time_bins = int((t_end - t_start) // intervalLength) + 1

        # Pre-allocate output array (n_time_bins, 240, 240, 3)
        ret = np.zeros((n_time_bins, self.img_size, self.img_size, self.config.data.input_size[2]), dtype=np.float32)

        # Assign events to time bins
        bin_indices = ((t_events['t'] - t_start) // intervalLength).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, n_time_bins - 1)  # Ensure bounds

        # Extract event properties and ensure integer types
        x = t_events['x'].astype(np.int32)
        y = t_events['y'].astype(np.int32)
        p = t_events['p'].astype(np.int32)  # Ensure polarity is integer
        t = t_events['t']

        # Normalize timestamp for each time bin
        normalized_t = ((t - (t_start + bin_indices * intervalLength)) / intervalLength) * 255

        # Use np.add.at to accumulate normalized values into the correct time bin and pixel location
        np.add.at(ret, (bin_indices, x, y, p), normalized_t)

        # Clip values to [0, 255] and convert to uint8
        ret = np.clip(ret, 0, 255).astype(np.uint8)

        return ret

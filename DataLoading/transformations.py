from tonic import transforms
import numpy as np

class Transformations:
    def __init__(self):
        pass

    def three_c_representation(self, events):#working
        transform = transforms.Compose([
                        transforms.MergePolarities(),
                        transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                        transforms.Downsample(spatial_factor=240/720),
                        transforms.ToTimesurface(dt=333,tau=200,sensor_size=(240,240,1))
                    ])
        transformed_events = transform(events)
        #create frames from the transformed events
        ret = []
        for frame in range(0, (len(transformed_events)//3)*3, 3):
                #empty frame
                rgb_frame = np.zeros((240,240,3), dtype=np.uint8)
                #stack 3 frames into 3 channels
                #scale [0,1] to [0,255]
                rgb_frame[:,:,0] = transformed_events[frame]*255
                rgb_frame[:,:,1] = transformed_events[frame+1]*255
                rgb_frame[:,:,2] = transformed_events[frame+2]*255
                ret.append(rgb_frame)
        return ret

    def to_voxel_grid(self, events):#working
        transform = transforms.Compose([
                        transforms.MergePolarities(),
                        transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                        transforms.Downsample(spatial_factor=240/720),
                        transforms.ToVoxelGrid(sensor_size=(240, 240, 2), n_time_bins=60)
                    ])
        t_events = transform(events)
        ret = []
        for frame in range(0,len(t_events)):
            #empty frame
            rgb_frame = np.zeros((240,240,3), dtype=np.uint8)
            #scale [0,1] to [0,255]
            rgb_frame[:,:,0] = t_events[frame]*255
            ret.append(rgb_frame)
        return ret

    def two_polarity_time_surface(self, events):#working
        transform = transforms.Compose([
                        transforms.MergePolarities(),
                        transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                        transforms.Downsample(spatial_factor=240/720),
                        transforms.ToTimesurface(dt=1000, tau=200,sensor_size=(240,240,1))
                    ])

        events_positive = events[events['p'] == 1]
        events_negative = events[events['p'] == 0]
        t_positive_events = transform(events_positive)
        t_negative_events = transform(events_negative)
        ret = []
        for frame in range(0,len(t_positive_events)):
            #empty frame
            rgb_frame = np.zeros((240,240,3), dtype=np.uint8)
            #stack 3 frames into 3 channels
            #scale [0,1] to [0,255]
            rgb_frame[:,:,0] = t_positive_events[frame]*255
            rgb_frame[:,:,1] = t_negative_events[frame]*255
            ret.append(rgb_frame)
        return ret

    def two_d_histogram(self, events):#working
        transform = transforms.Compose([
                        transforms.MergePolarities(),
                        transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                        transforms.Downsample(spatial_factor=240/720),
                        transforms.ToFrame(sensor_size=(240,240,1), n_time_bins=60)
                    ])
        t_events = transform(events)
        ret = []
        for frame in range(0,len(t_events)):
            #empty frame
            rgb_frame = np.zeros((240,240,3), dtype=np.uint8)
            #scale [0,1] to [0,255]
            rgb_frame[:,:,0] = t_events[frame]*255
            ret.append(rgb_frame)
        return ret
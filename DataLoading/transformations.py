from tonic import transforms

def three_c_representation(events):
    transform = transforms.Compose([
                    transforms.MergePolarities(),
                    transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                    transforms.Downsample(spatial_factor=240/720),
                    transforms.ToTimesurface(dt=333,tau=200,sensor_size=(240,240,1))
                ])
    t_events = transform(events)
    #create frames from the transformed events
    ret = []
    for frame in range(0, (len(t_events)//3)*3, 3):
            #empty frame
            rgb_frame = np.zeros((240,240,3), dtype=np.uint8)
            #stack 3 frames into 3 channels
            #scale [0,1] to [0,255]
            rgb_frame[:,:,0] = sample_events[frame]*255
            rgb_frame[:,:,1] = sample_events[frame+1]*255
            rgb_frame[:,:,2] = sample_events[frame+2]*255
            ret.append(rgb_frame)
    return ret

def to_voxel_grid(events):
    transform = transforms.Compose([
                    transforms.MergePolarities(),
                    transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                    transforms.Downsample(spatial_factor=240/720),
                    tonic.transforms.ToVoxelGrid(sensor_size=(240, 240), n_time_bins=60)
                ])
    t_events = transform(events)
    ret = []
    for frame in range(0,len(t_events)):
        #empty frame
        rgb_frame = np.zeros((240,240,3), dtype=np.uint8)
        #scale [0,1] to [0,255]
        rgb_frame[:,:,0] = sample_events[frame]*255
        ret.append(rgb_frame)
    return ret

def two_polarity_time_surface(events):
    transform = transforms.Compose([
                    transforms.MergePolarities(),
                    transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                    transforms.Downsample(spatial_factor=240/720),
                    transforms.ToTimesurface(dt=1000, tau=200,sensor_size=(240,240,2))
                ])
    positive_events = events[events["polarity"] == 1]
    negative_events = events[events["polarity"] == -1]
    t_positive_events = transform(positive_events)
    t_negative_events = transform(negative_events)
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

def two_d_histogram(events):
    transform = transforms.Compose([
                    transforms.MergePolarities(),
                    transforms.CenterCrop(sensor_size=(1280,720,1), size = (720,720)),
                    transforms.Downsample(spatial_factor=240/720),
                    transforms.ToFrame(sensor_size=(240,240), n_time_bins=60)
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
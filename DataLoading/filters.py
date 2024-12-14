import numpy as np

def get_distribution(self, events, boundries: list[tuple[int, bool]]=None) -> bool:
    """Filters out the traces where events are outside the center of the image
    boundries: for eg. [(280, False), (360, True)] means that the x values should be between 280 and 360"""
    if boundries is None:
        boundries = self.boundries
        
    band_edges = np.array([boundry[0] for boundry in boundries])
    band_active = np.array([boundry[1] for boundry in boundries])


    sample_events_x = np.array([smpl[1] for smpl in events])
    # only three bins
    events_x_digitize = np.digitize(sample_events_x, band_edges, right=True)
    values, counts = np.unique(events_x_digitize, return_counts=True)

    fraction_active = np.sum(counts[band_active]) / np.sum(counts)

    
    return fraction_active
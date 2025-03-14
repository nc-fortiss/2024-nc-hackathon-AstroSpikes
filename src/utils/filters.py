import numpy as np


class Filters:
    def __init__(self, boundries: list[tuple[int, bool]] = [(280, False), (360, True)]):
        self.boundries = boundries

    def get_distribution(self, events) -> bool:
        # check if events is empty
        if len(events) == 0:
            return False
        """Filters out the traces where events are outside the center of the image
        boundries: for eg. [(280, False), (360, True)] means that the x values should be between 280 and 360"""

        band_edges = np.array([boundry[0] for boundry in self.boundries])
        band_active = np.array([boundry[1] for boundry in self.boundries])

        sample_events_x = np.array([smpl[1] for smpl in events])
        # only three bins
        events_x_digitize = np.digitize(sample_events_x, band_edges, right=True)
        values, counts = np.unique(events_x_digitize, return_counts=True)

        fraction_active = np.sum(counts[band_active]) / np.sum(counts)

        return fraction_active > 0.8

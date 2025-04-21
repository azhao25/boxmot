from boxmot.trackers.boosttrack.boosttrack import BoostTrack

class BoostTrackPP(BoostTrack):
    def __init__(self, reid_weights, device, half, *args, **kwargs):
        super().__init__(reid_weights, device, half, *args, **kwargs)
        ## BoostTrack++ parameters
        self.use_rich_s = True
        self.use_sb = True
        self.use_vt = True
        self.use_reid = True
    

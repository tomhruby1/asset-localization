import sys
import toml
from pathlib import Path
import typing as T

import config

class AssetLocalizationManager:
    def __init__(self, config_path:Path):
        with open(config_path) as f:
            self.config_dict = toml.load(f)
        self.config = config.Config(self.config_dict)
        
    def run(self):
        ''' execute the consecutive sections of pipeline '''
        for tag in self.config_dict:
            if tag.lower() in ['visualization']: # visualization is a global config section not a stage
                continue
            stage_func = getattr(self, tag.lower())
            stage_conf = getattr(self.config, tag.lower())
            # call the stage function
            stage_func(stage_conf)

            print()

    ## stage handling functions
    def initialization(self, cfg:config.Initialization):
        # 
        # undistort images and copy to work destination
        print("initialized")

    def detection(self, cfg:config.Detection):
        # in: (reel) frames path
        # out: detections.json
        print("detection")

    def raycasting(self, cfg:config.Raycasting):
        # in: undistorted_reel_frames, 
        # out: 

    def prefiltering(self, cfg:config.Prefiltering):
        print("prefiltering")

    def features(self, cfg:config.Features):
        print("features")
    
    def semantic_filtering(self, cfg:config.SemanticFiltering):
        print("semantic_filter")

    def clustering(self, cfg:config.Clustering):
        print("clustering")

    


if __name__== "__main__":
    # config_path = Path(sys.argv[1])
    config_path = 'config_default.toml'
    assloc = AssetLocalizationManager(config_path)
    assloc.run()
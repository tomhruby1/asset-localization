# Asset Localization

This repository contains code for the experimental localization pipeline introduced in thesis Multicamera Traffic Sign Detection and Spatial Localization. 
![alt text](<assets/Screenshot from 2024-08-20 18-09-12.png>)


## Repository Structure

This repository is organized into the following directories:

- `config`: Contains configuration files containing the pipeline settings
- `data`: Contains 2 folder for the 2 datasets used Arbes (`aka Dataset-1`) and Drtinova (aka `Dataset-2`)

## Getting Started
1. clone this repository
2. build the conda environment
    ```
        conda env create -f environment.yml
        conda activate assdet
    ```
3. Download the traffic signs classifier weights at: `https://drive.google.com/file/d/1o-kGZcQn8jraSD8EeBgrpMf88nL5-nxL/view?usp=sharing`
4. create or modify a config toml file (see `config/latest.toml`) to setup the pipeline; 
```
    work_dir = "[your work directory]"
    checkpoint = "[downloaded classifier weights]"
    
```
5. if using one of the two prepared datasets copy `reel_undistorted` and `detections.json` into the `work_dir`, for running detection on a new dataset use `https://github.com/tomhruby1/mmdetection2` to produce new detections
6. run the pipeline using `python assloc_manager.py [config_file.toml]` 


# Repository to predict the context (out of four contexts) in the FLASH dataset
The present code implements FLASH framework. The FLASH dataset is available in our public repository [here](https://genesys-lab.org/multimodal-fusion-nextg-v2x-communications).

### Pre-requisites

- Python 3.8

- Pytorch 1.10


### Cite This paper
To use this repository, please refer to our paper: 

 `@INPROCEEDINGS{flash,title = {FLASH: \underline{F}ederated \underline{L}earning for \underline{A}utomated \underline{S}election of \underline{H}igh-band mmWave Sectors}, booktitle = {{IEEE International Conference on Computer Communications (INFOCOM)}},year = "2022", author = {B. {Salehi} and J. {Gu} and D. {Roy} and K. {Chowdhury}}, month={May}}`
 
 ### Details about the Contexts in the FLASH dataset:
 *Context1: Category1 (LOS)
 *Context1: Category2 (NLOS, pedestrian as obstacle)
  *Context1: Category1 (NLOS, static car as obstacle)
   *Context1: Category1 (LOS, moving car as obstacle)
 
### Run the Context Prediction Code:
We use a fixed seed throughout all experiments. Run the commands below to generate the seed and global test data accordingly. Remember to change to base path to your own local machine. Run the training/validation/testing of the context prediction pipeline by running: `python main.py`
        
 The available options for using different sensor data to predict the contexts are: 
 *Predict using only Coordinates: `python main.py --input coord`
  *Predict using only LiDAR: `python main.py --input lidar`
   *Predict using only Image: `python main.py --input img`
    *Predict using only Image and LiDAR: `python main.py --input img lidar`
     *Predict using all three sensing data (Coordinate, Image, and LiDAR): `python main.py --input coord lidar img`
        
The trained models using different input options are stored in `model_folder`. 

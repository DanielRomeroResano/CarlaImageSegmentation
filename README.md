Carla segmentation in nighttime conditions using Unet, Deeplab, PspNet & Segnet
--------------------------------------------------------------------------------

There are multiple files used in this project that must be run in a certain order for them to work:

   - Dataset generation (GenerateDataset.ipynb):
       - Install CARLA simulator 0.9.13: https://github.com/carla-simulator/carla/releases/tag/0.9.13/
       - Python 3.9 version must be installed
       - Jupyter notebook/lab installed to open the notebook files
       - Install carla Python API: https://pypi.org/project/carla/
       - Make sure you have the rest of the common python libraries used in the file
       - Open the jupyter notebook and run all the cells
       - Two folders (out & outSeg) must be created in the same directory as the jupyter file, they will be the output folder for the images and GT
     
  - Dataset processing (DatasetProcessing.ipynb):
    - Numpy and OpenCV must be installed for this file to work
    - Set you input and output folders to be used at the beginning of the file & run the notbook, conv folder will contain the converted images
    - ShowMapping method is not being called, but it can be called for debugging proposes to see each of the classes in the Cityscapes format (see what each class actually is)
      
  - Model definition:
    - All the models are defined into the Models folder as a python file, these files are not meant to be run by themselves but called from a jupyter notebook for model training or testing   
      
  - Model Training (TrainAll.ipynb):
    - Cuda 12 & Cudnn compatible GPU is recomended (Code is meant to run in at least 12 GB of VRAM)
    - Tensorflow must be installed (Again GPU is highly recommended, training time on GPU would be huge)
    - Codecarbon python library must be installed to measure cpu consumption
    - Pandas, numpy, sklearn, and matplotlib are needed
      
  - Model Evaluation (TestModels.ipynb):
    - The models trained are stored into .h5 files now this file loads them and test them against a real or simulated dataset.


Note: .h5 files & dataset cannot be uploaded into gitlab due to their file size, therefore you´ll need to run the whole code for testing things out.

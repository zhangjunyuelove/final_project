# final_project
## Introduction
This is the codes for my final project to learn the attributes of both image and video<br>
## Preparation
1.Download the environment of Tensorflow <br>
2.Download cifar-10,ucf-50,ucf-101 datasets <br>
3.For video datasets, extract the frame level features first <br>
4.run the training codes from each file using different models from the frame_level_model.py<br>
5.Before run those training files, ensure all the flags especially file path set correctly <br>
## Testing
1. To test the models, run eval.py of image part and inteference.py for video classification<br>
2. The inteference.py can output a txt file that contains the predictions<br>
3. Enter txt file name in the file_averaging.py to ensemble the results<br>

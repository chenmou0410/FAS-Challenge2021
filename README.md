# FAS-Challenge2021
Using CVT and three different modals for anti-spoofing face detection 


The enviorment file is FAS.yaml
You can use it to initial enviorment by conda.

Preprocess:
The code will use three different modality: RBG, Dynamic texture and DCT image.
Run the preprocess.py to generate the Dynamic texture from RGB image.
For each image, it will generate the .npy file.
For train: please run the process_train()
For test: please run the process_val()


Train:
Run train_3m.py

Test:
Run output.py

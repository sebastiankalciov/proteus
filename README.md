# Proteus

---

A Mobile Application with hand gesture recognition to predict a hand gesture for Rock-Paper-Scissors.

## Technologies
- React Native
- Fast API
- CNN built with PyTorch

## Dataset structure
```
data /
|
|-- train/
|   |- rock/
|   |- paper/
|   |- scissors/
|
|-- test/
|   |- rock/
|   |- paper/
|   |- scissors/
```
The dataset consists of 700 images for each type:
- train: 540
- test: 160


## Model architecture
- 3 convolutional blocks
- fully connected layers
     
     - Dense(512) -> BatchNorm -> Dropout
     - Output layer with 3 units (one per class)


## How to run

### Mobile

Prequisites:
- NodeJS
- Expo Go installed on your mobile phone
- You need to download the dataset from my personal [gdrive folder](https://drive.google.com/drive/folders/11Axg84oDUUOQmnwZXKvqWP4wVR3IIgqA?usp=sharing) and place it inside `data/`, as it is not uploaded here on this repo.
```bash
 $ cd mobile
 
 # install dependencies
 $ npm install

 # start app, then scan qr code using expo go
 $ npx expo start
```

### Backend (FastAPI)

```bash
$ cd model

# create a virtual environment
$ python -m venv env
$ source venv/bin/activate  # Linux/macOS
$ env\Scripts\activate      # Windows

# install dependencies
$ pip install fastapi uvicorn torch numpy python-multipart

# start server
$ python api.py api
```

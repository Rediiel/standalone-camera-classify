# Standalone Camera Classify Application

This application accesses the camera that is built-in in your laptop (or any other configured camera) and identifies in real-time the object that is in front of the camera, using the trained model that is contained in the `model` subfolder.

## Identifiable Classes

The system can recognize the following classes:

- backpack  
- bag  
- gabriel (human with white skin tone)  
- glasses  
- headset  
- keyboard  
- laptop  
- monitor  
- mouse  
- pen  

## Installation
```bash
git clone https://github.com/Rediiel/standalone-camera-classify.git
cd standalone-camera-classify
poetry install
```


## How to Test This Application

```bash
poetry run test
```

## How to build This Application

```bash
poetry build
```

## How to Run This Application

```bash
poetry run camera_classify
```

## Static Tests

If you want to run static tests before commiting your code you can run:

```bash
poetry run black .
poetry run flake8 .
```

You can automatise this tasks before commit by installing the pre-commit tool:
```bash
poetry run pre-commit install
```
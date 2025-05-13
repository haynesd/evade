# EVADE – Elliptical Envelope-Based Anomaly Detection Engine

This repository contains the implementation of EVADE, A Lightweight, Unsupervised Framework for Malware Detection in Encrypted Network Traffic.  

** The data set and code for research purposes only**

## Reference
[Paper TBP]

The CIC-IoT-2023 data is downloadable as Merged01.csv files via [Kaggle](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset).

## File Structure
```
evade
├── LICENSE
├── README.md
├── requirements.txt
├── data
    ├── Merged01.csv
    ├── Merged02.csv
    ├── Merged03.csv
    ├── Merged04.csv
    ├── Merged05.csv
├── source 
    ├── __init__.py
    ├── ae.py
    ├── data_loader.py
    ├── main.py
    ├── model.py
    ├── utils.py
├── trained_models
    ├── trained_models_bundle.zip
```

## Setup Environment
To set up your Python on your workstation:

### 1. Install a python venv on the Linux distro
#### note: if you use the latest [git](https://git-scm.com/downloads) bash, there is no need to install venv, go to step 3.
```python3 -m venv venv```
### 2. Activate the virtual python directory and cd into the newly created directory
#### ```source venv/bin/activate``
### 3. Use git clone to retrieve the source
``` git clone git@github.com:haynesd/nad.git ```
### 4. The file structure should now look like this:
```
├── LICENSE
├── README.md
├── requirements.txt
├── source 
```
### 4. Install python import requirements
```pip install -r requirements.txt```
### 5. Make directory data
``` mkdir data```
### 6. Download [CIC-IoT-2023 data](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset) set
### 7. Extract archive.zip download
#### Download Merged01.csv, Merged02.csv Merged03.csv, Merged04.csv, and Merged05.csv from website and place in data directory
### 8. Run Training
#### changed directory to source: ```cd to evade```
#### run training by typing: ```python source/main.py --mode train --data_dir ../data --model_dir ../trained_model```
### 9. Run Test
#### You can move code to test on other devices such as a Raspberry Pi:  ```scp -r trained_models_bundle.zip david@192.168.1.249:/home/david/source/repos/evade/trained_models/```
#### run test by typing: ```python source/main.py --mode test --data_dir ../data --model_dir ../trained_models```



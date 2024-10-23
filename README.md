# Network Anamoly Detection (NAD)

This repository contains the implementation of a network anamoly dectection algorithm using unsupervised learning techniques.  

** The data set and code for research purpose only**

## Reference
[Paper TBP]

[AIS-NIDS: An intelligent and self-sustaining network intrusion detection system - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167404824002876)

[A sequential deep learning framework for a robust and resilient network intrusion detection system - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167404824002311?via%3Dihub)

The ACI-IoT-2023.csv data set is 5.4 gb and is accessible via [Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023) or [IEEE Data Port](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023).

## File Structure
```
nad
├── LICENSE
├── README.md
├── requirements.txt
├── data
    ├── ACI-IoT-2023-Payload.csv
├── source 
    ├── __init__.py
    ├── data_loader.py
    ├── data_preparation.py
    ├── data_processing.py
    ├── supervised_model.py
    ├── unsupervised_model.py
    ├── model_evaluation.py
    ├── test.py
    ├── visualization.py
```

## Setup Environment
To setup your python on your workstation:

### 1. Install a python venv on linux distro
#### note: if you are using the latest git bash, there is no need to install venv, goto step 3.
```python -m venv venv```
### 2. Activate the virtual python directory and cd into the newly created directory
#### ```source venv/bin/activate```
#### ``` cd /venv/``` 
### 3. Use git clone to retrieve the source
``` git clone git@github.com:haynesd/nad.git ```
### 4. File structure should now look like this:
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
### 5. Download [ACI-IoT-2023 data](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023) set 
### 6. Extract archive.zip download
#### Extract ACI-IoT-2023-Payload.csv from archive.zip and place in data directory
### 7. Run Test
#### changed directory to source: ```cd to nad/source```
#### run test by typing: ```python test.py```



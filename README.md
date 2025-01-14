# Network Anamoly Detection (NAD)

This repository contains the implementation of a network anamoly dectection algorithm using unsupervised learning techniques.  

** The data set and code for research purpose only**

## Reference
[Paper TBP]

[AIS-NIDS: An intelligent and self-sustaining network intrusion detection system - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167404824002876)

[A sequential deep learning framework for a robust and resilient network intrusion detection system - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167404824002311?via%3Dihub)

[Network Intrusion Detection for IoT-Botnet Attacks Using ML Algorithms](https://ieeexplore.ieee.org/document/10334188)

The Archive.zip data set is 5.4 gb and is accessible via [Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023) or [IEEE Data Port](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023).

The CIC-IoT-2024 data is downloadable by Merged01.csv files via [Kaggle](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset).

## File Structure
```
nad
├── LICENSE
├── README.md
├── requirements.txt
├── data
    ├── ACI-IoT-2023-Payload.csv
    ├── Merged01.csv
    ├── Merged02.csv
    ├── Merged03.csv
    ├── Merged04.csv
    ├── Merged05.csv
├── source 
    ├── __init__.py
    ├── ACI_IoT_Dataset_2023.py
    ├── CIC_IoT_Dataset_2023.py
    ├── main.py
    ├── supervised_model.py
    ├── unsupervised_model.py
    ├── model_evaluation.py

```

## Setup Environment
To setup your python on your workstation:

### 1. Install a python venv on linux distro
#### note: if you are using the latest [git](https://git-scm.com/downloads) bash, there is no need to install venv, goto step 3.
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
### 6. Download [CIC-IoT-2023 data](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset) set
### 7. Extract archive.zip download
#### Extract ACI-IoT-2023-Payload.csv from archive.zip and place in data directory
#### Download Merged01.csv, Merged02.csv Merged03.csv, Merged04.csv, and Merged05.csv from website and place in data directory
### 8. Run Test
#### changed directory to source: ```cd to nad/source```
#### run test by typing: ```python main.py```



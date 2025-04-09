# Network Anamoly Detection (NAD)

This repository contains the implementation of a network anomaly detection algorithm using unsupervised learning techniques.  

** The data set and code for research purposes only**

## Reference
[Paper TBP]

[AIS-NIDS: An intelligent and self-sustaining network intrusion detection system - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167404824002876)

[A sequential deep learning framework for a robust and resilient network intrusion detection system - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167404824002311?via%3Dihub)

[Network Intrusion Detection for IoT-Botnet Attacks Using ML Algorithms](https://ieeexplore.ieee.org/document/10334188)

The Archive.zip data set is 5.4 GB and is accessible via [Kaggle](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023) or [IEEE Data Port](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023).

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
    ├── model.py
    ├── util.py
├── trained_models
    ├── trained_models_bundle.zip
```

## Setup Environment
To setup your python on your workstation:

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
### 5. Download [ACI-IoT-2023 data](https://www.kaggle.com/datasets/emilynack/aci-iot-network-traffic-dataset-2023) set 
### 6. Download [CIC-IoT-2023 data](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset) set
### 7. Extract archive.zip download
#### Extract ACI-IoT-2023-Payload.csv from archive.zip and place it in the data directory
#### Download Merged01.csv, Merged02.csv Merged03.csv, Merged04.csv, and Merged05.csv from website and place in data directory
### 8. Run Test
#### changed directory to source: ```cd to nad```
#### run training by typing: ```python source/main.py --mode train --data_dir ./data --model_dir ./trained_model```
### 9. Run Training
#### You can move code to test on other devices such as a Raspberry Pi:  ```scp -r trained_models_bundle.zip david@192.168.1.249:/home/david/source/repos/nad/trained_models/```
#### run test by typing: ```python source/main.py --mode test --data_dir ./data --model_dir ./trained_model```



#! /bin/bash
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt install python-is-python3 -y
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt install python3-virtualenv -y
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install python3-pip -y
sudo apt install python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools
sudo apt-get install git -y
git clone https://github.com/Omkar0017/ML_Deployment.git
sudo apt install python3-venv -y
cd /ML_Deployment/ML\ CA1/
python3 -m venv ML\ CA1/
source ML\ CA1/bin/activate
pip install wheel
pip install uwsgi flask
sudo ufw allow 5000
pip3 install numpy
pip3 install pandas
pip3 install -U scikit-learn
python3 app.py

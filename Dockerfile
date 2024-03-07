FROM continuumio/miniconda3

RUN conda install jupyter -y 

RUN conda install jupyterlab -y

RUN jupyter-lab --generate-config

COPY ./env.yml ./env.yml

RUN conda env create -f env.yml

RUN conda install ipykernel

RUN conda install -c conda-forge nb_conda_kernels

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./.jupyter_config_files/jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py

COPY ./.jupyter_config_files/themes.jupyterlab-settings /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

CMD ["jupyter-lab", "./workshops/", "--allow-root"]

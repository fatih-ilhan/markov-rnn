FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install numpy==1.16.1
RUN pip install scikit-learn==0.20.2
RUN pip install pandas==0.23.4
RUN pip install seaborn==0.9.0
RUN pip install statsmodels==0.10.1
RUN pip install scipy==1.3.3

WORKDIR /workspace

FROM manifoldai/orbyter-dl-dev
ADD requirements.txt /build/requirements.txt
WORKDIR /build/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_md
RUN python -m spacy download en_core_web_trf
RUN jupyter labextension install jupyterlab-plotly@4.14.3
WORKDIR /mnt/

FROM python:3.9

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/clowder

RUN python -m pip install --upgrade pip
#RUN python -m pip install pyclowder

COPY requirements.txt ./
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY optimizer.py image_analysis_extractor.py Jim_ColorHistogram_ColorScatterPlot.py AnalysisToolbox_Jim.py ColorHistogram_ColorScatterPlot.py extractor_info.json ./

CMD python3 image_analysis_extractor.py

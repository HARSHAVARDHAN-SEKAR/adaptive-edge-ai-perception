# Universal image — runs on ANY Linux x86_64/arm64 machine, CPU-only by
# default. Two build targets:
#   lite: mock backend only (numpy/opencv), ~300 MB — demos, CI, dashboard
#   full: + ultralytics/torch — real YOLO models on CPU (or --gpus all)
#
#   docker build -f docker/Dockerfile --target lite -t edge-perception:lite .
#   docker build -f docker/Dockerfile --target full -t edge-perception:full .

FROM python:3.11-slim AS lite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir numpy opencv-python-headless psutil pyyaml
COPY . .
RUN python tests/test_pipeline.py            # image is broken if tests fail
EXPOSE 8090
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["dashboard"]

FROM lite AS full
RUN pip install --no-cache-dir -e ".[real]"
RUN python tests/test_pipeline.py

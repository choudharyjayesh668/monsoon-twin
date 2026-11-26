# Hosting image for the demo: the twin server plus the demo site, on open-source maths only.
# PyTorch comes from conda-forge, built against OpenBLAS (BSD-3), not Intel MKL or Apple Accelerate.
FROM mambaorg/micromamba:1.5.10

RUN micromamba install -y -n base -c conda-forge python=3.12 "pytorch=*=cpu_generic*" "libblas=*=*openblas" \
        numpy fastapi uvicorn && micromamba clean --all --yes

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# The twin listens inside the container; the demo site is the public address and passes requests on to the twin.
CMD ["/bin/bash", "-c", "python serve.py --port 8000 & exec python demo/serve_demo.py --port ${PORT:-8080} --twin http://127.0.0.1:8000"]

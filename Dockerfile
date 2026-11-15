# Optional container for the scoring API.
# NOTE: I wrote this file but could not build or run it, because Docker was
# not running on my machine. Treat it as untested. The supported way to run
# the project is the local setup in README.md.

FROM python:3.12-slim

# LightGBM needs the OpenMP runtime library on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at run time: code and trained artifacts.
COPY src ./src
COPY serving ./serving
COPY artifacts ./artifacts

EXPOSE 8711
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8711", "--no-access-log"]

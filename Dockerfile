FROM python:3.13-slim-bookworm

LABEL maintainer="druidblack"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl iputils-ping && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "fix_abs_metadata.py"]
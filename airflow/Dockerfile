FROM apache/airflow:2.10.4-python3.10

USER root

RUN mkdir -p /var/lib/apt/lists/partial && \
    chmod -R 755 /var/lib/apt/lists && \
    apt-get update && \
    apt-get install -y gosu && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow


RUN pip install --no-cache-dir \
    "langchain>=0.3.4" \
    "langchain-core>=0.3.4" \
    "langchain-community>=0.3.4" \
    "pydantic>=2.7,<3" \
    "pendulum==2.1.2" \
    requests \
    pymongo \
    python-dotenv \
    beautifulsoup4 \
    lxml \
    pymilvus



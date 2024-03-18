FROM python:3.8

# Install MLflow and dependencies
RUN pip install mlflow psycopg2-binary

# Set the MLflow server port
ENV MLFLOW_SERVER_PORT 5000

# Expose the MLflow server port
EXPOSE $MLFLOW_SERVER_PORT

# Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client

# Set up PostgreSQL environment variables
ENV POSTGRES_USER mlflow
ENV POSTGRES_PASSWORD mlflow
ENV POSTGRES_DB mlflow
ENV POSTGRES_HOST postgres
ENV POSTGRES_PORT 5432

# Set the entry point for running the MLflow server
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"]

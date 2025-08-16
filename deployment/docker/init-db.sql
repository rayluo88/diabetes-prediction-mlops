-- Database initialization script for MLflow and Prefect
-- This script creates the necessary databases and users

-- Create MLflow database (already exists as default)
-- POSTGRES_DB creates this automatically

-- Create Prefect database and user
CREATE DATABASE prefect;
CREATE USER prefect WITH ENCRYPTED PASSWORD 'prefect_password';
GRANT ALL PRIVILEGES ON DATABASE prefect TO prefect;

-- Grant connection permissions
GRANT CONNECT ON DATABASE mlflow TO mlflow;
GRANT CONNECT ON DATABASE prefect TO prefect;

-- Switch to prefect database to set up permissions
\c prefect;
GRANT ALL ON SCHEMA public TO prefect;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO prefect;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO prefect;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO prefect;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO prefect;

-- Switch back to mlflow database
\c mlflow;
GRANT ALL ON SCHEMA public TO mlflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow;
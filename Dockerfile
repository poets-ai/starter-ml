# Tensforlow 2 official image with Python 3.6 as base
FROM python:3.8

# Maintainer info
LABEL maintainer="svelez.velezgarcia@gamil.com"

# Make working directories
RUN  mkdir -p /home/project-api
WORKDIR /home/project-api/

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
COPY  . .

# Run the python application
CMD ["python", "app/main.py"]
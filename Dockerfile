FROM pytorch/pytorch

# Set the working directory
WORKDIR .

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the files
COPY . .
# CMD ["python", "main.py"]
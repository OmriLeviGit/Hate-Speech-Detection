import subprocess
import datetime
import os

# need to install pg_dump first by using "sudo apt install postgresql-client"
# used to save the entire database before closing the machine

# Database connection parameters
db_params = {
    'dbname': 'tagger_db',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5432'
}

# Generate a timestamped backup file name
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
backup_filename = f"tagger_db_backup_{timestamp}.sql"

# Set the PGPASSWORD environment variable so pg_dump can authenticate without prompting
os.environ['PGPASSWORD'] = db_params['password']

# Construct the pg_dump command
command = [
    'pg_dump',
    '-h', db_params['host'],
    '-p', db_params['port'],
    '-U', db_params['user'],
    '-F', 'p',  # plain SQL
    '-f', backup_filename,
    db_params['dbname']
]

# Run the command
try:
    subprocess.run(command, check=True)
    print(f"Backup completed successfully: {backup_filename}")
except subprocess.CalledProcessError as e:
    print("Backup failed.")
    print(e)

import subprocess
import os
import sys

# used to load the entire database from a backup_file.sql


# Database connection parameters
db_params = {
    'dbname': 'tagger_db',
    'user': 'postgres', 
    'password': '1234',
    'host': 'localhost',
    'port': '5432'
}

# Backup file to restore from
FILENAME = "tagger_db_backup_2025-06-16_06-37-55.sql"  # Change this to your file

# Set password environment variable
os.environ['PGPASSWORD'] = db_params['password']

# Construct the psql command
command = [
    'psql',
    '-h', db_params['host'],
    '-p', db_params['port'], 
    '-U', db_params['user'],
    '-d', db_params['dbname'],
    '-f', FILENAME
]

# Run the restore
try:
    subprocess.run(command, check=True)
    print(f"Database restored successfully from: {FILENAME}")
except subprocess.CalledProcessError as e:
    print("Restore failed.")
    print(e)
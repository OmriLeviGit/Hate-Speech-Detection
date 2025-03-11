from db_access import get_database_instance
from datetime import timedelta

db = get_database_instance()
emails = ['']
num_days = 21
for email in emails:
    passcode = db.create_passcode(email,num_days)
    print(f"Passcode created successfully for user '{email}': {passcode}")

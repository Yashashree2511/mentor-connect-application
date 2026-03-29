import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="appuser",
    password="1234",
    database="mentor_db"
)

print("Connected successfully!")
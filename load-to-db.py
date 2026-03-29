import pandas as pd
import mysql.connector

df = pd.read_csv("cleaned_data_v2.csv")

conn = mysql.connector.connect(
    host="localhost",
    user="appuser",
    password="1234",
    database="mentor_db"
)

cursor = conn.cursor()

query = """
INSERT INTO mentor (mentor_id, name, skills, experience_years, rating)
VALUES (%s, %s, %s, %s, %s)
"""

for _, row in df.iterrows():
    cursor.execute(query, (
        int(row['mentor_id']),
        row['name'],
        row['skills'],
        int(row['experience_years']),
        float(row['rating']),
    ))

conn.commit()
cursor.close()
conn.close()

print("Data inserted into MySQL")
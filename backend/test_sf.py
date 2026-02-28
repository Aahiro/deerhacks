import os
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'),
    role=os.getenv('SNOWFLAKE_ROLE')
)

cursor = conn.cursor()
try:
    cursor.execute("SELECT current_version()")
    one_row = cursor.fetchone()
    print("Snowflake connection successful. Version:", one_row[0])
    
    # Check tables
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    print("Tables:", [t[1] for t in tables])
finally:
    cursor.close()
    conn.close()

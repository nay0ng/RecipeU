# test_connection.py
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
uri = "neo4j+ssc://d36e26d2.databases.neo4j.io"
password = os.getenv("NEO4J_PASSWORD")

# 시도 1: neo4j
try:
    driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    with driver.session() as session:
        session.run("RETURN 1").single()
    print("✅ username: neo4j 성공!")
    driver.close()
except Exception as e:
    print(f"❌ neo4j 실패: {e}")

# 시도 2: d36e26d2
try:
    driver = GraphDatabase.driver(uri, auth=("d36e26d2", password))
    with driver.session() as session:
        session.run("RETURN 1").single()
    print("✅ username: d36e26d2 성공!")
    driver.close()
except Exception as e:
    print(f"❌ d36e26d2 실패: {e}")
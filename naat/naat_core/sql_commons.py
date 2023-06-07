import sqlite3

DB_NAME = "climate_litigation.db"


class DbConnection(object):
    def __init__(self, db_name=DB_NAME):
        self.connection = sqlite3.connect(db_name)

    def __enter__(self):
        return self.connection, self.connection.cursor()

    def __exit__(self, type, value, traceback):
        self.connection.close()

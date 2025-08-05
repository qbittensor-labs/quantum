import os
import sqlite3


class DatabaseManager:
    """
    This class handles connecting to the SQLite database
    """

    def __init__(self, db_name="database.db"):
        """
        Initialize the DatabaseManager.
        """
        if os.path.isabs(db_name) or os.path.sep in db_name:
            self.db_path = db_name
        else:
            self.db_path = os.path.join(os.getcwd(), db_name)

        self.conn = None
        self.cursor = None

    def connect(self):
        """
        Establishes a connection to the SQLite database.

        If a connection already exists, it will be closed and a new one will be opened.
        Handles sqlite3.Error exceptions during connection.
        """
        if self.conn:
            print("Closing existing connection before opening a new one.")
            self.close()

        try:
            # Connect to the SQLite database. If the file doesn't exist, it will be created.
            self.conn = sqlite3.connect(self.db_path)
            # Row factory to return rows as sqlite3.Row objects (access by column name)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()  # Get a cursor object to execute SQL commands
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None
            self.cursor = None

    def close(self):
        """
        Closes the database connection.
        Ensures the connection is committed before closing to save any pending changes.
        """
        if self.conn:
            try:
                self.conn.commit()  # Commit any pending transactions
                self.conn.close()  # Close the connection
            except sqlite3.Error as e:
                print(f"Error closing database connection: {e}")
            finally:
                self.conn = None
                self.cursor = None
        else:
            print("No active database connection to close.")

    def execute_query(self, query, params=()):
        """
        Executes a given SQL query with optional parameters.

        This method is suitable for DDL (CREATE, ALTER, DROP) and DML (INSERT, UPDATE, DELETE)
        statements that do not return results. Changes are committed automatically.

        Args:
            query (str): The SQL query string
            params (tuple): A tuple of parameters to substitute into the query

        Returns:
            bool: True if the query executed successfully, False otherwise.
        """
        if not self.conn:
            print("Error: Not connected to the database. Call .connect() first.")
            return False

        try:
            self.cursor.execute(query, params)
            self.conn.commit()  # Commit changes immediately after execution
            return True
        except sqlite3.Error as e:
            print(f"Error executing query '{query}': {e}")
            self.conn.rollback()  # Rollback in case of an error to maintain data integrity
            return False

    def fetch_all(self, query, params=()):
        """
        returns all resulting rows.

        Args:
            query (str): The SQL query string to execute
            params (tuple): A tuple of parameters to substitute into the query

        Returns:
            list: Returns an empty list if no results or on error.
            Rows can be accessed by column name (e.g., row['column_name'])
        """
        if not self.conn:
            print("Error: Not connected to the database. Call .connect() first.")
            return []

        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error fetching all results for query '{query}': {e}")
            return []

    def fetch_one(self, query, params=()):
        """
        Executes a SQL query and returns a single resulting row.

        This method is suitable for SELECT statements where only one row is expected

        Args:
            query (str): The SQL query string
            params (tuple): A tuple of parameters to substitute in the query

        Returns:
            sqlite3.Row or None: A single sqlite3.Row object representing the row,
            or None if no results or on error
            The row can be accessed by column name (e.g., row['column_name'])
        """
        if not self.conn:
            print("Error: Not connected to the database. Call .connect() first.")
            return None

        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error fetching one result for query '{query}': {e}")
            return None

    def get_table_names(self):
        """
        Retrieves a list of all table names in the current database.

        Returns:
            list: A list of strings, where each string is a table name.
            Returns an empty list on error or if no tables exist.
        """
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = self.fetch_all(query)
        # sqlite3.Row objects can be accessed like dictionaries or by index
        return [row["name"] for row in tables] if tables else []

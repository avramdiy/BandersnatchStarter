# reset_db.py

from app.data import Database

def main():
    db = Database()  # Initialize the Database instance
    db.reset()       # Call the reset method to delete all documents

if __name__ == "__main__":
    main()

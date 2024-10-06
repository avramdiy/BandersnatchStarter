import os
from pymongo import MongoClient
from pandas import DataFrame
import certifi
from dotenv import load_dotenv
from MonsterLab import Monster
from random import randint, uniform, choice

# Load environment variables from the .env file
load_dotenv()

class Database:
    """
    A class to manage interactions with a MongoDB database, including seeding data,
    resetting the collection, counting documents, and retrieving data in various formats.
    """

    def __init__(self):
        """
        Initializes the Database connection using the MongoDB URI from environment variables.
        Establishes a secure TLS connection with the CA certificate.
        """
        db_url = os.getenv('DB_URL')  # Fetch the MongoDB URI from environment variables
        if not db_url:
            raise ValueError("DB_URL is not set in the environment.")

        # Initialize MongoClient with TLS CA certificate for a secure connection
        self.client = MongoClient(db_url, tlsCAFile=certifi.where())
        self.db = self.client['Cluster0']  # Database Name
        self.collection = self.db['your_collection_name']  # Replace with your actual collection name

    def seed(self, amount: int):
        """
        Inserts a specified number of Monster documents into the MongoDB collection.

        :param amount: The number of documents to insert.
        """
        documents = []
        for _ in range(amount):
            monster = Monster()  # Instantiate a Monster object
            doc = {
                "Name": monster.name,
                "Type": monster.type,
                "Level": randint(1, 20),
                "Health": round(uniform(50, 200), 2),
                "Energy": round(uniform(10, 100), 2),
                "Sanity": round(uniform(0, 100), 2),
                "Rarity": self.random_rarity()
            }
            documents.append(doc)

        if documents:
            self.collection.insert_many(documents)
            print(f"Inserted {amount} documents into the collection.")

    def reset(self):
        """
        Deletes all documents from the MongoDB collection.
        """
        result = self.collection.delete_many({})
        print(f"Deleted {result.deleted_count} documents from the collection.")

    def count(self) -> int:
        """
        Returns the number of documents in the MongoDB collection.

        :return: The count of documents.
        """
        return self.collection.count_documents({})

    def dataframe(self) -> DataFrame:
        """
        Retrieves all documents from the collection and returns them as a pandas DataFrame.

        :return: A DataFrame containing all documents, or an empty DataFrame if none exist.
        """
        data = list(self.collection.find())
        if data:
            return DataFrame(data)
        else:
            return DataFrame()  # Return empty DataFrame if collection is empty

    def html_table(self) -> str:
        """
        Converts the collection's documents into an HTML table.

        :return: An HTML table string if documents exist, otherwise None.
        """
        df = self.dataframe()
        if not df.empty:
            return df.to_html()
        else:
            return None

    @staticmethod
    def random_rarity() -> str:
        """
        Generates a random rarity level for a Monster.

        :return: A string representing the rarity.
        """
        rarities = ["Common", "Uncommon", "Rare", "Epic", "Legendary"]
        return choice(rarities)

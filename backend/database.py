import os

from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError

_DEFAULT_MONGODB_URI = "mongodb://localhost:27017/"
_DEFAULT_DB_NAME = "healthai"

MONGODB_URI = os.getenv("MONGODB_URI", _DEFAULT_MONGODB_URI)
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", _DEFAULT_DB_NAME)

client = AsyncMongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
database = client.get_database(MONGODB_DB_NAME)
query_history_collection = database.get_collection("query_history")

_database_status = {
    "available": False,
    "message": "Database connection has not been verified yet.",
}


def _set_database_status(available: bool, message: str) -> None:
    _database_status["available"] = available
    _database_status["message"] = message


async def ping_database() -> None:
    try:
        await client.admin.command("ping")
        _set_database_status(True, "MongoDB connection verified.")
    except PyMongoError as exc:
        message = (
            "Unable to connect to MongoDB. Set MONGODB_URI or start a MongoDB "
            "server that is reachable from this machine."
        )
        _set_database_status(False, message)
        raise RuntimeError(message) from exc


async def ensure_indexes() -> None:
    try:
        await query_history_collection.create_index("timestamp")
        _set_database_status(True, "MongoDB connection verified.")
    except PyMongoError as exc:
        message = "Unable to ensure MongoDB indexes."
        _set_database_status(False, message)
        raise RuntimeError(message) from exc


async def close_database() -> None:
    await client.close()
    _set_database_status(False, "MongoDB client closed.")


def get_query_history_collection():
    return query_history_collection


def get_database_status() -> dict[str, str | bool]:
    return dict(_database_status)

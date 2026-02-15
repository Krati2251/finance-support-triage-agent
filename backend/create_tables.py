"""
Standalone script — run once to create all tables in the database.

Usage:
    python create_tables.py
"""

from database import engine, Base

# Import all models so Base.metadata is fully populated
from models import Ticket  # noqa: F401


def create_tables():
    print("🔗 Connecting to the database...")
    print(f"   Engine URL: {engine.url}")

    print("📦 Creating tables (if they don't already exist)...")
    Base.metadata.create_all(bind=engine)

    print("✅ All tables created successfully!")
    print()

    # List the tables that were registered
    for table_name in Base.metadata.tables:
        print(f"   • {table_name}")


if __name__ == "__main__":
    create_tables()

from pathlib import Path
import sqlite3

# ============================================================
# 1. Database Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_DIR = PROJECT_ROOT / "data" / "database"

DATABASE_PATH = DATABASE_DIR / "ecommerce.db"


# ============================================================
# 2. Create Tables
# ============================================================

def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create the tables required by the order system.
    """

    cursor = conn.cursor()

    # --------------------------------------------------------
    # orders
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL,
            created_at TEXT NOT NULL,
            paid_at TEXT
        )
        """
    )

    # --------------------------------------------------------
    # shipments
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS shipments (
            shipment_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            carrier TEXT NOT NULL,
            tracking_number TEXT NOT NULL,
            status TEXT NOT NULL,
            latest_location TEXT,
            estimated_delivery TEXT,

            FOREIGN KEY (order_id)
                REFERENCES orders(order_id)
        )
        """
    )

    conn.commit()


# ============================================================
# 3. Insert Demo Data
# ============================================================

def insert_demo_data(conn: sqlite3.Connection) -> None:
    """
    Insert several demo orders and shipment records.

    INSERT OR IGNORE makes this script safe to run repeatedly.
    """

    cursor = conn.cursor()

    orders = [
        (
            "ORD202610001",
            "U001",
            "PAID",
            299.00,
            "2026-10-01 10:30:00",
            "2026-10-01 10:35:00",
        ),
        (
            "ORD202610002",
            "U001",
            "SHIPPED",
            89.00,
            "2026-10-03 14:20:00",
            "2026-10-03 14:25:00",
        ),
        (
            "ORD202610003",
            "U002",
            "DELIVERED",
            159.00,
            "2026-10-05 09:10:00",
            "2026-10-05 09:15:00",
        ),
    ]

    cursor.executemany(
        """
        INSERT OR IGNORE INTO orders (
            order_id,
            user_id,
            status,
            total_amount,
            created_at,
            paid_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        orders,
    )

    shipments = [
        (
            "SHIP202610001",
            "ORD202610002",
            "顺丰速运",
            "SF202610002",
            "IN_TRANSIT",
            "上海转运中心",
            "2026-10-12",
        ),
        (
            "SHIP202610002",
            "ORD202610003",
            "京东物流",
            "JD202610003",
            "DELIVERED",
            "北京市朝阳区",
            "2026-10-07",
        ),
    ]

    cursor.executemany(
        """
        INSERT OR IGNORE INTO shipments (
            shipment_id,
            order_id,
            carrier,
            tracking_number,
            status,
            latest_location,
            estimated_delivery
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        shipments,
    )

    conn.commit()


# ============================================================
# 4. Initialize Database
# ============================================================

def init_database() -> None:
    """
    Initialize the local SQLite database.
    """

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        # SQLite does not enforce foreign keys by default.
        conn.execute("PRAGMA foreign_keys = ON")

        create_tables(conn)

        insert_demo_data(conn)

        print("Database initialized successfully.")
        print(f"Database path: {DATABASE_PATH}")

    finally:
        conn.close()


# ============================================================
# 5. Entry Point
# ============================================================

if __name__ == "__main__":
    init_database()
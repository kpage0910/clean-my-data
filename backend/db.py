import databases
import sqlalchemy

DATABASE_URL = "postgresql://neondb_owner:npg_qgDwC7R9vQGe@ep-gentle-frog-ae9w4zye-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("google_id", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("email", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("cleanings_this_month", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("last_reset", sqlalchemy.DateTime),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)

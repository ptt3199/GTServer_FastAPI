from sqlalchemy import create_engine, MetaData
from sqlalchemy_schemadisplay import create_schema_graph
from model import *

# Replace with your database credentials
DB_CONFIG = f"postgresql+asyncpg://postgres:1231@localhost:5432/gt_thanh"

# Create the database engine and metadata
engine = create_engine(DB_CONFIG)
metadata = MetaData(bind=engine)

# Create the ERD graph
graph = create_schema_graph(metadata=metadata, show_datatypes=False, show_indexes=False)
graph.write_png('erd.png')
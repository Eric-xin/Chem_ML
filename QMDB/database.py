import pandas as pd
import os

current_dir = os.getcwd()
sql_file = os.path.join(current_dir, 'qmdb__v1_1__102016.sql.gz')

# Read sql.gz file
df = pd.read_csv(sql_file, compression='gzip', header=None, sep='|')
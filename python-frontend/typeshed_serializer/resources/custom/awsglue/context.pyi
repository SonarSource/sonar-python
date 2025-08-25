from pyspark.sql import SQLContext

class GlueContext(SQLContext):
  def __init__(self, *args, **kwargs):
    ...

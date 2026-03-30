import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, lower

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


df = spark.read.csv(
    "s3://healthcare-risk-prediction-pipeline/RAW/Clinical_Notes/Clinical_Notes.csv",
    header=True,
    inferSchema=True
)

print("Initial columns:", df.columns)
df.show(5, truncate=False)

for c in df.columns: 
    new_c = c.strip().lower().replace(" ", "_")
    df = df.withColumnRenamed(c, new_c)

df = (
    df.withColumnRenamed(df.columns[0], "encounter_id")
      .withColumnRenamed(df.columns[1], "note_type")
      .withColumnRenamed(df.columns[2], "note_text")
)

print("Columns AFTER rename:", df.columns)
df.show(5, truncate=False)

from pyspark.sql.functions import col, trim

df = df.withColumn("note_type", trim(col("note_type")))
df = df.withColumn("note_text", trim(col("note_text")))
df = df.withColumn("encounter_id", trim(col("encounter_id")))

df = df.withColumn("encounter_id", col("encounter_id").cast("string"))

df_before = df.count()
df = df.dropDuplicates()
df_after = df.count()

print("Rows before removing duplicates:", df_before)
print("Rows after removing duplicates:", df_after)
print("Duplicates removed:", df_before - df_after)

from pyspark.sql.functions import col, when, trim

df = df.withColumn(
    "note_type",
    when(trim(col("note_type")) == "", None).otherwise(col("note_type"))
)

df = df.withColumn(
    "note_text",
    when(trim(col("note_text")) == "", None).otherwise(col("note_text"))
)

df = df.withColumn(
    "encounter_id",
    when(trim(col("encounter_id")) == "", None).otherwise(col("encounter_id"))
)

from pyspark.sql.functions import col, sum as spark_sum, when

df.select([
    spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in df.columns
]).show()

curated_path = "s3://healthcare-risk-prediction-pipeline/CURATED/hc_clinical_notes/"
df.write.mode("overwrite").parquet(curated_path)
job.commit()


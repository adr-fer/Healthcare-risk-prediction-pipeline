# Initialize the AWS Glue job, create the Spark/Glue context, and start the session
# required to read, transform, and write the dataset.
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

# Load the raw Clinical_Notes.csv file from the S3 RAW layer into a Spark DataFrame.
# inferSchema=True is used so Spark automatically detects the appropriate data types.
df = spark.read.csv(
    "s3://healthcare-risk-prediction-pipeline/RAW/Clinical_Notes/Clinical_Notes.csv",
    header=True,
    inferSchema=True
)

# Display the original column names and preview the first rows of the dataset
# to confirm the file loaded correctly and to inspect the raw structure.
print("Initial columns:", df.columns)
df.show(5, truncate=False)

# Standardize all column names by trimming spaces, converting to lowercase,
# and replacing internal spaces with underscores for consistency and easier querying.
for c in df.columns: 
    new_c = c.strip().lower().replace(" ", "_")
    df = df.withColumnRenamed(c, new_c)

# Rename the columns to clear and meaningful names so the dataset structure
# is easier to understand and reference in downstream transformations.
df = (
    df.withColumnRenamed(df.columns[0], "encounter_id")
      .withColumnRenamed(df.columns[1], "note_type")
      .withColumnRenamed(df.columns[2], "note_text")
)

# Display the updated column names and preview the dataset again to confirm
# that the renaming process was applied correctly.
print("Columns AFTER rename:", df.columns)
df.show(5, truncate=False)

# Remove leading and trailing spaces from the clinical notes fields
# to improve consistency and reduce formatting issues.
from pyspark.sql.functions import col, trim

df = df.withColumn("note_type", trim(col("note_type")))
df = df.withColumn("note_text", trim(col("note_text")))
df = df.withColumn("encounter_id", trim(col("encounter_id")))

# Cast encounter_id to string to preserve identifier formatting
# and ensure consistency across joins and downstream processing.
df = df.withColumn("encounter_id", col("encounter_id").cast("string"))

# Identify and remove duplicate rows from the dataset to improve data quality
# and ensure each clinical note record is represented only once.
df_before = df.count()
df = df.dropDuplicates()
df_after = df.count()

print("Rows before removing duplicates:", df_before)
print("Rows after removing duplicates:", df_after)
print("Duplicates removed:", df_before - df_after)

# Replace blank values with nulls in key columns so missing data can be
# identified consistently and handled more effectively in later analysis.
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

# Standardize the note_type field by trimming spaces and converting values
# to lowercase to reduce formatting inconsistencies across note categories.
from pyspark.sql.functions import lower, trim, col

df = df.withColumn("note_type", lower(trim(col("note_type"))))

# Display distinct note_type values to identify possible spelling issues,
# inconsistent labels, or unexpected note categories.
df.select("note_type").distinct().show(truncate=False)

# Count the number of distinct note_type values to evaluate the categorical
# structure of the clinical notes table.
from pyspark.sql import functions as F
df.select(F.countDistinct("note_type")).show()

# Summarize the length of note_text entries to evaluate free-text quality
# and identify unusually short or unusually long clinical notes.
from pyspark.sql.functions import length

df.select(length(col("note_text")).alias("note_length")).describe().show()

# Print the schema to confirm that the cleaned columns have the expected
# data types before writing the curated output.
df.printSchema()

# Save the cleaned Clinical Notes dataset to the S3 CURATED layer in Parquet format
# for efficient downstream querying and analytics.
curated_path = "s3://healthcare-risk-prediction-pipeline/CURATED/hc_clinical_notes/"
df.write.mode("overwrite").parquet(curated_path)

# Commit the Glue job to finalize execution after all transformations and writes complete.
job.commit()

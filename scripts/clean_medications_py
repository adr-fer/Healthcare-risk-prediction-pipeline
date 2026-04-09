# Initialize the AWS Glue job, create the Spark/Glue context, and start the session
# required to read, transform, and write the dataset.
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Load the raw Facilities.csv file from the S3 RAW layer into a Spark DataFrame.
# inferSchema=True is used so Spark automatically detects the appropriate data types.
df = spark.read.csv(
    "s3://healthcare-risk-prediction-pipeline/RAW/Medications/Medications.csv",
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
    
# Clean selected string-based categorical columns by trimming extra spaces
# and converting text to lowercase to reduce formatting inconsistencies.
from pyspark.sql import functions as F

df = df.withColumn(
    "encounter_id",
    F.lower(F.trim(F.col("encounter_id")))
)

df = df.withColumn(
    "medication_class",
    F.lower(F.trim(F.col("medication_class")))
)

# Count null values and blank strings across all columns to identify missing data
# and assess whether additional handling is needed before analysis.
from pyspark.sql.functions import col, sum as spark_sum, when, trim

df.select([
    spark_sum(
        when(col(c).isNull() | (trim(col(c)) == ""), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
]).show()

total_rows = df.count()

df.select([
    (
        spark_sum(
            when(col(c).isNull() | (trim(col(c)) == ""), 1).otherwise(0)
        ) / total_rows
    ).alias(c)
    for c in df.columns
]).show()

# Print the schema to confirm that the cleaned columns have the expected
# data types before writing the curated output.
df.printSchema()

# Identify and remove duplicate rows from the dataset to improve data quality
# and ensure each clinical note record is represented only once.
df_before = df.count()
df = df.dropDuplicates()
df_after = df.count()

print("Rows before removing duplicates:", df_before)
print("Rows after removing duplicates:", df_after)
print("Duplicates removed:", df_before - df_after)

# Display distinct values for key categorical columns to identify possible
# spelling issues, inconsistent labels, or unexpected categories.
df.select("encounter_id").distinct().show(truncate=False)
df.select("medication_class").distinct().show(truncate=False)


# Count the number of distinct values in each categorical column to evaluate
# cardinality and distinguish identifiers from low-cardinality grouping variables.
df.select(F.countDistinct("encounter_id")).show()
df.select(F.countDistinct("medication_class")).show()

# Calculate the ratio of distinct values to total rows for selected columns.
# This helps assess uniqueness and confirms which fields behave like identifiers.
total_rows = df.count()

df.select(
    (F.countDistinct("encounter_id") / F.lit(total_rows)).alias("cardinality_ratio")
).show()

df.select(
    (F.countDistinct("medication_class") / F.lit(total_rows)).alias("cardinality_ratio")
).show()

# Save the cleaned Encounters dataset to the S3 CURATED layer in Parquet format
# for efficient downstream querying and analytics.
curated_path = "s3://healthcare-risk-prediction-pipeline/CURATED/hc_medications/" 
df.write.mode("overwrite").parquet(curated_path)

# Commit the Glue job to finalize execution after all transformations and writes complete.
job.commit()    

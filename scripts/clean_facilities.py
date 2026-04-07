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
    "s3://healthcare-risk-prediction-pipeline/RAW/Facilities/Facilities.csv",
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
    "facility_id",
    F.lower(F.trim(F.col("facility_id")))
)

df = df.withColumn(
    "facility_name",
    F.lower(F.trim(F.col("facility_name")))
)

df = df.withColumn(
    "region",
    F.lower(F.trim(F.col("region")))
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
df.select("facility_id").distinct().show(truncate=False)
df.select("facility_name").distinct().show(truncate=False)
df.select("region").distinct().show(truncate=False)

# Count the number of distinct values in each categorical column to evaluate
# cardinality and distinguish identifiers from low-cardinality grouping variables.
df.select(F.countDistinct("facility_id")).show()
df.select(F.countDistinct("facility_name")).show()
df.select(F.countDistinct("region")).show()


# Calculate the ratio of distinct values to total rows for selected columns.
# This helps assess uniqueness and confirms which fields behave like identifiers.
total_rows = df.count()

df.select(
    (F.countDistinct("facility_id") / F.lit(total_rows)).alias("cardinality_ratio")
).show()

df.select(
    (F.countDistinct("facility_name") / F.lit(total_rows)).alias("cardinality_ratio")
).show()

df.select(
    (F.countDistinct("region") / F.lit(total_rows)).alias("cardinality_ratio")
).show()


#Checking numerical inconsinstencies 

# Display sample values from numerical columns to manually inspect formatting,
# confirm valid numeric representation, and check for obvious inconsistencies.
df.select("bed_count").show(5, truncate=False)
df.select("teaching_hospital").show(5, truncate=False)
df.select("avg_patients_per_nurse").show(5, truncate=False)

# Generate descriptive statistics for numerical columns, including count, mean,
# standard deviation, minimum, and maximum values, to identify unusual ranges.
df.select("bed_count").describe().show()
df.select("teaching_hospital").describe().show()
df.select("avg_patients_per_nurse").describe().show()

#Analyze the frequency distribution of the binary variable.
df.groupBy("teaching_hospital").count().show()
df.filter(~F.col("teaching_hospital").isin(0, 1)).show()


# Detect potential outliers in in numerical fields using the IQR method.
# Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged as extreme.
# Replace detected outliers in numerical fields with null values so extreme cases
# do not distort downstream analysis and modeling results.
numeric_cols = ["bed_count", "avg_patients_per_nurse"]

for col_name in numeric_cols:
    quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
    q1, q3 = quantiles
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f"\nOutliers for {col_name}:")
    df.filter(
        (F.col(col_name) < lower_bound) | 
        (F.col(col_name) > upper_bound)
    ).show()
    
#Identify records where bed_count is zero or negative, which may indicate invalid or inconsistent data.
df.filter(F.col("bed_count") <= 0).show()

df.filter(F.col("avg_patients_per_nurse") <= 0).show()

# Save the cleaned Encounters dataset to the S3 CURATED layer in Parquet format
# for efficient downstream querying and analytics.
curated_path = "s3://healthcare-risk-prediction-pipeline/CURATED/hc_facilities/" 
df.write.mode("overwrite").parquet(curated_path)

# Commit the Glue job to finalize execution after all transformations and writes complete.
job.commit()

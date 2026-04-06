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

# Load the raw Encounters.csv file from the S3 RAW layer into a Spark DataFrame.
# inferSchema=True is used so Spark automatically detects the appropriate data types.
df = spark.read.csv(
    "s3://healthcare-risk-prediction-pipeline/RAW/Encounters/Encounters.csv",
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

# Identify and remove duplicate rows from the dataset to improve data quality
# and ensure each record is represented only once.
df_before = df.count()
df = df.dropDuplicates()
df_after = df.count()

print("Rows before removing duplicates:", df_before)
print("Rows after removing duplicates:", df_after)
print("Duplicates removed:", df_before - df_after)


# Clean selected string-based categorical columns by trimming extra spaces
# and converting text to lowercase to reduce formatting inconsistencies.
from pyspark.sql import functions as F

string_cols = [
    "encounter_id",
    "patient_id",
    "facility_id",
    "admission_type",
    "department",
    "primary_diagnosis_group",
    "discharge_disposition"
]

for c in string_cols:
    df = df.withColumn(c, F.lower(F.trim(F.col(c))))


# Count null values and blank strings across all columns to identify missing data
# and assess whether additional handling is needed before analysis.
from pyspark.sql.functions import col, sum as spark_sum, when, trim

df.select([
    spark_sum(
        when(col(c).isNull() | (trim(col(c)) == ""), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
]).show()


# Print the schema to confirm that each column has the expected data type,
# especially numerical columns needed for statistical analysis.  
df.printSchema()

# Checking categorical columns inconsistencies. 

# Display distinct values for key categorical columns to identify possible
# spelling issues, inconsistent labels, or unexpected categories. 
df.select("encounter_id").distinct().show(truncate=False)
df.select("patient_id").distinct().show(truncate=False)
df.select("facility_id").distinct().show(truncate=False)
df.select("admission_type").distinct().show(truncate=False)
df.select("department").distinct().show(truncate=False)
df.select("primary_diagnosis_group").distinct().show(truncate=False)
df.select("discharge_disposition").distinct().show(truncate=False)

# Count the number of distinct values in each categorical column to evaluate
# cardinality and distinguish identifiers from low-cardinality grouping variables.
df.select(F.countDistinct("encounter_id")).show()
df.select(F.countDistinct("patient_id")).show()
df.select(F.countDistinct("facility_id")).show()
df.select(F.countDistinct("admission_type")).show()
df.select(F.countDistinct("department")).show()
df.select(F.countDistinct("primary_diagnosis_group")).show()
df.select(F.countDistinct("discharge_disposition")).show()

total_rows = df.count()

# Calculate the ratio of distinct values to total rows for selected columns.
# This helps assess uniqueness and confirms which fields behave like identifiers.
df.select(
    (F.countDistinct("encounter_id") / F.lit(total_rows)).alias("encounter_id_ratio"),
    (F.countDistinct("patient_id") / F.lit(total_rows)).alias("patient_id_ratio"),
    (F.countDistinct("facility_id") / F.lit(total_rows)).alias("facility_id_ratio"),
    (F.countDistinct("admission_type") / F.lit(total_rows)).alias("admission_type_ratio"),
    (F.countDistinct("department") / F.lit(total_rows)).alias("department_ratio"),
    (F.countDistinct("primary_diagnosis_group") / F.lit(total_rows)).alias("primary_diagnosis_group_ratio"),
    (F.countDistinct("discharge_disposition") / F.lit(total_rows)).alias("discharge_disposition_ratio")
).show()

#Checking numerical inconsinstencies 

# Display sample values from numerical columns to manually inspect formatting,
# confirm valid numeric representation, and check for obvious inconsistencies.
df.select("encounter_severity").show(5, truncate=False)
df.select("length_of_stay_days").show(5, truncate=False)
df.select("total_cost_usd").show(5, truncate=False)
df.select("adverse_outcome_flag").show(5, truncate=False)
df.select("readmitted_30_days_flag").show(5, truncate=False)
df.select("polypharmacy_flag").show(5, truncate=False)
df.select("opioid_prescribed_flag").show(5, truncate=False)

# Generate descriptive statistics for numerical columns, including count, mean,
# standard deviation, minimum, and maximum values, to identify unusual ranges.
df.select("encounter_severity").describe().show()
df.select("length_of_stay_days").describe().show()
df.select("total_cost_usd").describe().show()
df.select("adverse_outcome_flag").describe().show()
df.select("readmitted_30_days_flag").describe().show()
df.select("polypharmacy_flag").describe().show()
df.select("opioid_prescribed_flag").describe().show()

# Detect potential outliers in in numerical fields using the IQR method.
# Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged as extreme.
# Replace detected outliers in numerical fields with null values so extreme cases
# do not distort downstream analysis and modeling results.
col_name = "total_cost_usd"

quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
q1, q3 = quantiles
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df.filter(
    (F.col(col_name) < lower_bound) | 
    (F.col(col_name) > upper_bound)
).show()

df = df.withColumn(
    col_name,
    F.when(
        (F.col(col_name) < lower_bound) | 
        (F.col(col_name) > upper_bound),
        None
    ).otherwise(F.col(col_name))
)


# 2. length_of_stay_days
col_name = "length_of_stay_days"

quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
q1, q3 = quantiles
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df.filter(
    (F.col(col_name) < lower_bound) | 
    (F.col(col_name) > upper_bound)
).show()

df = df.withColumn(
    col_name,
    F.when(
        (F.col(col_name) < lower_bound) | 
        (F.col(col_name) > upper_bound),
        None
    ).otherwise(F.col(col_name))
)


# 3. encounter_severity
col_name = "encounter_severity"

quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
q1, q3 = quantiles
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df.filter(
    (F.col(col_name) < lower_bound) | 
    (F.col(col_name) > upper_bound)
).show()

df = df.withColumn(
    col_name,
    F.when(
        (F.col(col_name) < lower_bound) | 
        (F.col(col_name) > upper_bound),
        None
    ).otherwise(F.col(col_name))
)
# Save the cleaned Encounters dataset to the S3 CURATED layer in Parquet format
# for efficient downstream querying and analytics.
curated_path = "s3://healthcare-risk-prediction-pipeline/CURATED/hc_encounters/"
df.write.mode("overwrite").parquet(curated_path)

# Commit the Glue job to finalize execution after all transformations and writes complete.
job.commit()

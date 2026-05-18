import dlt
from pyspark.sql.functions import count, avg, min, max, countDistinct, col

# -------------------------
# Diagnostic Mapping (Materialized)
# -------------------------
@dlt.table(
  name="diagnostic_mapping",
  comment="Reference data for diagnosis codes",
  table_properties={"quality": "bronze"}
)
@dlt.expect_or_drop("diag_code_not_null", "diagnosis_code IS NOT NULL")
@dlt.expect_or_drop("diag_desc_not_null", "diagnosis_description IS NOT NULL")
def diagnostic_mapping():
    df = spark.read.table("healthcare_cat.default.diagnosis_mapping_raw")
    return df.select(
        col("diagnosis_code").cast("string"),
        col("diagnosis_description").cast("string")
    )

# -------------------------
# Daily Patients (Streaming)
# -------------------------
@dlt.table(
  name="daily_patents",
  comment="Streaming patient ingestion",
  table_properties={"quality": "bronze"}
)
@dlt.expect_or_drop("pk_not_null", "patient_id IS NOT NULL")
@dlt.expect_or_drop("required_fields",
    """name IS NOT NULL and age IS NOT NULL and gender IS NOT NULL 
       and contact_number IS NOT NULL and admission_date IS NOT NULL""")
def daily_patents():
    df = spark.readStream.table("healthcare_cat.default.patients_daily_file_raw")
    return df.select(
        col("patient_id").cast("string"),
        col("name").cast("string"),
        col("age").cast("integer"),
        col("gender").cast("string"),
        col("address").cast("string"),
        col("contact_number").cast("string"),
        col("admission_date").cast("date"),
        col("diagnosis_code").cast("string")
    )

# -------------------------
# Processed Patient Data (Streaming)
# -------------------------
@dlt.table(
    name="processed_patient_data",
    comment="Enriched streaming data",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("has_diagnosis", "diagnosis_description IS NOT NULL")
def processed_patient_data():
    patients_df = dlt.read_stream("daily_patents")   # streaming
    diagnosis_df = dlt.read("diagnostic_mapping")    # batch

    return (
        patients_df.alias("p")
        .join(
            diagnosis_df.alias("d"),
            col("p.diagnosis_code") == col("d.diagnosis_code"),
            "left"
        )
        .select(
            col("p.patient_id"),
            col("p.name"),
            col("p.age"),
            col("p.gender"),
            col("p.address"),
            col("p.contact_number"),
            col("p.admission_date"),
            col("d.diagnosis_description")
        )
    )

# -------------------------
# Gold Tables (Materialized)
# -------------------------

# By admission date
@dlt.table(
   name="patient_statistics_by_admission_date",
   table_properties={"quality": "gold"}
)
def patient_statistics_by_admission_date():
    df = dlt.read("processed_patient_data")  # NOT streaming
    return df.groupBy("admission_date", "diagnosis_description") \
             .agg(count("*").alias("patient_count"),
                  avg("age").alias("avg_age"))

# By diagnosis
@dlt.table(
  name="patient_statistics_by_diagnosis",
  table_properties={"quality": "gold"}
)
def patient_statistics_by_diagnosis():
    df = dlt.read("processed_patient_data")
    return df.groupBy("diagnosis_description").agg(
        count("patient_id").alias("patient_count"),
        avg("age").alias("avg_age"),
        min("age").alias("min_age"),
        max("age").alias("max_age"),
        countDistinct("gender").alias("distinct_gender")
    )

# By gender
@dlt.table(
  name="patient_statistics_by_gender",
  table_properties={"quality": "gold"}
)
def patient_statistics_by_gender():
    df = dlt.read("processed_patient_data")
    return df.groupBy("gender").agg(
        count("patient_id").alias("patient_count"),
        avg("age").alias("avg_age"),
        min("age").alias("min_age"),
        max("age").alias("max_age"),
        countDistinct("diagnosis_description").alias("distinct_diagnosis")
    )
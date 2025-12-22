from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from pyspark.sql.window import Window

schema = StructType([
    StructField("date", StringType(), True),
    StructField("time", IntegerType(), True),
    StructField("eyeballs", IntegerType(), True),
    StructField("zeroes", IntegerType(), True),
    StructField("completed_trips", IntegerType(), True),
    StructField("requests", IntegerType(), True),
    StructField("unique_driver", IntegerType(), True),
])

spark = SparkSession.builder.appName("UberDataAnalysis").getOrCreate()

uber_df = spark.read.option("header", "true").schema(schema).csv("./uber-data-analytics/uber.csv")

# ✅ Parse: 10-Sep-12  -> dd-MMM-yy
uber_df = uber_df.withColumn("date_parsed", func.to_date(func.col("date"), "dd-MMM-yy"))

print("Schema")
uber_df.printSchema()

print("Sample rows")
uber_df.show(6, truncate=False)

# -------------------------
# Question 1
# -------------------------
print("\nQuestion 1")
completed_trips_df = uber_df.groupBy("date_parsed").agg(
    func.sum("completed_trips").alias("total_completed_trips")
)

answer1 = (
    completed_trips_df
    .where(func.col("date_parsed").isNotNull())
    .orderBy(func.desc("total_completed_trips"))
    .first()
)

print(answer1)

# -------------------------
# Question 3
# -------------------------
print("\nQuestion 3")
answer3 = (
    uber_df.groupBy("time")
    .agg(func.sum("requests").alias("total_requests"))
    .orderBy(func.desc("total_requests"))
    .first()
)
print(answer3)

# -------------------------
# Question 4
# -------------------------
print("\nQuestion 4")
total_zero_count = uber_df.agg(func.sum("zeroes").alias("zero_count")).first()["zero_count"]

# -------------------------
# Weekend/late-night filter
# Spark dayofweek: 1=Sun ... 7=Sat
# Fri=6, Sat=7, Sun=1
# -------------------------

weekend_zero_count = uber_df.filter(
    ((func.col("time") > 17) & (func.dayofweek("date_parsed") == 6)) |
    (func.dayofweek("date_parsed") == 7) |
    ((func.col("time") < 3) & (func.dayofweek("date_parsed") == 1))
).agg(func.sum("zeroes").alias("weekend_zero_count")).first()["weekend_zero_count"]

print("Percentage of weekend zeroes", 100*weekend_zero_count/total_zero_count)

print("\nQuestion 5")
"""
can’t use a window function inside an aggregate function
First you should do the aggregation and the apply the window
"""

window = Window.orderBy(func.col("time").desc()).rowsBetween(0,3)

hourly_df = uber_df.groupBy("time").agg(func.sum("requests").alias("total_request_in_hour")).orderBy("time")

result = hourly_df.withColumn("window_start",func.col("time")).withColumn("window_end",func.min("time").over(window)).withColumn("total_request_in_hour",func.sum("total_request_in_hour").over(window)).withColumn("time_interval",func.concat_ws("-",func.col("window_start"),func.col("window_end")))

result.orderBy(func.desc("total_request_in_hour")).limit(1).select("time_interval","total_request_in_hour").show()

print("\nQuestion 8")

requests_col = func.col("requests")
time_col = func.col("time")
zeros_col = func.col("zeroes")

uber_df.withColumn("zero_to_requests",func.when(requests_col==0,func.lit(None)).otherwise(func.round(zeros_col/requests_col,2))).orderBy("date","time").select("zero_to_requests","date","time").show()
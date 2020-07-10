#!/usr/bin/env python3

import sys

try:
    # Load libraries
    from pyspark.context import SparkContext
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.feature import (OneHotEncoderEstimator, StringIndexer,
                                    VectorAssembler)
    from pyspark.ml.pipeline import Estimator, Transformer
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml.linalg import Vectors

    print ("Successfully imported Libraries")

except ImportError as e:

    sys.exit(1)

spark = SparkSession.builder.appName('price').getOrCreate()

def data_processing(df):
    '''
    :param data: A PySpark dataframe
    :return: A preprocessed data that has been cleaned, indexed and assembled
    '''
    df.createOrReplaceTempView("data")

    processed_data = spark.sql("""
    select
        host_id,
        price,
        bathrooms,
        bedrooms,
        room_type,
        property_type,
        case when host_is_superhost = True
            then 1.0
            else 0.0
        end as host_is_superhost,
        accommodates,
        cancellation_policy,
        minimum_nights,
        maximum_nights,
        availability_30,
        availability_60,
        availability_90,
        availability_365,
        case when security_deposit is null
            then 0.0
            else security_deposit
        end as security_deposit,
        case when number_of_reviews is null
            then 0.0
            else number_of_reviews
        end as number_of_reviews,
        case when extra_people is null
            then 0.0
            else extra_people
        end as extra_people,
        case when instant_bookable = True
            then 1.0
            else 0.0
        end as instant_bookable,
        case when cleaning_fee is null
            then 0.0
            else cleaning_fee
        end as cleaning_fee,
        case when review_scores_rating is null
            then 0.0
            else review_scores_rating
        end as review_scores_rating,
        case when review_scores_accuracy is null
            then 0.0
            else review_scores_accuracy
        end as review_scores_accuracy,
        case when review_scores_cleanliness is null
            then 0.0
            else review_scores_cleanliness
        end as review_scores_cleanliness,
        case when review_scores_checkin is null
            then 0.0
            else review_scores_checkin
        end as review_scores_checkin,
        case when review_scores_communication is null
            then 0.0
            else review_scores_communication
        end as review_scores_communication,
        case when review_scores_location is null
            then 0.0
            else review_scores_location
        end as review_scores_location,
        case when review_scores_value is null
            then 0.0
            else review_scores_value
        end as review_scores_value,
        case when square_feet is not null and square_feet > 100
            then square_feet
            when (square_feet is null or square_feet <=100) and (bedrooms is null or bedrooms = 0)
            then 350.0
            else 380 * bedrooms
        end as square_feet,
        case when bathrooms >= 2
            then 1.0
            else 0.0
        end as n_bathrooms_more_than_two,
        case when amenity_wifi = True
            then 1.0
            else 0.0
        end as amenity_wifi,
        case when amenity_heating = True
            then 1.0
            else 0.0
        end as amenity_heating,
        case when amenity_essentials = True
            then 1.0
            else 0.0
        end as amenity_essentials,
        case when amenity_kitchen = True
            then 1.0
            else 0.0
        end as amenity_kitchen,
        case when amenity_tv = True
            then 1.0
            else 0.0
        end as amenity_tv,
        case when amenity_smoke_detector = True
            then 1.0
            else 0.0
        end as amenity_smoke_detector,
        case when amenity_washer = True
            then 1.0
            else 0.0
        end as amenity_washer,
        case when amenity_hangers = True
            then 1.0
            else 0.0
        end as amenity_hangers,
        case when amenity_laptop_friendly_workspace = True
            then 1.0
            else 0.0
        end as amenity_laptop_friendly_workspace,
        case when amenity_iron = True
            then 1.0
            else 0.0
        end as amenity_iron,
        case when amenity_shampoo = True
            then 1.0
            else 0.0
        end as amenity_shampoo,
        case when amenity_hair_dryer = True
            then 1.0
            else 0.0
        end as amenity_hair_dryer,
        case when amenity_family_kid_friendly = True
            then 1.0
            else 0.0
        end as amenity_family_kid_friendly,
        case when amenity_dryer = True
            then 1.0
            else 0.0
        end as amenity_dryer,
        case when amenity_fire_extinguisher = True
            then 1.0
            else 0.0
        end as amenity_fire_extinguisher,
        case when amenity_hot_water = True
            then 1.0
            else 0.0
        end as amenity_hot_water,
        case when amenity_internet = True
            then 1.0
            else 0.0
        end as amenity_internet,
        case when amenity_cable_tv = True
            then 1.0
            else 0.0
        end as amenity_cable_tv,
        case when amenity_carbon_monoxide_detector = True
            then 1.0
            else 0.0
        end as amenity_carbon_monoxide_detector,
        case when amenity_first_aid_kit = True
            then 1.0
            else 0.0
        end as amenity_first_aid_kit,
        case when amenity_host_greets_you = True
            then 1.0
            else 0.0
        end as amenity_host_greets_you,
        case when amenity_translation_missing_en_hosting_amenity_50 = True
            then 1.0
            else 0.0
        end as amenity_translation_missing_en_hosting_amenity_50,
        case when amenity_private_entrance = True
            then 1.0
            else 0.0
        end as amenity_private_entrance,
        case when amenity_bed_linens = True
            then 1.0
            else 0.0
        end as amenity_bed_linens,
        case when amenity_refrigerator = True
            then 1.0
            else 0.0
        end as amenity_refrigerator
    from data
    where bedrooms is not null
    """)

    processed_data = processed_data.na.drop()

    cat_cols = [f.name for f in processed_data.schema.fields if isinstance(f.dataType, StringType)]
    num_cols = [f.name for f in processed_data.schema.fields if isinstance(f.dataType, IntegerType)]
    decimal_cols = [f.name for f in processed_data.schema.fields if isinstance(f.dataType, DecimalType)]
    double_cols = [f.name for f in processed_data.schema.fields if isinstance(f.dataType, DoubleType)]
    num_features = num_cols + decimal_cols + double_cols
    dataset_imputed = processed_data.persist()

    stages = []
    for x in cat_cols:
        cats_indexer = StringIndexer(inputCol=x, outputCol=x + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[cats_indexer.getOutputCol()],
                                            outputCols=[x + "encode"])
        stages += [cats_indexer, encoder]

    assembler_inputs = [c + "encode" for c in cat_cols] + num_features
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    stages += [assembler]
    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(dataset_imputed)
    df = pipeline_model.transform(dataset_imputed)
    
    return df

raw_data = spark.read.parquet('data/listings_processed.parquet')
clean_data = data_processing(raw_data)

gbt = GBTRegressor(featuresCol='features', labelCol='price', maxIter=10)
gbt_model = gbt.fit(clean_data)
gbt_model.write().overwrite().save("models")
print ("Successfully saved model")

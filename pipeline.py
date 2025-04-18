import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role

role = get_execution_role()
sess = sagemaker.Session()
bucket = sess.default_bucket()
prefix = "house-price-prediction"

sklearn_estimator = SKLearn(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    py_version="py3",
    hyperparameters={
        "n-estimators": 100,
        "min-samples-leaf": 3
    },
    source_dir=".",
    output_path=f"s3://{bucket}/{prefix}/output"
)

sklearn_estimator.fit()

predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    entry_point="inference.py",
    source_dir="."
)


sample_input = [[8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]]
response = predictor.predict(sample_input)
print(f"Predicted house price: {response}")

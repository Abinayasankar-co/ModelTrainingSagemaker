import boto3
import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput

s3 = boto3.client('s3')
bucket_name = os.environ['S3_BUCKET_NAME']
image_folder = os.environ['LOCAL_IMAGE_FOLDER']
s3_prefix = os.environ['S3_PREFIX']
role = get_execution_role()
sess = sagemaker.Session()

for root, dirs, files in os.walk(image_folder):
    for file in files:
        path = os.path.join(root, file)
        s3.upload_file(path, bucket_name, f"{s3_prefix}/{file}")

s3_path = f"s3://{bucket_name}/{s3_prefix}"

estimator = TensorFlow(
    entry_point='train_script.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='2.12.0',
    py_version='py39',
    script_mode=True,
    hyperparameters={}
)

estimator.fit({'train': TrainingInput(s3_data=s3_path, content_type='application/x-image')})

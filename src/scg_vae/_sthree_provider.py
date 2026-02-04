import time

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


class S3Provider:
    """Provider for uploading artifacts to the mlflow bucket using boto3"""

    def __init__(self, bucket: str, max_retries=5):
        self.client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self.bucket = bucket
        self.max_retries = max_retries

    def upload_file(self, local_path, remote_path):
        """Upload local file to remote path in s3 with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.client.upload_file(local_path, self.bucket, remote_path)
                print(f"Successfully uploaded {local_path} to {self.bucket}/{remote_path}")
                return
            except ClientError as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise Exception(
                        f"Failed to upload {local_path} after {self.max_retries} attempts: {str(e)}"
                    ) from None

    def download_file(self, remote_path, local_path):
        """Download file from remote path in s3 to local path"""
        if not self.key_exists(remote_path):
            raise FileNotFoundError(f"File {remote_path} not found in bucket {self.bucket}")

        self.client.download_file(self.bucket, remote_path, local_path)

    def key_exists(self, remote_path):
        """Check if a key exists in the S3 bucket

        Args:
            remote_path (str): The path to check in S3

        Returns
        -------
            bool: True if the key exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"Key {remote_path} does not exist in bucket {self.bucket}")
                return False
            elif e.response["Error"]["Code"] == "403":
                print(f"No permission to access {remote_path} in bucket {self.bucket}")
                return False
            else:
                print(f"Error checking key {remote_path}: {str(e)}")
                return False

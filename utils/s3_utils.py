# utils/s3_utils.py

import boto3
from botocore.exceptions import ClientError
import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
from .config import config

# Setup logger
logger = logging.getLogger(__name__)

class S3Manager:
    """S3 Manager for handling all S3 operations"""
    
    def __init__(self):
        """Initialize S3 client with credentials from config"""
        try:
            # Get AWS config
            aws_config = config.aws_config
            
            # Validate required config
            if not all([
                aws_config.get('access_key_id'),
                aws_config.get('secret_access_key'),
                aws_config.get('region'),
                aws_config.get('bucket_name')
            ]):
                raise ValueError("Missing required AWS configuration")
            
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_config['access_key_id'],
                aws_secret_access_key=aws_config['secret_access_key'],
                region_name=aws_config['region']
            )
            
            self.bucket_name = aws_config['bucket_name']
            self.app_prefix = aws_config.get('app_prefix', 'streamlit-app')
            
            logger.info(f"✅ S3Manager initialized for bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3Manager: {e}")
            raise
    
    # ==================== Basic S3 Operations ====================
    
    def list_files(self, prefix: str = '', max_keys: int = 1000) -> List[Dict]:
        """
        List files in S3 bucket with optional prefix filter
        
        Args:
            prefix: S3 prefix to filter files
            max_keys: Maximum number of files to return
            
        Returns:
            List of file dictionaries with metadata
        """
        try:
            # Ensure prefix ends with / if provided
            if prefix and not prefix.endswith('/'):
                prefix += '/'
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Skip directory markers and .keep files
                    if obj['Key'].endswith('/') or obj['Key'].endswith('.keep'):
                        continue
                        
                    files.append({
                        'key': obj['Key'],
                        'name': obj['Key'].split('/')[-1],
                        'size': obj['Size'],
                        'size_mb': round(obj['Size'] / 1024 / 1024, 2),
                        'last_modified': obj['LastModified'],
                        'etag': obj.get('ETag', '').strip('"')
                    })
            
            logger.info(f"Listed {len(files)} files with prefix: {prefix}")
            return files
            
        except ClientError as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def get_folders(self, prefix: str = '') -> List[str]:
        """
        Get list of folders (common prefixes) in bucket
        
        Args:
            prefix: Parent prefix to search within
            
        Returns:
            List of folder names
        """
        try:
            # Ensure prefix ends with / if provided
            if prefix and not prefix.endswith('/'):
                prefix += '/'
                
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter='/'
            )
            
            folders = []
            if 'CommonPrefixes' in response:
                for prefix_info in response['CommonPrefixes']:
                    folder_path = prefix_info['Prefix']
                    folder_name = folder_path.rstrip('/').split('/')[-1]
                    folders.append(folder_name)
            
            return sorted(folders)
            
        except ClientError as e:
            logger.error(f"Error getting folders: {e}")
            return []
    
    def upload_file(self, file_content: bytes, key: str, content_type: str = None) -> Tuple[bool, str]:
        """
        Upload file to S3
        
        Args:
            file_content: File content as bytes
            key: S3 key (path) for the file
            content_type: MIME type of the file
            
        Returns:
            Tuple of (success: bool, result: str)
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_content,
                **extra_args
            )
            
            logger.info(f"Successfully uploaded file to: {key}")
            return True, key
            
        except ClientError as e:
            error_msg = f"Failed to upload file: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def download_file(self, key: str) -> Optional[bytes]:
        """
        Download file content from S3
        
        Args:
            key: S3 key of the file
            
        Returns:
            File content as bytes or None if error
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            content = response['Body'].read()
            logger.info(f"Successfully downloaded file: {key}")
            return content
            
        except ClientError as e:
            logger.error(f"Error downloading file {key}: {e}")
            return None
    
    def delete_file(self, key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            key: S3 key of the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            logger.info(f"Successfully deleted file: {key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting file {key}: {e}")
            return False
    
    def get_presigned_url(self, key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for file access
        
        Args:
            key: S3 key of the file
            expiration: URL expiration time in seconds (default 1 hour)
            
        Returns:
            Presigned URL or None if error
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key
                },
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {key}: {e}")
            return None
    
    def get_file_info(self, key: str) -> Optional[Dict]:
        """
        Get detailed file information
        
        Args:
            key: S3 key of the file
            
        Returns:
            Dictionary with file metadata or None if error
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            return {
                'size': response['ContentLength'],
                'size_mb': round(response['ContentLength'] / 1024 / 1024, 2),
                'content_type': response.get('ContentType', 'unknown'),
                'last_modified': response['LastModified'],
                'etag': response.get('ETag', '').strip('"'),
                'metadata': response.get('Metadata', {})
            }
            
        except ClientError as e:
            logger.error(f"Error getting file info for {key}: {e}")
            return None
    
    def file_exists(self, key: str) -> bool:
        """
        Check if file exists in S3
        
        Args:
            key: S3 key to check
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return True
        except ClientError:
            return False
    
   
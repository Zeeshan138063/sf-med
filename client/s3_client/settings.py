"""S3 client configuration settings."""

S3_RETRIES_MAX_ATTEMPTS: int = 10
S3_RETRIES_MODE: str = "standard"  # "legacy", "standard" or "adaptive"
S3_READ_TIMEOUT: int = 120
S3_CONNECT_TIMEOUT: int = 120

"""Mixin class for the S3Client."""
import logging
from functools import cached_property

from snorefox_med.client.s3_client.s3_client import S3Client


class S3ClientMixin:
    """S3ClientMixin."""

    def __init__(self: "S3ClientMixin") -> None:
        """S3 client for communication with S3 bucket."""
        self._logger = logging.getLogger(self.__class__.__qualname__)
        self._logger.info("Initialising S3 client...")

    @cached_property
    def _s3_client(
        self: "S3ClientMixin",
    ) -> S3Client:
        """S3 client for communication with S3 bucket.

        Returns:
            S3Client: S3 client for communication with S3 bucket.
        """
        return S3Client()

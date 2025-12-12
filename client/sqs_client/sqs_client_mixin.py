"""Mixin class for the SQSClient."""
import logging
from functools import cached_property

from snorefox_med.client.sqs_client.sqs_client import SQSClient


class SQSClientMixin:
    """SQSClientMixin."""

    def __init__(self: "SQSClientMixin") -> None:
        """SQS client for communication with an AWS SQS queue."""
        self._logger = logging.getLogger(self.__class__.__qualname__)
        self._logger.info("Initialising SQS client...")

    @cached_property
    def _sqs_client(
        self: "SQSClientMixin",
    ) -> SQSClient:
        """SQS client for communication with an AWS SQS queue.

        Returns:
            SQSClient: SQS client for communication with an AWS SQS queue.
        """
        return SQSClient()

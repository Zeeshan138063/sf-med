"""The SQSClient class."""
import logging

import boto3
from botocore.config import Config

RETRIES_SQS_MESSAGE: int = 5


class SQSClient:
    """A class for interacting with Amazon SQS."""

    def __init__(self: "SQSClient") -> None:
        """Initialize an SQSClient."""
        self._sqs = boto3.client(
            "sqs",
            config=Config(retries={"max_attempts": RETRIES_SQS_MESSAGE, "mode": "standard"}),
        )
        self._logger = logging.getLogger(self.__class__.__qualname__)

    def send_message(
        self: "SQSClient",
        message: str,
        sqs_queue_url: str,
        message_group_id: str,
        message_deduplication_id: str,
    ) -> dict[str, str | dict[str, str]]:
        """Sends a message to a SQS queue.

        Args:
            message (str): Message that should be send to the queue. Will be also used as MessageGroupId.
            sqs_queue_url (str): URL of the SQS queue to which the message should be send.
            message_group_id (str): Tag that specifies that a message belongs to a specific message group.
            message_deduplication_id (str): Token used for deduplication of sent messages.

        Returns:
            dict[str, str | dict[str, str]]: SQS response dict, defined like this:
            {
                "MD5OfMessageAttributes": "string",
                "MD5OfMessageBody": "string",
                "MD5OfMessageSystemAttributes": "string",
                "MessageId": "string",
                "SequenceNumber": "string"
            }
        """
        return self._sqs.send_message(  # type: ignore[no-any-return]
            QueueUrl=sqs_queue_url,
            MessageBody=message,
            DelaySeconds=0,
            MessageGroupId=message_group_id,
            MessageDeduplicationId=message_deduplication_id,
        )

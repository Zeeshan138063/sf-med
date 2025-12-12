"""Takes a POST request from the API and sends a sleep base id to the SQS work queue."""

import json
import logging
from datetime import UTC, datetime

from botocore.exceptions import ClientError

from snorefox_med.api_sqs_bridge.settings import (
    API_TOKEN,
    API_URL,
    HTTPX_RETRIES,
    SQS_QUEUE_URL,
)
from snorefox_med.client.snorefox_med_client.api_client_mixin import ApiClientMixin
from snorefox_med.client.sqs_client.sqs_client_mixin import SQSClientMixin
from snorefox_med.metadata.backend_metadata import BackendMetadataEvent, add_backend_metadata_event_to_sleep_base


class ReturnMessage:
    """Httpx return message consisting of a status code, headers and message that can be returned."""

    def __init__(
        self: "ReturnMessage",
        status_code: int,
        message: str | None = None,
    ) -> None:
        """Holds all info needed for a return message to the calling api.

        Args:
            status_code (int): Status code that should be sent back.
            message (str | None, optional): Message that should be attached to the body. Defaults to None.
        """
        self.status_code = str(status_code)
        self.body = json.dumps(message) if message is not None else None
        self.headers = {"Content-Type": "application/json"}

    @property
    def json(self: "ReturnMessage") -> dict[str, str]:
        """Dict that has a status code and body that can be interpreted by the calling api.

        :return: JSON response that will be sent back to the calling api. Example:
            {
                "statusCode": "200",
                "body": "Sent message successfully to SQS.",
                "headers": {"Content-Type": "application/json"}
            }
        :rtype: dict
        """
        if self.body is None:
            return {
                "statusCode": self.status_code,
                "headers": json.dumps(self.headers),
            }

        return {
            "statusCode": self.status_code,
            "body": self.body,
            "headers": json.dumps(self.headers),
        }


class ApiSqsBridge(SQSClientMixin, ApiClientMixin):
    """Get triggered by Brayn's api to send a sleep base id into a SQS queue."""

    def __init__(self: "ApiSqsBridge") -> None:
        """Constructor of the ApiSqsBridge."""
        self._logger = logging.getLogger(self.__class__.__qualname__)

        SQSClientMixin.__init__(self)

        ApiClientMixin.__init__(
            self,
            api_url=API_URL,
            api_token=API_TOKEN,
            httpx_retries=HTTPX_RETRIES,
        )

    def send_sleep_base_id_to_sqs(self: "ApiSqsBridge", sleep_base_id: int) -> ReturnMessage:
        """Sends a message with the given sleep base id to a SQS queue and adds metadata event.

        Args:
            sleep_base_id (int): Id of a stopped recording.

        Returns:
            ReturnMessage:  Object that holds all info for a return message (has a `statusCode` and `body`).
        """
        try:
            enqueue_time = datetime.now(UTC)
            response = self._sqs_client.send_message(
                sqs_queue_url=SQS_QUEUE_URL,
                message=json.dumps({"sleep_base_id": sleep_base_id}),
                message_group_id=str(sleep_base_id),
                message_deduplication_id=str(sleep_base_id),
            )
        except ClientError:
            self._logger.exception("[%s] Got a client error when trying to send message to SQS.", sleep_base_id)
            return ReturnMessage(500, "Error when sending message to SQS.")

        try:
            sqs_status_code = int(response["ResponseMetadata"]["HTTPStatusCode"])  # type: ignore[index]
        except (KeyError, TypeError):
            sqs_status_code = None

        acceptable_response = 200
        # if response is correct
        if sqs_status_code == acceptable_response:
            self._logger.info("[%s] Sent message successfully to SQS.", sleep_base_id)
        else:
            self._logger.error("[%s] Can't send message to SQS, got this response: %s", sleep_base_id, response)
            return ReturnMessage(500, "Can't send message to SQS.")

        try:
            add_backend_metadata_event_to_sleep_base(
                api_client=self._api_client,
                sleep_base_id=sleep_base_id,
                existing_backend_info=None,
                event=BackendMetadataEvent(
                    name="analysis-pipeline",
                    value="triggered",
                    occurred_at=enqueue_time,
                ),
                fetch_backend_info=True,
            )
        except Exception:
            exception_message = (
                f"Failed to add backend metadata event for sleep base {sleep_base_id} after sending message to SQS."
            )
            self._logger.exception(
                exception_message,
            )
            return ReturnMessage(500, exception_message)

        return ReturnMessage(200, "Message successfully sent to SQS and metadata event added.")

"""Entrypoint for the api sqs bridge."""
import json
import logging
from typing import TypedDict

from aws_lambda_typing.context import Context

from snorefox_med.api_sqs_bridge.api_sqs_bridge import ApiSqsBridge, ReturnMessage
from snorefox_med.api_sqs_bridge.settings import (
    IS_LOGGING_LEVEL_DEBUG,
)
from snorefox_med.utils import log_config

# set up the logger
log_config.setup_logging(debug=IS_LOGGING_LEVEL_DEBUG)


class ApiSqsBridgeInput(TypedDict):
    """Input for the api sqs bridge."""

    sleepBaseId: int
    state: str


class NestedApiSqsBridgeInput(TypedDict):
    """Nested api sqs bridge input (the api sends a message in this format)."""

    body: ApiSqsBridgeInput | str


def lambda_handler(event: NestedApiSqsBridgeInput, _: Context = None) -> dict[str, str]:
    """Extracts a sleep base id from the event body and sends a message with it to a SQS queue.

    This triggers an analysis of the recording with the given sleep base id.

    Args:
        event (NestedApiSqsBridgeInput | str): AWS lambda event, has the sleep base id and state of a recording.
        The state has to be "stopped", otherwise no message will be sent to the SQS queue. Example:
        {
            "body":
            {
                "sleepBaseId": 18,
                "state": "stopped"
            }
        }
        _ (Context, optional): AWS Lambda context object, not used here. Defaults to None.

    Returns:
        dict[str, str]: A json containing a status code and an optional return message in this format. Example:
        {
            "statusCode": "200",
            "body": "Sent message successfully to SQS."
        }
    """
    logger = logging.getLogger("entrypoint")

    body = event["body"]
    if isinstance(body, str):
        body = json.loads(body)
    if not (isinstance(body, dict) and "sleepBaseId" in body and "state" in body):
        error_message = "Malformed input."
        raise ValueError(error_message)

    sleep_base_id = body["sleepBaseId"]
    measurement_state = body["state"]

    api_sqs_bridge = ApiSqsBridge()
    # check whether the recording was stopped ("state": "stopped" in event)
    if measurement_state == "stopped":
        return api_sqs_bridge.send_sleep_base_id_to_sqs(sleep_base_id).json

    logger.info("[%s] Recording wasn't stopped, nothing to do.", sleep_base_id)
    return ReturnMessage(200, "Recording wasn't stopped, nothing to do.").json


if __name__ == "__main__":
    from sys import argv

    # parse the path to the json file from the command line
    number_of_cmd_arguments = 1
    if len(argv) - 1 != number_of_cmd_arguments:
        exception_message = "Exactly one argument needed."
        raise ValueError(exception_message)

    lambda_input: NestedApiSqsBridgeInput = {
        "body": {
            "sleepBaseId": int(argv[1]),
            "state": "stopped",
        },
    }

    lambda_handler(lambda_input)

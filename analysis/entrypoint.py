"""Entrypoint to the Snorefox Med Analysis component."""
import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
from aws_lambda_typing.context import Context

from snorefox_med.analysis.analysis import AnalysisOutput, SnorefoxAnalysis
from snorefox_med.analysis.settings import (
    IS_LOGGING_LEVEL_DEBUG,
    LOCAL_PATH_FOR_WRITES,
)
from snorefox_med.shared_analysis_report import chart
from snorefox_med.shared_analysis_report.result import AnalysisResults
from snorefox_med.utils import env_config, log_config
from snorefox_med.utils.utils import clear_directory

log_config.setup_logging(debug=IS_LOGGING_LEVEL_DEBUG)


class NdArrayEncoder(json.JSONEncoder):
    """JSONEncoder that converts numpy arrays to lists."""

    def default(self: "NdArrayEncoder", obj: object) -> object:
        """JSON encoding that works also for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class InputSleepBase(TypedDict):
    """Type for the sleep base data input for this module."""

    sleep_base_id: int
    audio_file_s3_uri: str
    start_timestamp: str
    end_timestamp: str
    updated_start_timestamp: str
    updated_end_timestamp: str


def lambda_handler(
    sleep_base_data: InputSleepBase,
    _: Context = None,
) -> dict[str, AnalysisResults | dict[str, list[float]] | str]:
    """Entrypoint to the Snorefox Med algorithm.

    Args:
        sleep_base_data (InputSleepBase): JSON object containing the input data in the form:
            {
                "sleep_base_id" : 12345,
                "audio_file_s3_uri" : "s3://bucket_name/object_name",
                "start_timestamp" : "2023-08-15 22:00:00",
                "end_timestamp" : "2023-08-16 06:00:00",
                "updated_start_timestamp" : "2023-08-15 22:00:00",
                "updated_end_timestamp" : "2023-08-16 06:00:00"
            }
        Note that `start_timestamp`, `end_timestamp` and `updated_end_timestamp` are not used at the moment.

    Returns:
        dict[str, AnalysisResults | dict[str, list[float]] | str]:
            ``AnalysisOutput`` object represented as a plain dictionary.
    """
    logger = logging.getLogger("entrypoint")
    try:
        # initialise and run the algorithm
        analysis = SnorefoxAnalysis(
            sleep_base_id=sleep_base_data["sleep_base_id"],
            audio_file_s3_uri=sleep_base_data["audio_file_s3_uri"],
            updated_start_timestamp=sleep_base_data["updated_start_timestamp"],
        )
        output: AnalysisOutput = analysis.run()
        # format the output
        return {
            "results": output["results"],
            "charts": chart.as_dict(output["charts"]),
            "updated_start_timestamp": output["updated_start_timestamp"],
        }
    finally:
        if env_config.get_environment_type() != env_config.EnvironmentType.LOCAL:
            logger.debug("Clearing ephemeral storage...")
            clear_directory(LOCAL_PATH_FOR_WRITES)


if __name__ == "__main__":
    from sys import argv

    # parse the path to the json file from the command line
    number_of_cmd_arguments = 1
    if len(argv) - 1 != number_of_cmd_arguments:
        exception_message = "Exactly one argument needed."
        raise ValueError(exception_message)

    # load the json input from a file (filename given as command line argument)
    json_input = json.loads(Path(argv[1]).read_text())

    # read the needed data into an appropriate object; malformed / missing data will raise Errors
    sleep_base_data_json = InputSleepBase(
        sleep_base_id=int(json_input["sleep_base_id"]),
        audio_file_s3_uri=json_input["audio_file_s3_uri"],
        start_timestamp=json_input["start_timestamp"],
        end_timestamp=json_input["end_timestamp"],
        updated_start_timestamp=json_input["updated_start_timestamp"],
        updated_end_timestamp=json_input["updated_end_timestamp"],
    )
    # run the analysis
    json_output = json.dumps(lambda_handler(sleep_base_data_json), indent=3, cls=NdArrayEncoder)

    # write the output to disk
    with (LOCAL_PATH_FOR_WRITES / Path(f"report_input_{json_input['sleep_base_id']}.json")).open(
        mode="w",
    ) as f:
        f.write(json_output)

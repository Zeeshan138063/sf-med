"""Snorefox Med analysis module."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
from librosa import get_samplerate

from snorefox_med.analysis.algorithm.algo_main import (
    SnorefoxAlgo as SnorefoxAlgorithm,
)
from snorefox_med.analysis.settings import (
    API_TOKEN,
    API_URL,
    DB_DATEFORMAT,
    HTTPX_RETRIES,
    LOCAL_PATH_FOR_WRITES,
    TIMEZONE,
)
from snorefox_med.client.s3_client.s3_client_mixin import S3ClientMixin
from snorefox_med.client.snorefox_med_client.api_client_mixin import (
    ApiClientMixin,
)
from snorefox_med.patient.patient import PatientBuilder
from snorefox_med.scoring.scoring import Scoring
from snorefox_med.shared_analysis_report.chart import ChartCollection
from snorefox_med.shared_analysis_report.result import AnalysisResults
from snorefox_med.utils.utils import time_to_str


class AnalysisOutput(TypedDict):
    """Analysis Component output class."""

    results: AnalysisResults
    charts: ChartCollection
    updated_start_timestamp: str


class SnorefoxAnalysis(ApiClientMixin, S3ClientMixin):
    """Class for Snorefox Med analysis."""

    def __init__(
        self: "SnorefoxAnalysis",
        sleep_base_id: int,
        audio_file_s3_uri: str,
        updated_start_timestamp: str,
    ) -> None:
        """Constructor of SnorefoxAnalysis.

        Args:
            sleep_base_id (int): database id of the recording that is to be analysed
            audio_file_s3_uri (str): uri (in S3) where the audio for the recording can be found
            updated_start_timestamp (str): updated start time of the recording: start of usable portion of the recording
        """
        self._logger = logging.getLogger(self.__class__.__qualname__)

        ApiClientMixin.__init__(
            self,
            api_url=API_URL,
            api_token=API_TOKEN,
            httpx_retries=HTTPX_RETRIES,
        )
        S3ClientMixin.__init__(self)

        self._sleep_base_id = sleep_base_id
        self._audio_url = audio_file_s3_uri

        self._start = datetime.strptime(
            updated_start_timestamp,
            DB_DATEFORMAT,
        ).replace(
            tzinfo=TIMEZONE,
        )

    def _download_audio(
        self: "SnorefoxAnalysis",
        local_path: None | Path = None,
    ) -> Path:
        """Downloads the audio file on which further analysis can be performed.

        The local path at which the file should be saved can be either given directly or
        via the `TMPDIR` environment variable.

        Args:
            local_path (Path): Optional local path where to save the file,
                default to the environment variable `TMPDIR`

        Returns:
            Path: Path to the downloaded file.
        """
        if local_path is None:
            local_path = LOCAL_PATH_FOR_WRITES

        self._logger.info("Downloading recording audio from S3...")
        local_file_path = local_path / Path(self._audio_url).name
        download_path = self._s3_client.download_with_uri(self._audio_url, local_file_path)

        if isinstance(download_path, list):
            error_message = (
                "Multiple files downloaded instead of a single one! Incorrect `audio_uri` provided as input."
            )
            raise ValueError(error_message)  # noqa: TRY004

        return download_path

    def _run_analysis(self: "SnorefoxAnalysis", path_to_audio_file: Path) -> SnorefoxAlgorithm:
        """Run a Snorefox Med analysis on a given audio file.

        The results of the analysis will be saved in the `snorefox_algorithm` object.

        Args:
            path_to_audio_file (Path): Path to an audio file.

        Returns:
            SnorefoxAlgorithm: object that holds the results of the Snorefox algorithm run.
        """
        self._logger.info("Running Snorefox Med algorithm on an audio recording...")
        # initialise the algorithm class and let it run
        snorefox_algorithm = SnorefoxAlgorithm(
            path_to_audio_file,
            sample_rate=get_samplerate(path_to_audio_file),
        )
        snorefox_algorithm.compute_ahi_and_other_metrics()

        return snorefox_algorithm

    def _get_algorithm_results(
        self: "SnorefoxAnalysis",
        snorefox_algorithm: SnorefoxAlgorithm,
    ) -> AnalysisResults:
        """Creates a result dictionary that has all the needed info for creating reports and frontend display.

        Args:
            snorefox_algorithm (SnorefoxAlgorithm): object that holds the results of the Snorefox algorithm run.

        Returns:
            AnalysisResults: typed result dictionary
        """
        self._logger.info("Getting patient info for the result dictionary...")
        patient = PatientBuilder.from_sleep_base_id(
            sleep_base_id=self._sleep_base_id,
            api_client=self._api_client,
        )
        score = Scoring(
            api_client=self._api_client,
            user_id=patient.user_id,
            is_premium_user=patient.is_premium_user,
            ahi=snorefox_algorithm.AHI,
            usable_by_algorithm=snorefox_algorithm.usable_by_algorithm,
        )

        self._logger.info("Constructing the result dictionary from patient info and algorithm results...")
        return AnalysisResults(
            {
                "is_questionnaire_filled": score.questionnaire_score is not None,
                "questionnaire_score": score.questionnaire_score,
                "questionnaire_risk": score.questionnaire_risk,
                "questionnaire_name": score.questionnaire_name,
                "usable_by_algorithm": snorefox_algorithm.usable_by_algorithm,
                "ahi": score.ahi,
                "ahi_risk": score.ahi_risk,
                "rec_risk": score.rec_risk,
                "analysis_result": score.analysis_result,
                "mean_loudness": snorefox_algorithm.meanLoudness,
                "snoring_rate": snorefox_algorithm.snoringRate * 100,
                "total_length": snorefox_algorithm.totalSignalTime,
                "snoring_length": snorefox_algorithm.totalSnoringTimeSeconds,  # type: ignore[has-type]
                "total_length_str": time_to_str(
                    snorefox_algorithm.totalSignalTime,
                ),
                "snoring_length_str": time_to_str(
                    snorefox_algorithm.totalSnoringTimeSeconds,  # type: ignore[has-type]
                ),
                "invalid_length_str": time_to_str(
                    snorefox_algorithm.totalInvalidPeriods,
                ),
                "silent_length_str": time_to_str(
                    snorefox_algorithm.totalSilentInvalidPauses,
                ),
                "noise_length_str": time_to_str(
                    snorefox_algorithm.totalNoiseTime,
                ),
                "is_premium_user": patient.is_premium_user,
                "phone_brand": patient.device.brand,
                "phone_model": patient.device.model,
                "known_phone": patient.device.known,
                "whitelisted": patient.device.whitelisted,
                "sleep_base_id": self._sleep_base_id,
            },
        )

    def run(self: "SnorefoxAnalysis") -> AnalysisOutput:
        """Downloads an audio file, runs a Snorefox Med analysis on it and creates output for the report module.

        Returns:
            AnalysisOutput: Output of the Analysis component.
        """
        local_audio_file_path = self._download_audio()
        snorefox_algorithm = self._run_analysis(local_audio_file_path)

        self._logger.info("Analysis done.")
        return AnalysisOutput(
            results=self._get_algorithm_results(snorefox_algorithm),
            charts={
                "apnea": np.array(
                    snorefox_algorithm.apnea_chart_data.tolist(),
                ),
                "snore": np.array(
                    snorefox_algorithm.snore_chart_data.tolist(),
                ),
                "noise": np.array(
                    snorefox_algorithm.invalid_chart_data.tolist(),
                ),
            },
            updated_start_timestamp=self._start.strftime(DB_DATEFORMAT),
        )

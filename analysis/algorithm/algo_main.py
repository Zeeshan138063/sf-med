import logging
from pathlib import Path
from typing import Iterable, Union

import librosa
import numpy as np
import scipy.signal as sigAPI

import snorefox_med.analysis.algorithm.parameters as Params
from snorefox_med.analysis.algorithm import algo_utils as algoUtils


class SnorefoxAlgo:
    def __init__(
        self,
        stream,
        sample_rate=10200,
    ) -> None:
        self.streamfile = stream
        self.sample_rate = sample_rate

        # this initialises the logging
        self._logger = logging.getLogger(self.__class__.__qualname__)

    def compute_ahi_and_other_metrics(self) -> None:
        try:
            if isinstance(self.streamfile, str) or isinstance(self.streamfile, Path):
                # this is when the program should stream an indicated file, not read it in one fell swoop

                self._logger.info(
                    "Setting up parameters for trimming audio at the start / end of recording when streaming.",
                )

                (
                    offsetToStartAt,
                    newAudioDurationInSeconds,
                    timeTrimmedOutAtTheEnd,
                ) = self.checkAndAdaptStreamingParametersForAudioFile(self.streamfile)

                def audioToAnalyze():
                    return algoUtils.openAudioStreamAtSnorefoxParameters(
                        self.streamfile,
                        offsetToStartAt=offsetToStartAt,
                        durationToRead=newAudioDurationInSeconds,
                    )

            else:
                # trim the audio stream if it is long enough
                samplestoCutOut = Params.enforcedSilencePeriodLengthInSeconds * self.sample_rate
                if len(self.streamfile) / self.sample_rate >= Params.minUsableAudioDuration:
                    self._logger.info("Trimming audio at the start / end of recording.")
                    audioToAnalyze = self.streamfile[samplestoCutOut:(-samplestoCutOut):1]
                    offsetToStartAt = Params.enforcedSilencePeriodLengthInSeconds
                    timeTrimmedOutAtTheEnd = Params.enforcedSilencePeriodLengthInSeconds
                else:
                    self._logger.info(
                        "Not trimming audio at the start / end of recording -- audio too short.",
                    )
                    audioToAnalyze = self.streamfile
                    offsetToStartAt = 0
                    timeTrimmedOutAtTheEnd = 0

            self._logger.info("Extracting custom onsets.")
            # extract onsets from the trimmed audiostream
            allOnsets, STFTTimeAxisAll = algoUtils.extractCustomOnsets(
                audioToAnalyze,
                self.sample_rate,
            )

            allBreathingOnsets = allOnsets["breathing"]
            allSnoringOnsets = allOnsets["snoring"]
            allSnoringOnsetsLow = allOnsets["snoring_low"]
            STFTTimeAxis = STFTTimeAxisAll["breathing"]
            STFTTimeAxisSnoring = STFTTimeAxisAll["snoring"]
            del allOnsets

            # account for period to trim out / silence at beginning / end of file
            STFTTimeAxis += offsetToStartAt
            STFTTimeAxisSnoring += offsetToStartAt

            self._logger.info(
                "Identifying valid regions and silent pauses for breathing/snoring event detection.",
            )
            (
                validBreathingPeriodsStartsEnds,
                invalidBreathingPeriodsStartsEnds,
                silentPauses,
                _,
                _,
            ) = algoUtils.detect_valid_breathing_regions(
                STFTTimeAxisAll,
                allBreathingOnsets,
                STFTTimeAxis,
                allSnoringOnsetsLow,
                STFTTimeAxisSnoring,
                offsetToStartAt,
            )
            del allSnoringOnsetsLow

            allBreathingOnsets = algoUtils.onset_normalization(allBreathingOnsets)
            self.totalValidPeriods = algoUtils.computeTotalIntervalDurations(validBreathingPeriodsStartsEnds)
            self.totalInvalidPeriods = algoUtils.computeTotalIntervalDurations(invalidBreathingPeriodsStartsEnds)

            if self.totalValidPeriods == 0:
                # can't do anything in this case -- no processing is possible on audio signal with exactly zero
                #   valid period
                raise ValueError("No valid periods could be detected.")

            self._logger.info("Detecting breathing events.")
            (
                breathingEventStartEnds,
                breathingEventStrengths,
                _,
            ) = algoUtils.detectBreathingOrSnoringEvents(
                allBreathingOnsets,
                STFTTimeAxis,
                validBreathingPeriodsStartsEnds=validBreathingPeriodsStartsEnds,
                invalidBreathingPeriodsStartsEnds=invalidBreathingPeriodsStartsEnds,
            )

            self._logger.info("Detecting snoring events.")
            (
                snoringEventStartEnds,
                snoringEventStrengths,
                _,
            ) = algoUtils.detectBreathingOrSnoringEvents(
                allSnoringOnsets,
                STFTTimeAxisSnoring,
                validBreathingPeriodsStartsEnds=validBreathingPeriodsStartsEnds,
                invalidBreathingPeriodsStartsEnds=invalidBreathingPeriodsStartsEnds,
                relativeBreathingOrSnoringThreshold=Params.relativeSnoringThreshold,
            )

            self._logger.info("Calculating apnea metrics.")
            breathingEventStrengths = algoUtils.onset_normalization(np.array(breathingEventStrengths))
            self.computeApneasAndMetricsOnBreathingDetections(
                breathingEventStartEnds,
                breathingEventStrengths,
                validBreathingPeriodsStartsEnds=validBreathingPeriodsStartsEnds,
                invalidBreathingPeriodsStartsEnds=invalidBreathingPeriodsStartsEnds,
                silentPauses=silentPauses,
                breathingOnsetSignal=allBreathingOnsets,
                breathingOnsetSignalTimeAxis=STFTTimeAxis,
            )

            apneaDurations = [end - start for (start, end) in self.apneaStartEnds]
            self.totalApneaDuration = sum(apneaDurations)

            self.apneaDetections = [
                ((apneaBoundaries[0] + apneaBoundaries[1]) / 2) for apneaBoundaries in self.apneaStartEnds
            ]

            self.totalSignalTime = (
                self.totalValidPeriods + self.totalInvalidPeriods + offsetToStartAt + timeTrimmedOutAtTheEnd
            )
            if len(self.validPeriods) > 0:
                (
                    self.totalSnoringTimeSeconds,
                    _,
                    snoringIntervals,
                    snoringStrength,
                ) = self.totalSnoringLength(
                    snoringEventStartEnds,
                    snoringEventStrengths,
                    self.validPeriods,
                    int(self.totalSignalTime / 60),
                )
                self.apnea_chart_data = np.array(
                    algoUtils.minuteMovingAVGFilter(
                        self.apneaDetections,
                        int(self.totalSignalTime / 60),
                    ),
                )
                self.snore_chart_data = np.array(snoringStrength)

                self.snoringRate = self.totalSnoringTimeSeconds / (self.totalSignalTime)

                self.meanLoudness = np.nanmean(snoringEventStrengths)
                if np.isnan(self.meanLoudness):
                    self.meanLoudness = 0
            else:
                (
                    self.totalSnoringTimeSeconds,
                    self.snoringIntervals,
                    self.snoringStrength,
                ) = (0, [], [])
                self.apnea_chart_data = np.empty(0)
                self.snore_chart_data = np.empty(0)
                self.snoringRate = None
                self.meanLoudness = 0

            # load the audio array (again) to extract the invalid intervals
            # TODO do this while getting the breathing and snoring sounds
            audio, _ = librosa.load(self.streamfile, sr=self.sample_rate)

            # get the strengths / loudness for all the invalid intervals while
            # converting start and end times from seconds (float) to the stream indices
            self.invalidStrengths = [
                audio[round(start * self.sample_rate) : round(end * self.sample_rate)]
                for (start, end) in self.invalidPeriods
            ]
            del audio

        except ValueError:
            self._logger.exception("Audio stream is invalid.")
            self.AHI = -1
            self.usable_by_algorithm = False
            self.apneaStartEnds = []
            self.apneaDetections = []
            self.totalApneaDuration = -1
            self.totalInvalidPeriods = -1
            self.totalValidPeriods = -1
            self.totalNoiseTime = -1
            self.silentInvalidPauses = []
            self.totalSilentInvalidPauses = -1
            self.validPeriods = []
            self.invalidPeriods = []
            self.invalidStrengths = []
            self.totalSignalTime = -1
            self.apnea_chart_data = np.empty(0)
            self.snore_chart_data = np.empty(0)
            self.meanLoudness = 0
            self.snoringRate = 0
            self.totalSnoringTimeSeconds = 0
            self.totalSignalTime = 0
            self.totalSilentInvalidPauses = 0
            self.totalNoiseTime = 0

        self.calculate_invalid_plot()

    def calculate_invalid_plot(self):
        # get start and end times of all invalid periods
        invalidPeriods = [(int(s / 60), int(e / 60)) for (s, e) in self.invalidPeriods]

        invalid_strengths_per_minute = []
        # iterate over the start and end times of the invalid intervals (and an index)
        for idx, (start, end) in enumerate(invalidPeriods):
            # check whether we have an interval of at least length 1
            if end > start:
                invalid_strengths_per_minute.append(
                    [
                        # choose the 90th percentile of loudness per minute
                        np.percentile(values, 90) if len(self.invalidStrengths[idx]) != 0 else 0.0013
                        # split the array into the given intervals and fetch the loudness / strength
                        for values in np.array_split(
                            self.invalidStrengths[idx],
                            end - start,
                        )
                    ],
                )

        # make sure that we look at correctly set periods
        invalidPeriods = [(s, e) for (s, e) in invalidPeriods if e > s]

        # add an empty interval to circumvent checks against empty lists
        if not len(invalidPeriods):
            invalidPeriods = [[0, 0]]
            invalid_strengths_per_minute = [[0.0]]

        # convert to a numpy array
        invalidPeriods = np.array(invalidPeriods)
        # prepare a list that fits all invalid periods
        invalid_chart_data = np.zeros(max(invalidPeriods[:, 1]))
        # display a bin for every invalid minute
        for idx, (s, e) in enumerate(invalidPeriods):
            invalid_chart_data[s:e] = [
                abs(val) for val in invalid_strengths_per_minute[idx] if len(invalid_strengths_per_minute[idx]) != 0
            ]

        # multiply magic number to go from floats to 16 bit integer range
        self.invalid_chart_data = np.array([int(s * 32768) for s in invalid_chart_data])

        self.apnea_chart_data = self.apnea_chart_data * 60

        self._logger.info(
            f"The recording is {'NOT ' if not self.usable_by_algorithm else ''}usable by the algorithm.",
        )

    def computeApneasAndMetricsOnBreathingDetections(
        self,
        breathingEventStartEnds,
        breathingEventStrengths,
        validBreathingPeriodsStartsEnds=None,
        invalidBreathingPeriodsStartsEnds=None,
        silentPauses=None,
        enforceApneaAmplitudeReduction: bool = True,
        enforceApneaAmplitudeIncrease: bool = True,
        breathingOnsetSignal=None,
        breathingOnsetSignalTimeAxis=None,
        breathingOnsetSignalTimeAxisIsUniform=True,
    ):
        """Function used to detect apneas/hypopneas and asses AHI / other metrics on pre-detected breathing/snoring
        events

        :param breathingEventStartEnds: start and end time instants (in seconds) of the breathing/snoring events
        :type breathingEventStartEnds: list of 2-element lists or pair-tuples, or 2D numpy array
            (preferably of floats, or of ints)
        :param breathingEventStrengths: combined/composite strength/salience of each of the events
        :type breathingEventStrengths: list or 1D numpy array (preferably of floats, or of ints)
        :param validBreathingPeriodsStartEnds: time intervals for which the bio-signal can be considered valid. Leave
         set to None if no restriction to valid   periods in the bio-signal is desired. Defaults to None
        :type validBreathingPeriodsStartEnds: list of 2-element lists of pair-tuples, or 2D numpy array
            (preferably of floats, or of ints), optional
        :param invalidBreathingPeriodsStartEnds: time intervals for which the bio-signal is considered invalid. Left to
            None simultaneously with validBreathingPeriodsStartEnds. Defaults to None
        :type invalidBreathingPeriodsStartsEnds: list of 2-element lists of pair-tuples, or 2D numpy array
            (preferably of floats, or of ints), optional
        :param silentPauses: time intervals where only low-volume noise is audible (e.g. silent breathing etc). Leave
            set to None if no detection / treatment of silent periods is desired. Is simultaneously set to None
            with invalidBreathingPeriodsStartsEnds and validBreathingPeriodsStartEnds. Defaults to None
        :type silentPauses: list of 2-element lists of pair-tuples, or 2D numpy array
            (preferably of floats, or of ints), optional
        :param enforceApneaAmplitudeReduction: True if you want to enable enforcement of breathing/snoring amplitude
            (strength/salience) reduction for detected apneas (recommended). Defaults to True
        :type enforceApneaAmplitudeReduction: bool, optional
        :param enforceApneaAmplitudeIncrease: True if you want to enable enforcement of breathing/snoring amplitude
            (strength/salience) increase for detected apneas. Defaults to False
        :type enforceApneaAmplitudeIncrease: bool, optional
        :param breathingOnsetSignal: breathing onset bio-signal. Provide it if you want to use apnea amplitude
            reduction enforcement. Defaults to None
        :type breathingOnsetSignal: list, numpy array or similar, optional
        :param breathingOnsetSignalTimeAxis: time axis (in seconds) corresponding to breathing onset signal. Should
            be provided whenever the breathingOnsetSignal variable is provided. Defaults to None
        :type breathingOnsetSignalTimeAxis: list, numpy array or similar, optional
        :return:
            apneaDetections: apnea/hypopnea detections as described by a single characteristic time in seconds
            apneaStartEnds: apnea/hypopnea detections as described by their start/end times in seconds
            apneaDurations: duration of each apnea/hyponpnea (in seconds)
            totalApneaDuration: total time the patient was found to suffer form apnea given the breathing/snoring events
            invalidPeriods: time intervals for which the bio-signal is considered invalid. This is returned even when
                invalidBreathingPeriodsStartsEnds was not originally provided, as there is a rudimentary way to produce
                basic valid/invalid period segmentation.
            totalInvalidPeriods: total time in signal considered to be invalid, in which no apnea/hypopneas are detected
            validPeriods: time intervals for which the bio-signal is considered invalid. Also returned even when no
                validBreathingPeriodsStartEnds were originally provided.
            totalValidPeriods: total time in signal considered to be valid, in which apnea/hypopneas are detected
            silentInvalidPauses: time intervals in signal was found to be both invalid and also largely silent.
                These are practically assumed to involve no apneas (AHI=0)
            totalSilentInvalidPauses: total time (in s) in which signal was found to be both invalid and also largely
                silent
            totalNoisyInvalidPauses: total time (in s) in which signal was found to be both invalid
                AND also loud / non-silent / noisy
            AHI: apnea-hypopnea index as estimated based on the number of detected apneas/hypopneas
            apneaToTotalAHITimeBasis: ratio (in %, form 0 to 100) of total time suffering from apnea
                (totalApneaDuration)to the total duration of the "valid" regions in the signal + the total duration of
                any silent invalid pauses
        :rtype:
            apneaDetections: list of floats
            apneaStartEnds: list of pair-tuples
            apneaDurations: list of floats
            totalApneaDuration: float
            invalidPeriods: list of pair-tuples (of floats)
            totalInvalidPeriods: float
            validPeriods: list of pair-tuples (of floats)
            totalValidPeriods: float
            silentInvalidPauses: list of pair-tuples (of floats)
            totalSilentInvalidPauses: float
            totalNoisyInvalidPauses: float
            AHI: float
            apneaToTotalAHITimeBasis: float
        """
        if enforceApneaAmplitudeReduction or enforceApneaAmplitudeIncrease:
            breathingOnsetSignalTimeAxisMinimum = np.nanmin(
                breathingOnsetSignalTimeAxis,
            )
            breathingOnsetSignalTimeAxisMaximum = np.nanmax(
                breathingOnsetSignalTimeAxis,
            )

        self._logger.info("Detecting apnea and other metrics based on breathing events")

        self._logger.info("Computing candidate apneas")
        # this needs to be forced to be a sorted list, not in random order, and needs to be based
        # on the same breathing event order as breathingEventStartEnds
        breathingEventGapBoundaries = [
            (breathingEventStartEnds[n][1], breathingEventStartEnds[n + 1][0])
            for n in range(len(breathingEventStartEnds) - 1)
        ]

        if validBreathingPeriodsStartsEnds is not None:
            self._logger.info(
                "Restricting candidate apneas to those lying in pre-given valid regions",
            )
            # discarding breathing gaps if they have an invalid breathing region inside of them
            # -- to avoid detecting apneas in invalid regions
            breathingEventGapBoundaries = [
                (startTime, endTime)
                for (startTime, endTime) in breathingEventGapBoundaries
                if sum(
                    [1 for (start, end) in invalidBreathingPeriodsStartsEnds if start >= startTime and end <= endTime],
                )
                == 0
            ]
            self.validPeriods = validBreathingPeriodsStartsEnds
        else:
            self._logger.info(
                "Proceeding without restricting candidate apneas to those lying in pre-given valid regions",
            )
            self.validPeriods = None

        breathingEventGapDurations = [end - start for (start, end) in breathingEventGapBoundaries]

        if invalidBreathingPeriodsStartsEnds is not None:
            self.invalidPeriods = invalidBreathingPeriodsStartsEnds
        if invalidBreathingPeriodsStartsEnds is None:
            self._logger.info(
                "Detecting rudimentary invalid regions based on those with long breathing gaps",
            )
            # computing total of periods (in s) where no breaths are detected for long
            # -- these are additionally re-flagged as invalid

            self.invalidPeriods = [
                breathingEventGapBoundaries[n]
                for n, i in enumerate(breathingEventGapDurations)
                if i > Params.maxApneaLengthInSeconds
            ]

        self._logger.info("Computing total span/duration of invalid intervals")
        self.totalInvalidPeriods = algoUtils.computeTotalIntervalDurations(
            self.invalidPeriods,
        )
        if validBreathingPeriodsStartsEnds is not None:
            # checking which silent pauses fall under invalid regions
            self.silentInvalidPauses = algoUtils.siftIntervalsForThoseOverlappingAnyOtherInterval(
                silentPauses,
                self.invalidPeriods,
            )
        else:
            self.silentInvalidPauses = []

        # computing total of periods (in s) where enough breaths / periodic breathing are detected -- these are valid
        if validBreathingPeriodsStartsEnds is not None:
            self._logger.info("Computing total span/duration of valid intervals")
            self.totalValidPeriods = algoUtils.computeTotalIntervalDurations(
                self.validPeriods,
            )
            self._logger.info(
                "Computing total span/duration of silent invalid pause intervals",
            )
            self.totalSilentInvalidPauses = algoUtils.computeTotalIntervalDurations(
                self.silentInvalidPauses,
            )
        else:
            self.totalValidPeriods = (
                breathingOnsetSignalTimeAxis[-1]
                - breathingOnsetSignalTimeAxis[0]
                # total signal duration
            ) - self.totalInvalidPeriods  # minus ad-hoc detected invalid time
            self.totalSilentInvalidPauses = 0

        self.totalNoiseTime = self.totalInvalidPeriods - self.totalSilentInvalidPauses

        nonEmptySignal = self.totalValidPeriods > 0 and len(breathingEventStartEnds) > 0

        if nonEmptySignal:
            # it should check everything if it's desirable to detect both hypopneas and apneas,
            # but it can also explicitly exclude most hypopneas by pre-sifting the candidates for
            # those with a minimum duration
            minApneaLimitForPresifting = 0

            self._logger.info(
                "Enforcing (some of) basic apnea assumptions on apnea candidates",
            )
            self.apneaStartEnds = [
                breathingEventGapBoundaries[n]
                for n, i in enumerate(breathingEventGapDurations)
                if minApneaLimitForPresifting <= breathingEventGapDurations[n]
                and breathingEventGapDurations[n] <= Params.maxApneaLengthInSeconds
            ]
            del breathingEventGapDurations

            self._logger.info(
                "Enforcing minimum number of breathing/snoring events before and\
            after each apnea, and relative amplitude reduction",
            )
            newApneaStartEnds = []
            if enforceApneaAmplitudeReduction or enforceApneaAmplitudeIncrease:
                apneaAmplitudeDeviations = []
            else:
                apneaAmplitudeDeviations = None

            latestApneaBoundaries = (0, 0)
            for currentApneaBoundaries in self.apneaStartEnds:
                if not currentApneaBoundaries[1] <= latestApneaBoundaries[1]:
                    # checking for each apnea, how many breaths are detected in the maximal seconds taken by
                    # the defined Params.noOfPreviousBreathstoEnforceBeforeApnea, the duration is computed based
                    #  on the maximal time those would take. Restricting these "previous" breaths to those occuring
                    # after the previous apnea detection too
                    previousBreathDetectionsSinceLastApnea = [
                        {
                            "startEnd": (start, end),
                            "strength": breathingEventStrengths[index],
                        }
                        for index, (start, end) in enumerate(breathingEventStartEnds)
                        if start <= currentApneaBoundaries[0] and start >= latestApneaBoundaries[1]
                    ]
                    # since the apneas start and end at the starts of the inhales, comparing them
                    # against the START of breathing events here

                    previousBreathDetectionsInImmediateNeighborhood = (
                        algoUtils.siftBreathingEventsForSpecificPastOrFutureNumberAllowance(
                            previousBreathDetectionsSinceLastApnea,
                            currentApneaBoundaries[0],
                            Params.noOfPreviousBreathstoEnforceBeforeApnea,
                        )
                    )

                    # checking for the each apnea all upcoming breaths starting at or after its end
                    # but restricting the next breaths to those occurring within the current valid period (if used),
                    # to not have the apnea candidate consider breaths from different valid periods,
                    # because otherwise an apnea can fall inside an invalid region between valid
                    # regions -- which is not allowed
                    if validBreathingPeriodsStartsEnds is not None:
                        # checking in which valid region the apnea started
                        (
                            _,
                            _,
                            whichValidRegionsApneaBelongsTo,
                        ) = algoUtils.retainTimesWithinPredefinedWindows(
                            [currentApneaBoundaries[0]],
                            validBreathingPeriodsStartsEnds,
                        )

                        # assuming the apnea start -- e.g. the last breath end before it --
                        # can only occur in one valid region since they're distinct
                        if len(whichValidRegionsApneaBelongsTo) > 0:
                            whichValidRegionsApneaBelongsTo = whichValidRegionsApneaBelongsTo[0][0]
                        else:
                            whichValidRegionsApneaBelongsTo = 0

                        currentValidRegionEnd = validBreathingPeriodsStartsEnds[whichValidRegionsApneaBelongsTo][1]
                    else:
                        currentValidRegionEnd = np.inf

                    nextBreaths = [
                        {"startEnd": breathingEventStartEnd}
                        for breathingEventStartEnd in breathingEventStartEnds
                        if breathingEventStartEnd[0] >= currentApneaBoundaries[1]
                        and breathingEventStartEnd[1] <= currentValidRegionEnd
                    ]

                    # only keep the apnea if there are at least Params.noOfPreviousBreathstoEnforceBeforeApnea
                    # breaths in the corresponding period before
                    if (
                        len(previousBreathDetectionsInImmediateNeighborhood)
                        >= Params.noOfPreviousBreathstoEnforceBeforeApnea
                    ):
                        if enforceApneaAmplitudeReduction:
                            averageOnsetAmplitudeInPreviousNeighborhood = algoUtils.computeOnsetSignalBaselineAmplitude(
                                previousBreathDetectionsSinceLastApnea,
                                currentApneaBoundaries[0],
                                breathingOnsetSignal,
                                breathingOnsetSignalTimeAxis,
                                pastOrFuture="past",
                                onsetSignalTimeAxisMinimum=breathingOnsetSignalTimeAxisMinimum,
                                onsetSignalTimeAxisMaximum=breathingOnsetSignalTimeAxisMaximum,
                                onsetSignalTimeAxisIsUniform=breathingOnsetSignalTimeAxisIsUniform,
                            )
                        else:
                            averageOnsetAmplitudeInPreviousNeighborhood = None

                        (
                            currentApneaBoundaries,
                            latestApneaBoundaries,
                            newApneaStartEnds,
                            apneaAmplitudeDeviations,
                        ) = algoUtils.lookForwardForApneaClosureWithEnoughBreaths(
                            nextBreaths,
                            currentApneaBoundaries,
                            latestApneaBoundaries,
                            newApneaStartEnds,
                            apneaAmplitudeDeviations,
                            enforceApneaAmplitudeReduction=enforceApneaAmplitudeReduction,
                            enforceApneaAmplitudeIncrease=enforceApneaAmplitudeIncrease,
                            breathingOnsetSignal=breathingOnsetSignal,
                            breathingOnsetSignalTimeAxis=breathingOnsetSignalTimeAxis,
                            breathingOnsetSignalTimeAxisIsUniform=breathingOnsetSignalTimeAxisIsUniform,
                            breathingOnsetSignalTimeAxisMinimum=breathingOnsetSignalTimeAxisMinimum,
                            breathingOnsetSignalTimeAxisMaximum=breathingOnsetSignalTimeAxisMaximum,
                            averageOnsetAmplitudeInPreviousNeighborhood=averageOnsetAmplitudeInPreviousNeighborhood,
                        )

            self.apneaStartEnds = newApneaStartEnds
            del newApneaStartEnds

            self.apneaDurations = [i[1] - i[0] for i in self.apneaStartEnds]
            self._logger.info(
                "Finished enforcing all apnea assumptions on candidates",
            )

            self.totalApneaDuration = sum(self.apneaDurations)
            self.apneaDetections = [
                # TODO: (apnea_start + apnea_end) / 2 for (apnea_start, apnea_end) in self.apneaStartEnds
                ((apneaBoundaries[0] + apneaBoundaries[1]) / 2)
                for apneaBoundaries in self.apneaStartEnds
            ]

            if len(self.apneaStartEnds) != 0:
                self._logger.info(
                    "Detected a total of " + str(len(self.apneaStartEnds)) + " apneotic events",
                )
            else:
                self._logger.info("Could not detect any apneotic events")

        if nonEmptySignal:
            totalAHITimeBasis = self.totalValidPeriods + self.totalSilentInvalidPauses
            self.usable_by_algorithm = bool(totalAHITimeBasis >= Params.minimumAHITimeBasis)

            self.computeAHIFromDetections(
                self.apneaStartEnds,
                totalAHITimeBasis,
                apneasAndHypopneaAmplitudeDeviations=apneaAmplitudeDeviations,
            )
            self.apneaToTotalAHITimeBasis = 100 * self.totalApneaDuration / totalAHITimeBasis
        else:
            self.apneaDetections = []
            self.apneaStartEnds = []
            self.apneaDurations = []
            self.totalApneaDuration = -1
            self.AHI = -1
            self.apneaToTotalAHITimeBasis = -1
            self.usable_by_algorithm = False

    def computeAHIFromDetections(
        self,
        apneasAndHypopneaDetections: Iterable,
        totalTime: Union[int, float],
        apneasAndHypopneaAmplitudeDeviations=None,
    ):
        """Function used to compute an apnea-hyponea (AHI) or apnea (AI) index
        based on a number of hypopnea/apnea detections

        :param apneasAndHypopneaDetections: apnea and/or hypopnea detections (their times instants /intervals)
        :type apneasAndHypopneaDetections: list, set or 1D numpy array of floats, ints or intervals - anything with a
                                            length
        :param totalTime: total time (in seconds) on which the apneas are computed / time basis to use for the AHI/AI
        :type totalTime: float
        :param apneasAndHypopneaAmplitudeDeviations: amplitude deviations observed for each apnea. Supply if you
            want to use apnea amplitude deviation based weighting. Otherwise leave to None. Defaults to None.
        :type apneasAndHypopneaAmplitudeDeviations: list 1D numpy array of floats, optional
        :return: apnea-hyponea (AHI) or apnea (AI) index, depending on which detections were supplied
        :rtype: float
        """
        if len(apneasAndHypopneaDetections) > 0:
            temp = apneasAndHypopneaDetections[0]
            intervalsAreTimeStamps = not isinstance(temp, Iterable)

            maxApneaTime = np.max(np.max(np.array(apneasAndHypopneaDetections)))
            apneaIndicatorMaxTime = maxApneaTime + Params.maxApneaTimeMargin
            apneaIndicatorTimeAxis = np.linspace(
                0,
                apneaIndicatorMaxTime,
                int(np.ceil(Params.apneaIndicatorFs * apneaIndicatorMaxTime)),
                endpoint=False,
            )

            apneaIndicator = algoUtils.computeCenterIndicatorSignalFromIntervals(
                apneasAndHypopneaDetections,
                len(apneaIndicatorTimeAxis),
                apneaIndicatorTimeAxis,
                intervalsAreTimeStamps=intervalsAreTimeStamps,
            )

            weightedApneaIndicator = np.ones(apneaIndicator.shape)
            apneaClusterFilter = algoUtils.prepareMovingAverageFilter(
                Params.apneaClusterFactorTimeBasis,
                1 / Params.apneaIndicatorFs,
                normalizeFlag=False,
            )

            apneaClusterIndicator = sigAPI.convolve(
                apneaIndicator,
                apneaClusterFilter,
                mode="same",
            )

            minApneaWeightOutsideClusters = Params.minApneaWeightOutsideClusters
            clusterWeightedApneaIndicator = (
                ((1 - minApneaWeightOutsideClusters) * apneaClusterIndicator)
                / Params.minNoOfApneasInCluster
                * apneaIndicator
            ) + minApneaWeightOutsideClusters
            clusterWeightedApneaIndicator[clusterWeightedApneaIndicator <= minApneaWeightOutsideClusters] = 0
            weightedApneaIndicator = weightedApneaIndicator * clusterWeightedApneaIndicator
            del clusterWeightedApneaIndicator

            del apneaIndicator, apneaIndicatorTimeAxis

            apneaCount = np.sum(weightedApneaIndicator)
            del weightedApneaIndicator

        else:
            apneaCount = 0

        if totalTime == 0:
            self.AHI = np.nan
        else:
            self.AHI = apneaCount / (totalTime / 60 / 60)

    def totalSnoringLength(
        self,
        currentSnoringEventStartEnds,
        snoringStrength,
        validPeriods,
        totalMinutes,
    ) -> int:
        """Find total snoring length and snoring seconds

        :param currentSnoringEventStartEnds: Snoring events detected in seconds
        :type currentSnoringEventStartEnds: tuple array
        :param snoringStrength: Strenth of the snoring in the intervals
        :type snoringStrength: float array
        :param validPeriods: Valid periods
        :type validPeriods: tuple array
        :return: total SnoringTimeSeconds,
            snoringIntervalSecond,
            snoringIntervalMinutes,
            snoringIntervalStrength,
        :rtype: int, tuple array, tuple array, float array
        """
        if len(currentSnoringEventStartEnds) > 0 and len(validPeriods) > 0:
            startSec = np.array(list(zip(*currentSnoringEventStartEnds))[0])
            startMin = (startSec / 60).astype(int)
            snoreMinutes, numberOfSnores = np.unique(startMin, return_counts=True)
            avgSnoringPerMinute = [
                sum(
                    snoringStrength[sum(numberOfSnores[:i]) : sum(numberOfSnores[: i + 1])],
                )
                / num
                for i, num in enumerate(numberOfSnores)
            ]
            validMinutes = np.concatenate(
                [np.arange(int(i[0] / 60), int(i[1] / 60)) for i in validPeriods],
            )
            snoreIntervals = algoUtils.ranges(snoreMinutes)
            snoringIntervalSeconds = []
            snoringIntervalMinutes = []
            snoringIntervalStrength = []
            for _, (start, end) in enumerate(snoreIntervals):
                if start != end:
                    snoringIntervalSeconds = np.concatenate(
                        [snoringIntervalSeconds, np.arange((start - 1) * 60, end * 60)],
                    )
                    snoringIntervalMinutes = np.concatenate(
                        [snoringIntervalMinutes, np.arange(start, end)],
                    )
                    snoringIntervalStrength = np.concatenate(
                        [
                            snoringIntervalStrength,
                            avgSnoringPerMinute[list(snoreMinutes).index(start) : list(snoreMinutes).index(end)],
                        ],
                    )
                elif start == end and numberOfSnores[list(snoreMinutes).index(start)] > 1:
                    snoringIntervalSeconds = np.concatenate(
                        [snoringIntervalSeconds, np.arange((start - 1) * 60, end * 60)],
                    )
                    snoringIntervalMinutes = np.concatenate(
                        [snoringIntervalMinutes, [start]],
                    )
                    snoringIntervalStrength = np.concatenate(
                        [
                            snoringIntervalStrength,
                            [avgSnoringPerMinute[list(snoreMinutes).index(end)]],
                        ],
                    )
                else:
                    pass
            # TODO: (start, end) instead of i
            validSnoringSeconds = np.concatenate(
                [np.arange(int(i[0]), int(i[1])) for i in validPeriods],
            )
            self.totalSnoringTimeSeconds = len(
                np.intersect1d(snoringIntervalSeconds, validSnoringSeconds),
            )
            snoringIntervalSecond = algoUtils.ranges(validSnoringSeconds)
        else:
            self.totalSnoringTimeSeconds = 0
            snoringIntervalSecond = []
            snoringIntervalMinutes = []
            snoringIntervalStrength = []

        snoringPlot = []
        for i in range(totalMinutes):
            if i not in snoringIntervalMinutes or i not in validMinutes:
                snoringPlot.append(0)
            else:
                snoringPlot.append(
                    int(snoringIntervalStrength[list(snoringIntervalMinutes).index(i)]),
                )

        return (
            max(0.0, self.totalSnoringTimeSeconds),
            snoringIntervalSecond,
            snoringIntervalMinutes,
            snoringPlot,
        )

    # TODO: Please use snake case for the method name to match the format of the rest of the project.
    def checkAndAdaptStreamingParametersForAudioFile(
        self,
        pathOfFileToTest: Union[str, int, Path],
    ):
        """
        Function used to verify/fix standard snorefox audio streaming (start offset etc)
        parameters for a specific audio file

        :param pathOfFileToTest: path of the audio file to read
        :type pathOfFileToTest: str, int, or Path object
        :return:
            offsetToStartAt: checked (and potentially re-corrected) offset (in s) at which the audio streaming starts
            newAudioDurationInSeconds: checked (and potentially re-corrected) total reading duration parameter (in s)
                for determining how long to stream
            timeTrimmedOutAtTheEnd: amount of time (in s) to be trimmed at the end of the streamed audio file
        :rtype:
            offsetToStartAt: float
            newAudioDurationInSeconds: float
            timeTrimmedOutAtTheEnd: float
        """

        originalAudioDurationInSeconds = librosa.get_duration(filename=pathOfFileToTest)

        if originalAudioDurationInSeconds >= Params.minUsableAudioDuration:
            offsetToStartAt = Params.enforcedSilencePeriodLengthInSeconds
            newAudioDurationInSeconds = originalAudioDurationInSeconds - offsetToStartAt
            timeTrimmedOutAtTheEnd = Params.enforcedSilencePeriodLengthInSeconds
        else:
            offsetToStartAt = 0
            newAudioDurationInSeconds = originalAudioDurationInSeconds
            timeTrimmedOutAtTheEnd = 0
            self._logger.warning(
                "Configuring audio streaming time parameters to disable trimming due to audio being too short",
            )

        return offsetToStartAt, newAudioDurationInSeconds, timeTrimmedOutAtTheEnd

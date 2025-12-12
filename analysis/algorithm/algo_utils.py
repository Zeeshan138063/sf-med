import gc
import logging
import operator
import warnings
from typing import Iterable, Union

import librosa
import numpy as np
import scipy.interpolate
import scipy.signal as sigAPI

import snorefox_med.analysis.algorithm.parameters as Params

logger = logging.getLogger(__name__)


def compute_STFT(
    audio_array,
    # TODO: I would import this variable directly from the settings file to
    #  conform with how we handle the config in other parts of the project.
    sample_rate: int = Params.SAMPLE_RATE_STFT,
    n_fft: int = Params.n_fft,
    hop_length: int = Params.spectrum_hop_length,
    reAdaptSTFTParamsFlag: bool = True,
    computeSTFTPhaseFlag: bool = True,
):
    """
    Function used to compute short-time fourier transform (STFT)
    magnitudes (& optionally phases), from an audio signal

    also handles some overhead / generalizations and preparations / side tasks around the STFT computation, and also
    performs the computation by audio block to save memory while handling boundary effects when compu

    :param audio_array: input audio or other 1D signal
    :type audio_array: list or 1D numpy array (preferably of floats or ints), or generator (stream) returning those
    :param sample_rate: sample rate/frequency in Hz, defaults to Params.SAMPLE_RATE_STFT
    :type sample_rate: int, optional
    :param n_fft: frame/window size in samples at the reference sample rate (Params.SAMPLE_RATE_STFT).
        Defaults to Params.n_fft
    :type n_fft: int, optional
    :param hop_length: number of samples (step) by which to move each window, taken at the reference
        sample rate (Params.SAMPLE_RATE_STFT)), defaults to Params.spectrum_hop_length
    :type hop_length: int, optional
    :param reAdaptSTFTParamsFlag: whether to translate the parameters for the working sample rate
        (!YOU SHOULD IN ALMOST ALL CASES!), defaults to True
    :type reAdaptSTFTParamsFlag: bool, optional
    :param computeSTFTPhaseFlag: True if want to compute phases, False otherwise, defaults to True
    :type computeSTFTPhaseFlag: bool, optional
    :return:
        STFT_magnitude: short time fourier transform coefficients magnitudes (how many depends on chosen number),
                      for each window / time frame in the audio data (dimensions: frequency x time)
        STFT_phase: same as STFT_magnitude but with phases
        time_axis: time instants for the windows / time frame on which the coefficients are computed,
                   these correspond to the temporal dimenson of the STFT ouptus
        frequency_axis: frequencies for which the coefficients are computed,
                   these correspond to the frequency dimenson of the STFT ouptut
    :rtype:
        STFT_magnitude: numpy array of floats
        STFT_phase: numpy array of floats
        time_axis: numpy array of floats
        frequency_axis: numpy array of floats
    """
    isArray = isinstance(audio_array, np.ndarray) or isinstance(audio_array, list)
    isLibrosaGenerator = callable(audio_array)

    if isLibrosaGenerator:
        # calling the generator function to return a generator
        audio_array, sample_rate, audioBlockDuration, totalAudioDuration = audio_array()

        audio_array_length = int(np.ceil(totalAudioDuration * sample_rate))
    elif isArray:
        audio_array_length = len(audio_array)

    if reAdaptSTFTParamsFlag:
        logger.info("Re-adapting STFT parameters")
        # used to map the STFT parameters from those used at 16 kHz
        n_fft_readapted = int(np.ceil(n_fft * sample_rate / Params.SAMPLE_RATE_STFT))
        hop_length_readapted = int(
            np.ceil(hop_length * sample_rate / Params.SAMPLE_RATE_STFT),
        )
    else:
        n_fft_readapted, hop_length_readapted = (n_fft, hop_length)

    if isLibrosaGenerator:
        offset_n_fft = n_fft_readapted
    elif isArray:
        offset_n_fft = None

    STFT_time_axis = compute_spectrum_time_axis(
        audio_array_length,
        hop_length_readapted,
        sample_rate,
        n_fft=offset_n_fft,
    )

    frequency_axis = compute_STFT_freq_axis(
        sample_rate=sample_rate,
        n_fft=n_fft_readapted,
    )

    if isArray:
        logger.info("Computing STFT (one fell-swoop)")
        STFT = librosa.stft(
            audio_array,
            n_fft=n_fft_readapted,
            hop_length=hop_length_readapted,
        )
        logger.info("Finished computing STFT")
    elif isLibrosaGenerator:
        logger.info("Computing STFT (by blocks)")
        audioBlockSizeInCurrentNffts = int(
            np.floor(audioBlockDuration / (n_fft_readapted / sample_rate)),
        )
        if computeSTFTPhaseFlag:
            STFT = np.zeros(
                (len(frequency_axis) + 1, len(STFT_time_axis)),
                dtype=np.csingle,
            )
        else:
            # if not computing the phase, then no need to store / allocate double the memory for
            # complex values -- half the memory is only needed
            STFT = np.zeros(
                (len(frequency_axis) + 1, len(STFT_time_axis)),
                dtype=np.single,
            )

        for n, audio_block in enumerate(audio_array):
            audio_block_start_time = n * audioBlockDuration
            audio_block_stop_time = (n + 1) * audioBlockDuration

            audio_block_start_bin_in_STFT = (n) * audioBlockSizeInCurrentNffts
            audio_block_stop_bin_in_STFT = min(
                (n + 1) * audioBlockSizeInCurrentNffts,
                len(STFT_time_axis) - 1,
            )

            if audio_block_start_time <= totalAudioDuration:
                current_STFT_block = librosa.stft(
                    audio_block,
                    n_fft=n_fft_readapted,
                    hop_length=hop_length_readapted,
                    center=False,
                )

                if not computeSTFTPhaseFlag:
                    # overwriting the STFT block itself in this case to save memory
                    current_STFT_block = abs(current_STFT_block)

                if audio_block_stop_time > totalAudioDuration:
                    # for the last block there is an mismatch with the all-file STFT due to unavoidable numerical
                    # imprecisions and quantization errors, so making sure it fits into the STFT, by discarding the
                    # last spurious frames in the all-file STFT or the current STFT block
                    noOfSpuriousGlobalSTFTbins = (
                        audio_block_stop_bin_in_STFT - audio_block_start_bin_in_STFT - current_STFT_block.shape[1]
                    )

                    if noOfSpuriousGlobalSTFTbins >= 0:
                        current_STFT_block = np.concatenate(
                            [
                                current_STFT_block,
                                np.zeros(
                                    (
                                        len(frequency_axis) + 1,
                                        noOfSpuriousGlobalSTFTbins,
                                    ),
                                ),
                            ],
                            axis=1,
                        )
                    else:
                        current_STFT_block = current_STFT_block[
                            :,
                            :noOfSpuriousGlobalSTFTbins,
                        ]

                else:
                    noOfSpuriousGlobalSTFTbins = 0

                STFT[
                    :,
                    audio_block_start_bin_in_STFT:audio_block_stop_bin_in_STFT,
                ] = current_STFT_block
                del current_STFT_block
                gc.collect()

            else:
                noOfSpuriousGlobalSTFTbins = 0

            del audio_block
            gc.collect()
            n += 1

        if noOfSpuriousGlobalSTFTbins > 0:
            STFT = STFT[:, :-noOfSpuriousGlobalSTFTbins]

        logger.info("Finished computing STFT")

    # enforcing this to avoid bugs -- sometimes there's a difference of 1 meta-sample/bin-shift
    # due to numerical differences betwen the predicted time axis and the actual STFT frame-shifts
    STFT_time_axis = STFT_time_axis[0 : STFT.shape[1]]

    logger.info("Splitting STFT into magnitude (and phase)")
    if computeSTFTPhaseFlag:
        STFT_magnitude = abs(STFT)
        STFT_magnitude = STFT_magnitude[1:, :]

        STFT_phase = np.angle(STFT)
        del STFT
        gc.collect()
    else:
        # overwriting the STFT itself in this case to save memory
        if isArray:
            # should not do this again if already done when block-reading, otherwise
            # it would cause an array copy

            STFT = abs(STFT)
        STFT = STFT[1:, :]
        STFT_phase = None
        STFT_magnitude = STFT

    logger.info("Finished splitting STFT")

    return STFT_magnitude, STFT_phase, STFT_time_axis, frequency_axis


def mag2db(
    signal: Union[
        Iterable[float],
        Iterable[int],
        Iterable[Union[Iterable[float], Iterable[int]]],
    ],
    threshold: float = Params.dB_standard_cutoff_threshold,
    renormalizeFlag: bool = True,
):
    """Function used to conver a signal from linear-scale to decibel scale, while handling the need thresholding

    CAREFUL: this operates on the signal in-place in a destructive manner

    :param signal: (MAGNITUDE not POWER) signal in linear scale to map to decibel scale.
        CAREFUL: the function does not take the magnitude of signals (for memory usage reasons),
        only supply signals which are already in magnitude form / positive
    :type signal: list, numpy array or similar (of floats or ints) (including 2-D arrays)
    :param threshold: threshold in decibels to use for limiting dynamic range.
        Defaults to Params.dB_standard_cutoff_threshold
    :type threshold: float, optional
    :param renormalizeFlag: whether to normalize the data before conversion, defaults to True
    :type renormalizeFlag: bool, optional
    :return: signal: signal converted to decibel scale
    :rtype: numpy array (of floats)
    """
    logger.info("Transforming signal to decibel scale")

    if isinstance(signal, np.ndarray):
        nonSingleTonSignal = signal.size > 1
    else:
        signalSize = len(signal)
        if isinstance(signal[0], Iterable):
            signalSize = signalSize * len(signal[0])

        nonSingleTonSignal = signalSize > 1

    if renormalizeFlag and nonSingleTonSignal:
        logger.info("Normalizing signal wrt max")
        signal = signal / np.max(
            np.abs(signal).reshape(signal.size),
        )  # normalizing wrt absolute maximum

    signalMaximumLinear = np.max(
        np.max(signal),
    )  # will simply be one if using the re-normalization flag
    np.seterr(divide="ignore")  # TODO: This ignores all division by zero errors? Why?
    signalMaximumDecibel = 20 * np.log10(signalMaximumLinear)
    # TODO: I think a more elegant solution would be to read the current error level with np.geterr
    #  before setting divide="ignore" and then applying it here again.
    #  Also, we should add comments explaining why the zero division errors should be ignored.
    np.seterr(divide="warn")
    logger.info("Clipping any values below insignificant levels (to not allow zero)")
    # replacing numbers near zero with very low float
    lowPrecisionCutoff = (10**-20) * signalMaximumLinear
    if nonSingleTonSignal:
        signal[signal <= lowPrecisionCutoff] = lowPrecisionCutoff
    else:
        signal = max(signal, lowPrecisionCutoff)

    logger.info("Taking the whole to decibel scale")
    # this is explicitly written using a for loop and NOT using whole-array operations
    # to enforce that the operation occur in-place and NOT spuriously copy the signal
    np.seterr(divide="ignore")  # TODO: Same as above.
    signal = 20 * np.log10(signal)
    np.seterr(divide="warn")  # TODO: Same as above.

    logger.info("Clipping the values ourside the chosen dynamic range")
    # readapting then threshold to falling 50 (or other whatever other) dB level below the maximum dB level
    if nonSingleTonSignal:
        signal[signal <= signalMaximumDecibel + threshold] = signalMaximumDecibel + threshold
    else:
        signal = max(signal, signalMaximumDecibel + threshold)

    return signal


def compute_spectrum_time_axis(
    signal_length: int,
    hop_length: int = Params.spectrum_hop_length,
    sample_rate: int = Params.SAMPLE_RATE_STFT,
    n_fft: int = None,
):
    """Function used to compute time instants for the windows / time frames of a spectrogram (STFT, MFCC etc)

    computes the center-time (center of the buffer/chunk/window time, in seconds) for
    the time bins for a spectrogram or MEL-cepstrum etc, when librosa does its center-padding,
    otherwise when there's no center-padding (n_fft=None) it corrects for the offset and computes
    the start-time of the bins instead

    :param signal_length: length in samples of the signal used to compute the spectrogram
    :type signal_length: int
    :param hop_length: number of samples (step) by which to move each window, taken at the provided
        sample rate, defaults to Params.spectrum_hop_length
    :type hop_length: int, optional
    :param sample_rate: sample rate/frequency in Hz, defaults to Params.SAMPLE_RATE_STFT
    :type sample_rate: int, optional
    :param n_fft: frame/window size in samples at the provided sample rate, defaults to Params.n_fft
    :type n_fft: int, optional
    :return: time_axis: time axis with the center (or start) times of the time frame bins
    :rtype: numpy array (of floats)
    """

    logger.info("Computing STFT/spectrum time axis")
    time_axis = np.arange(0, signal_length + 1, hop_length)
    time_axis = time_axis / sample_rate

    if n_fft is not None:
        time_axis = time_axis + (n_fft / sample_rate) / 2
        time_axis = time_axis[:-1]

    return time_axis


def compute_STFT_freq_axis(
    sample_rate: int = Params.SAMPLE_RATE_STFT,
    n_fft: int = Params.n_fft,
    removeDCFlag: bool = True,
):
    """Function used to compute the frequencies for which STFT coefficients were computed

    :param sample_rate: sample rate/frequency in Hz, defaults to Params.SAMPLE_RATE_STFT
    :type sample_rate: int, optional
    :param n_fft: frame/window size in samples at the provided sample rate, defaults to Params.n_fft
    :type n_fft: int, optional
    :param removeDCFlag: True if the spectrogram had its DC removed, False otherwise, defaults to True
    :type removeDCFlag: bool, optional
    :return: frequency_axis: frequency axis with the individual frequencies
    :rtype: numpy array of floats
    """

    logger.info("Computing STFT/spectrum frequency axis")
    frequency_axis = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    if removeDCFlag:
        frequency_axis = frequency_axis[1:]

    return frequency_axis


def addDominantFrequencyIndexToSelection(
    fineSTFTFrame,
    absolutePeriodicityThreshold: float,
    currentDominantFrequencySelection,
    minDominantFreqeucnyPeakDistance: int,
    frequencyIndexToAdd: int,
):
    """
    Function used to add a selected dominant-frequency bin to an
    existing selection of dominant-frequency bins in an STFT

    the function takes care of some sanity and other checks while adding the bin. Used when doing
    frequency selection on an STFT per time frame

    :param fineSTFTFrame: frame/window of STFT magnitude coefficients on which the selected frequency was selected
    :type fineSTFTFrame: 1D numpy array
    :param absolutePeriodicityThreshold: threshold (in STFT magnitude units) used to "select" a frequency as "domiant"
    :type absolutePeriodicityThreshold: float
    :param currentDominantFrequencySelection: current selection of dominant frequencies to add the new selection to
    :type currentDominantFrequencySelection: int or (empty) 1D numpy array
    :param minDominantFreqeucnyPeakDistance: minimum number of bins to enforce between the new selection
        and existing selections
    :type minDominantFreqeucnyPeakDistance: int
    :param frequencyIndexToAdd: new selected frequency bin index to add to the existing selection
    :type frequencyIndexToAdd: int
    :return: updated dominant frequency selection
    :rtype: numpy array (of floats)
    """
    if fineSTFTFrame[frequencyIndexToAdd] > absolutePeriodicityThreshold:
        if len(currentDominantFrequencySelection) > 0:
            if min(abs(currentDominantFrequencySelection - (frequencyIndexToAdd))) >= minDominantFreqeucnyPeakDistance:
                addIndexFlag = True
            else:
                addIndexFlag = False
        else:
            addIndexFlag = True
    else:
        addIndexFlag = False

    if addIndexFlag:
        currentDominantFrequencySelection = np.array(
            list(
                set(currentDominantFrequencySelection).union(set([frequencyIndexToAdd])),
            ),
        )

    return currentDominantFrequencySelection


def convertIndicatorToBoolean(indicatorSignal, lookForOnOrOff="on"):
    """Function used to convert indicator signal to a boolean with "True" and "False" for on/off regions (or vise versa)

    :param indicatorSignal: a signal consisting of two values: 0 (off) or 1 (on) (with numerical inaccuracies allowed)
    :type indicatorSignal: list, 1D numpy array or similar (of floats or ints or bools)
    :param lookForOnOrOff: whether to set the parts that are "on" to True, or those that are off.
        Values possible are "on" or "off".
        Defaults to those that are "on" ("on")
    :type lookForOnOrOff: _type_, optional
    :return: indicator signal transformed to boolean scale
    :rtype: 1D numpy array (of bools)
    """
    if not isinstance(indicatorSignal, np.ndarray):
        indicatorSignal = np.array(indicatorSignal)

    if lookForOnOrOff == "on":
        indicatorSignal = indicatorSignal >= 1 - Params.indicatorNoiseTolerance
    elif lookForOnOrOff == "off":
        indicatorSignal = indicatorSignal < 1 - Params.indicatorNoiseTolerance

    indicatorSignal = np.array(indicatorSignal)

    return indicatorSignal


def computeIndicatorOnsetsAndEndIndices(indicatorSignal):
    """Function used to compute the start/end indices of the 1/on regions in an indicator signal

    :param indicatorSignal: a signal consisting of two values: 0 (off) or 1 (on) (with numerical inaccuracies allowed)
    :type indicatorSignal: list, 1D numpy array or similar (of floats, ints or bools)
    :return: start and end indices of 1/on regions
    :rtype: list of pair-tuples (of ints)
    """
    logger.info("Computing indicator onset indices")
    indicatorEges = sigAPI.convolve(indicatorSignal, [[1], [-1]], mode="same")

    if indicatorSignal[0] > 1 - Params.indicatorNoiseTolerance or indicatorSignal[0] is True:
        indicatorEges[0] = 1
    else:
        indicatorEges[0] = 0

    if indicatorSignal[-1] > 1 - Params.indicatorNoiseTolerance or indicatorSignal[-1] is True:
        indicatorEges[-1] = -1
    else:
        indicatorEges[-1] = 0

    onsetsAndEnds = [
        (n, k - 1)
        for (n, k) in zip(
            (indicatorEges.squeeze() > Params.indicatorNoiseTolerance).nonzero()[0],
            (indicatorEges.squeeze() < -Params.indicatorNoiseTolerance).nonzero()[0],
        )
    ]

    return onsetsAndEnds


def computeCenterIndicatorSignalFromIntervals(
    intervals,
    indicatorSignalLength: int,
    timeAxis=None,
    intervalTimeCap: Union[int, float] = np.inf,
    intervalsAreTimeStamps: bool = False,
    intervalIndicatorWeights=None,
):
    """Function used to create a 1D indicator signal with "on" periods given by select period, "off" elsewhere

    :param intervals: the intervals on which the indicator signal should be "on" in time or samples, or alternatively
        timestmaps (e.g. interval mid-times)
    :type intervals: list of 2-element lists of pair-tuples (preferably of floats, or ints), or list of floats or ints
    :param indicatorSignalLength: indicator signal size (check your input's sampling rate for this)
    :type indicatorSignalLength: int
    :param timeAxis: the time axis for the indicator. Supply (same unit) if giving intervals in time not samples.
        Defaults to None
    :type timeAxis: 1D numpy array (preferably of floats, or of ints), optional
    :param intervalTimeCap: a maximum time cap to impose on the intervlas -- shorter ones are discarded --.
        Defaults to np.inf
    :type intervalTimeCap: int or float (same unit and type as intervals), optional
    :param intervalsAreTimeStamps: whether the supplied intervals are actually timestamps / mid-times of intervals.
        True if yes, False if not. Defaults to False
    :type intervalsAreTimeStamps: bool, optional
    :return: indicator signal with the regions falling under the given intervals set to "on", "off" elsewhere
    :rtype: 1D numpy array (of ints)
    """
    if not intervalsAreTimeStamps:
        intervalCenters = [
            ((i[1] + i[0]) / 2) for i in intervals if computeTotalIntervalDurations([i]) < intervalTimeCap
        ]
    else:
        intervalCenters = intervals

    if timeAxis is None:
        intervalCenters = [int(np.round(i)) for i in intervalCenters]
    elif len(intervalCenters) > 0:
        intervalCenters = timeToIndex(timeAxis, intervalCenters)
        if isinstance(intervalCenters, Iterable):
            intervalCenters = list(intervalCenters)
        else:
            intervalCenters = [intervalCenters]

        # in case some interval centers are outside the time axis, discard those
        intervalCenters = [i for i in intervalCenters if not np.isnan(i)]

    indicatorSignal = np.zeros((indicatorSignalLength, 1))
    if intervalIndicatorWeights is None:
        indicatorSignal[intervalCenters] = 1
    else:
        for n, i in enumerate(intervalCenters):
            indicatorSignal[i] = intervalIndicatorWeights[n]

    return indicatorSignal


def expandTooShortActiveRegionsInIndicator(
    indicatorSignal,
    minNoOfActiveSamplesToEnforce: int,
):
    """Funciton used to extend "on" regions in indicator signal when they're too small/short

    :param indicatorSignal: indicator signal with "on" and "off" regions. If other (slightly) different signal
        signal is provided it's interpreted as an indicator signal -- 1D signals will have values less than approx. 1
        take as "off" and everything else taken as "on". This function is used because after thresholding
        previously-processed indicator signal, it can happen that the retained regions are smaller than the actual
        filter having produced them (e.g. an STFT filter, a majority filter, an energy filter etc), they should retain
        that filter length at least, otherwise the processing doesn't make sense
    :type indicatorSignal: 1D numpy array (preferably of floats, ints or bools)
    :param minNoOfActiveSamplesToEnforce: minimum number of samples on the indicator an "on" region is supposed to last
    :type minNoOfActiveSamplesToEnforce: int
    :return: modified indicator signal
    :rtype: 1D numpy array (of ints)
    """

    activePeriodsStartEndIndices = computeIndicatorOnsetsAndEndIndices(indicatorSignal)

    # finding the contiguous regions whose length is less than the minimum valid region width
    shortActivePeriodsIndicator = computeCenterIndicatorSignalFromIntervals(
        activePeriodsStartEndIndices,
        indicatorSignal.size,
        intervalTimeCap=minNoOfActiveSamplesToEnforce,
    )

    # expanding the shorter-than-minimum-length regions into at least the minimum length
    expandedShortActivePeriodsIndicator = sigAPI.convolve(
        shortActivePeriodsIndicator,
        np.ones((minNoOfActiveSamplesToEnforce, 1)),
        mode="same",
    )
    indicatorSignal = expandedShortActivePeriodsIndicator + indicatorSignal

    indicatorSignal[indicatorSignal > 1] = 1

    return indicatorSignal


def computeStartAndEndTimesFromIndicator(indicatorSignal, timeAxis=None):
    """Function used to to compute the start/end TIMES (or indices) of the 1/on regions in an indicator signal

    this is a wrapper around computeIndicatorOnsetsAndEndIndices that handles some extra overhead / prep. One typical
    use of such a function is to compute the start/end times of valid regions in an onset signal

    :param indicatorSignal: a signal consisting of two values: 0 (off) or 1 (on) (with numerical inaccuracies allowed)
    :type indicatorSignal: list, 1D numpy array or similar (of floats, ints or bools)
    :param timeAxis: time axis in seconds corresponding to the indicator signal, leave as None if you just
        want the indices but not the times corresponding to the indices, otherwise provide it. Defaults to None
    :type timeAxis: list or numpy array, optional
    :return: start and end indices of 1/on regions
    :rtype: list of pair-tuples (of floats or ints)
    """
    logger.info(
        "Preparing indicator signal for onset start/end (time or index) detecton",
    )
    if type(indicatorSignal) is list:
        indicatorSignal = np.array(indicatorSignal)

    noOfIndicatorDimensions = len(indicatorSignal.shape)

    if noOfIndicatorDimensions == 1:
        indicatorSignal = np.expand_dims(
            indicatorSignal.reshape(indicatorSignal.size),
            axis=1,
        )

    # a period ending at a certain bin should mean the period ends at the END of that bin, hence the k+1 below
    startEnds = [
        (max(n, 0), min(k + 1, len(indicatorSignal) - 1))
        for (n, k) in computeIndicatorOnsetsAndEndIndices(indicatorSignal)
    ]

    if timeAxis is not None:
        startEnds = [(timeAxis[i[0]], timeAxis[i[1]]) for i in startEnds]

    return startEnds


def siftIntervalsForThoseOverlappingOtherInterval(intervalsToSift, otherInterval):
    """
    Function used to check which sub-intervals of a list of intervals
    overlap any interval in another list of intervals

    :param intervalsToSift: intervals to sift
    :type intervalsToSift: list of intervals (sub-lists of 2 elements or pair-tuples) of floats or ints
    :param otherInterval: the other interval with which all the returned intervals need to overlap
    :type otherInterval: list of 2 elements or pair-tuple of floats or ints
    :return: original intervals sifted to remove sub-intervals not overlapping the provided other interval
    :rtype: list of pair-tuples (of floats or ints depending on input)
    """
    # there four types of periods overlapping the neighborhood:
    # starting before it and ending in it
    overlapingIntervalsWithinOtherInterval = [
        (otherInterval[0], firstInterval[1])
        for firstInterval in intervalsToSift
        if firstInterval[0] < otherInterval[0]
        and firstInterval[1] >= otherInterval[0]
        and firstInterval[1] <= otherInterval[1]
    ]
    # starting in it and ending in it
    overlapingIntervalsWithinOtherInterval = overlapingIntervalsWithinOtherInterval + [
        firstInterval
        for firstInterval in intervalsToSift
        if firstInterval[0] >= otherInterval[0] and firstInterval[1] <= otherInterval[1]
    ]
    # starting in it and ending atfer it
    overlapingIntervalsWithinOtherInterval = overlapingIntervalsWithinOtherInterval + [
        (firstInterval[0], otherInterval[1])
        for firstInterval in intervalsToSift
        if firstInterval[0] >= otherInterval[0]
        and firstInterval[0] <= otherInterval[1]
        and firstInterval[1] > otherInterval[1]
    ]
    # starting before it and ending after it
    overlapingIntervalsWithinOtherInterval = overlapingIntervalsWithinOtherInterval + [
        otherInterval
        for firstInterval in intervalsToSift
        if firstInterval[0] < otherInterval[0] and firstInterval[1] > otherInterval[1]
    ]

    return overlapingIntervalsWithinOtherInterval


def siftIntervalsForThoseOverlappingAnyOtherInterval(intervalsToSift, otherIntervals):
    """
    Function used to check which sub-intervals of a list of intervals overlap any interval in another list of intervals

    an extension of siftIntervalsForThoseOverlappingOtherInterval to keep sub-intervals overlapping with any of the
    other provided selection of intervals

        CAREFUL: ONLY DESIGNED FOR INTERVAL ARGUMENTS THAT ARE THEMSELVES NOT OVERLAPPING

    :param intervalsToSift: intervals to sift
    :type intervalsToSift: list of 2-element lists of pair-tuples of floats or ints
    :param otherIntervals: the other intervals of whom 1 at least the returned intervals need to overlap with
    :type otherIntervals: list of 2-element lists of pair-tuples of floats or ints
    :return: original intervals sifted to remove sub-intervals not overlapping any of the the provided other intervals
    :rtype: list of pair-tuples (of floats or ints depending on input)
    """
    overlapingIntervalsWithinOtherIntervals = []
    for otherInterval in otherIntervals:
        overlapingIntervalsWithinOtherIntervals = (
            overlapingIntervalsWithinOtherIntervals
            + siftIntervalsForThoseOverlappingOtherInterval(
                intervalsToSift,
                otherInterval,
            )
        )

    return overlapingIntervalsWithinOtherIntervals


def switchFlagsInIntervalOnValidityIndicator(
    validityIndicator,
    validityIndicatorTimeAxis,
    selectInterval,
    newSwitchedFlag: Union[float, int, bool],
):
    """Function used to change the values of an indicator signal for a specific time interval to a new chosen value

    :param validityIndicator: a validity signal consisting of two values: 0 (off) or 1 (on) or boolean values
    :type validityIndicator: list, 1D numpy array or similar (of floats, ints or bools)
    :param validityIndicatorTimeAxis: time axis in seconds corresponding to the validity indicator signal
    :type validityIndicatorTimeAxis: list or 1D numpy array (of floats, ints or bools)
    :param selectInterval: time interval (in seconds) on which to switch the indicator signal value
    :type selectInterval: pair-tuple or list of 2 elements
    :param newSwitchedFlag: new value to use within the selected interval
    :type newSwitchedFlag: float, int or bool
    :return: updated validityIndicator with the values corresponding to the selectInterval in time changed
    :rtype: list, numpy array or similar (of floats, ints or bools)
    """
    currentIntervalIndicesToSwitch = timeToIndex(
        validityIndicatorTimeAxis,
        selectInterval,
    )

    if not np.isnan(currentIntervalIndicesToSwitch[0]):
        validityIndicator[
            currentIntervalIndicesToSwitch[0] : int(
                np.nanmin(
                    [currentIntervalIndicesToSwitch[1] + 1, len(validityIndicator)],
                ),
            )
        ] = newSwitchedFlag

    return validityIndicator


def checkForAndRemoveHarmonicsFromFreqencySelection(
    frequencySelection,
    f0: Union[float, int],
    frequencyResolution: float,
    maxFreq: Union[float, int],
):
    """Function used to check for and remove inter-related harmonics in a set of pitch detections

    :param frequencySelection: pitch detections on which to check for harmonics
    :type frequencySelection: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param f0: fundamental / baseline frequency whose harmonics to check for
    :type f0: preferably float, or int
    :param frequencyResolution: minimal frequency resolution separating the pitch detections
    :type frequencyResolution: float
    :param maxFreq: max frequency up until which to compute the harmonics
    :type maxFreq: preferably float, or int
    :return: pitch detections refined for removing harmonics of baseline frequency
    :rtype: list (of floats, or of ints depending on input)
    """
    frequencySelection = list(frequencySelection)

    maxHarmonicPossibleInRange = np.floor(maxFreq / f0)
    f0Harmonics = np.arange(2, maxHarmonicPossibleInRange + 1) * f0
    for f1 in frequencySelection:
        if f1 != f0 and len(f0Harmonics) >= 1:
            if min(abs(f1 - f0Harmonics)) <= frequencyResolution:
                frequencySelection.remove(f1)

    return frequencySelection


def prepareMovingAverageFilter(
    filterLengthInSeconds: float,
    timeResolutionInSeconds: float,
    normalizeFlag: bool = True,
):
    """Funciton used to prepare a simple 1D moving-average filter

    :param filterLengthInSeconds: the length of the filter in seconds
    :type filterLengthInSeconds: float
    :param timeResolutionInSeconds: the time resolution (inverse/reciprocal of sampling rate/frequency) of the filter
    :type timeResolutionInSeconds: float
    :param normalizeFlag: whether to normalize the filter by its length. Leave to yes if you want a standard moving
        average filter, otherwise put to False if you want a counting not a moving average filter. Defaults to True
    :type bool
    :return: the moving average / counting filter
    :rtype: numpy array (of floats)
    """
    filterLengthInSamples = int(
        np.ceil(filterLengthInSeconds / timeResolutionInSeconds),
    )
    movingAverageFilter = np.ones((filterLengthInSamples, 1))

    if normalizeFlag:
        movingAverageFilter = movingAverageFilter / filterLengthInSamples

    return movingAverageFilter


def retainTimesWithinPredefinedWindows(basicTimesSet, predefinedWindows):
    """Function used to sift a set of times to keep only those falling within specific time intervals

    :param basicTimesSet: set of time instants to sift
    :type basicTimesSet: list, numpy array, set or similar (of floats or ints)
    :param predefinedWindows: intervals in which the sifted time instances need to fall
    :type predefinedWindows: list or set of pair-tuples or 2-element sub-lists
    :return:
        newTimesSet: sifted time instances
        retainedTimesIndicesInOriginalSet: indices of the time instances in the original set which they were retained
        belongingIndicesforEachTime: index of the interval(s) in which each of the
            sifted time instance in newTimesSet falls / is located
    :rtype:
        newTimesSet: list of floats
        retainedTimesIndicesInOriginalSet: list of integers
        belongingIndicesforEachTime: list of sub-lists of integers (or empty sub-list)
    """
    newTimesSet = []
    retainedTimesIndicesInOriginalSet = []
    belongingIndicesforEachTime = []

    for n, timeInSeconds in enumerate(basicTimesSet):
        currentBelongingIndices = [
            n for n, i in enumerate(predefinedWindows) if timeInSeconds >= i[0] and timeInSeconds <= i[1]
        ]
        timeBelongsToAnyValidRegion = len(currentBelongingIndices) > 0

        if timeBelongsToAnyValidRegion:
            newTimesSet.append(timeInSeconds)
            retainedTimesIndicesInOriginalSet.append(n)
            belongingIndicesforEachTime.append(currentBelongingIndices)

    return (
        np.array(newTimesSet),
        np.array(retainedTimesIndicesInOriginalSet),
        belongingIndicesforEachTime,
    )


def getResolutionFromAxis(axis):
    """Function used for inferring time resolution / sampling frequency from time or frequency axis

    :param axis: time axis or frequency axis corresponding to a time-signal
    :type axis: list or numpy array (preferably floats, or ints)
    :return: time resolution (reciprocal of sampling frequency/rate) or frequency resolution
    :rtype: float
    """
    return abs(axis[1] - axis[0])


def addValuetoAllTupleElements(
    tuplesToManipulate: list[
        tuple[
            Union[
                int,
                float,
            ]
        ]
    ],
    valuetoAdd: Union[int, float],
):
    """Function used to add a value to all tuple elements in a list of tuples

    :param tuplesToManipulate: list of tuples on which to add the given value
    :type tuplesToManipulate: list of pair-tuples (of floats or ints)
    :param valuetoAdd: the value to add to those tuples
    :type valuetoAdd: float or int
    :return: list of tuples with the value added to its tuples' elements
    :rtype: list of pair-tuples (of floats or ints depending on input)
    """

    logger.info("Adding a scalar to all values of working tuple")
    if not (type(tuplesToManipulate) is list or isinstance(tuplesToManipulate, np.ndarray)):
        tuplesToManipulate = [tuplesToManipulate]

    newTuples = [(i[0] + valuetoAdd, i[1] + valuetoAdd) for i in tuplesToManipulate]

    if not (type(tuplesToManipulate) is list or isinstance(tuplesToManipulate, np.ndarray)):
        newTuples = newTuples[0]

    return newTuples


def siftOutDetectionsCloseToOutliers(
    mainDetectionSets,
    outlierDetectionSet,
    minDistanceStrict: float,
    minDistanceLeniant: float,
    noOfLeniantOutlierstoAllow: int,
    causalityTolerance: float,
):
    """Function used to sift out from a set of detections those that are too close to a set of outliers

    :param mainDetectionSets: original set(s) of detections to sift
    :type mainDetectionSets: list or 1D numpy array of floats (or tuple thereof if multiple).
        Accepts either a single detection set as a list, or a slew of detection sets to sift
    :param outlierDetectionSet: set of outliers used to supress neighboring detections
    :type outlierDetectionSet: list or 1D numpy array of floats
    :param minDistanceStrict: min distance to enforce betwee outliers and detections (anything below discarded)
    :type minDistanceStrict: float
    :param minDistanceLeniant: min distance within which to check for enough outliers (if enough, detection discarded)
    :type minDistanceLeniant: float
    :param noOfLeniantOutlierstoAllow: no of outliers to allow within the lenient neighborhood
    :type noOfLeniantOutlierstoAllow: int
    :param causalityTolerance: tolerance (s) allowing outliers coming BEFORE a detection to non-causally supress latter
    :type causalityTolerance: float
    :return: originals set(s) modified by removing the detections with (enough) outliers nearby
    :rtype: list or numpy array of floats (or tuple thereof if multiple were given)
    """
    logger.info("Sifting out detections close to given outliers")

    # with the same parameters as a tuple of lists
    if type(mainDetectionSets) is not tuple:
        noOfSetsTosift = 1
        mainDetectionSets = [mainDetectionSets]
    else:
        noOfSetsTosift = len(mainDetectionSets)

    siftedDetectionSets = [[]] * noOfSetsTosift

    for currentSetNo in range(noOfSetsTosift):
        for detection in mainDetectionSets[currentSetNo]:
            causalOutLierDistances = detection - outlierDetectionSet
            causalOutLierDistances = causalOutLierDistances[causalOutLierDistances > -causalityTolerance]

            if len(causalOutLierDistances) >= 1:
                minCausalOutlierDistance = np.nanmin(causalOutLierDistances)
                outlierPeakInStrictNeighborhood = minCausalOutlierDistance < minDistanceStrict
                enoughOutlierPeaksInLeniantNeighborhood = (
                    len(
                        causalOutLierDistances[causalOutLierDistances < minDistanceLeniant],
                    )
                    > noOfLeniantOutlierstoAllow
                )
            else:
                outlierPeakInStrictNeighborhood = False
                enoughOutlierPeaksInLeniantNeighborhood = False

            if not outlierPeakInStrictNeighborhood and not enoughOutlierPeaksInLeniantNeighborhood:
                siftedDetectionSets[currentSetNo].append(detection)

    siftedDetectionSets = tuple(siftedDetectionSets)

    return siftedDetectionSets


def timeToIndex(
    timeAxis,
    specificTimes,
    onsetSignalTimeAxisMinimum=None,
    onsetSignalTimeAxisMaximum=None,
    uniformTimeAxis=False,
):
    """Funciton used to map back a time to an index corrsponding to it on a time axis

    :param timeAxis: time axis which on which the specific times were computed
    :type timeAxis: list or 1D numpy array of floats
    :param specificTimes: a single or multiple time instant(s) in seconds to translate back into an index/indices
        accepts either a single time as a scalar, or a slew of times as a tuple, list of 1D numpy array.
    :type specificTimes: float or tuple (of floats)
    :return: the index or indices corresponding to the specified time instants
    :rtype: float or tuple (of floats) depending on whether a single or slew of times were given in specificTimes
    """
    if isinstance(specificTimes, tuple) or isinstance(specificTimes, list) or isinstance(specificTimes, np.ndarray):
        noOfSpecificTimes = len(specificTimes)
    else:
        noOfSpecificTimes = 1
        specificTimes = [specificTimes]

    specificIndices = [np.nan] * noOfSpecificTimes

    if onsetSignalTimeAxisMinimum is None:
        minTime = np.min(timeAxis)
    else:
        minTime = onsetSignalTimeAxisMinimum

    if onsetSignalTimeAxisMaximum is None:
        maxTime = np.max(timeAxis)
    else:
        maxTime = onsetSignalTimeAxisMaximum

    for i in range(noOfSpecificTimes):
        if specificTimes[i] >= minTime and specificTimes[i] <= maxTime:
            if not uniformTimeAxis:
                specificIndex = np.argmin(np.abs(timeAxis - specificTimes[i]))
            else:
                timeAxisRes = getResolutionFromAxis(timeAxis)
                specificIndex = int(np.round(specificTimes[i] / timeAxisRes)) - int(
                    np.round(minTime / timeAxisRes),
                )
        else:
            specificIndex = np.nan

        specificIndices[i] = specificIndex

    if noOfSpecificTimes != 1:
        specificIndices = tuple(specificIndices)
    else:
        specificIndices = specificIndices[0]

    return specificIndices


def getMinOrMaxOfStartsEnds(startsEnds, funcToUse=min, timeAxis=None):
    """Function used to return either the min or the max time over a series of (start, end) pairs

    Usually used to retrieved the minimum time instant for all in a series of events, or the maximum time instant,
    while optionally translating them also from time instants to indices on a corresponding time axis

    :param startsEnds: set of events with start/end times (in seconds)
    :type startsEnds: list of 2-element lists of pair-tuples
    :param funcToUse: which mode of operation to use. Either sipply min as a function to detect starts, or max as a
        function to detect ends. Defaults to min
    :type funcToUse: function, optional
    :param timeAxis: time axis on which the supplied start and end times were computed. Supply if you want to also
        translate the start/end times in seconds to their corresponding indices on the time axis, or keep to None
        if this translation is not needed. Defaults to None
    :type timeAxis: list or 1D numpy array of floats, optional
    :return:
        minOrMaxTime: mininimum or maximum time instant across the events, depending on mode
        minOrMaxTimeIndex: indices corresponding to the time in in minOrMaxTime
    :rtype:
        minOrMaxTime:  float
        minOrMaxTimeIndex: int
    """
    minOrMaxTime = funcToUse([funcToUse(i) for i in startsEnds])
    if timeAxis is not None:
        minOrMaxTimeIndex = timeToIndex(timeAxis, minOrMaxTime)
    else:
        minOrMaxTimeIndex = None

    return minOrMaxTime, minOrMaxTimeIndex


def computeTotalIntervalDurations(intervalsToComputeOn):
    """Function used to compute the total/cumulative occupied by a set of non-overlapping intervals

    :param intervalsToComputeOn: a list of intervals or a single interval whose durations to sum
    :type intervalsToComputeOn: list of 2-element lists or pair-tuples or similar
    :return: total/cumulative duration of the intervals
    :rtype: float
    """
    if len(intervalsToComputeOn) > 0:
        if not (type(intervalsToComputeOn) is list or isinstance(intervalsToComputeOn, np.ndarray)):
            intervalsToComputeOn = [intervalsToComputeOn]

        intervalDurations = [(i[1] - i[0]) for i in intervalsToComputeOn]
        total = sum(intervalDurations)
    else:
        total = 0

    return total


def normalizeSound(
    y: Union[
        Iterable[float],
        Iterable[int],
        Iterable[Union[Iterable[float], Iterable[int]]],
    ],
):
    """Funciton for normalizing sound to avoid too-low dBFS sound levels / playback volumes

    CAREFUL: this destructively modifies the sound signal in-place

    :param y: input audio or similar signal
    :type y: list, numpy array or similar (including 2D array collections thereof)
        (preferably of floats, or at least of ints)
    :return: normalized audio signal
    :rtype: numpy array (potentially n-D just like y)
    """
    logger.info("Normalizing provided sound")
    return y / np.max(np.abs(y).reshape(y.size))


def locateSignificantPeriodicIntervalsInOtherInterval(
    contiguousPeriodicBinIntervals,
    otherInterval,
):
    """Function used to sift periodic-breathing intervals for those long-enough and overlapping another interval

    :param contiguousPeriodicBinIntervals: intervals to sift
    :type contiguousPeriodicBinIntervals: list of intervals (sub-lists of 2 elements or pair-tuples)
    :param otherInterval: the other interval with which all the returned intervals need to overlap
    :type otherInterval: 2-element list of pair-tuple
    :return: original intervals sifted according to the above
    :rtype: list of 2-element lists of pair-tuples (of floats)
    """
    contiguousPeriodicBinPeriodsWithinOtherInterval = siftIntervalsForThoseOverlappingOtherInterval(
        contiguousPeriodicBinIntervals,
        otherInterval,
    )

    # making sure to discard very short / insignicant adjacent periods
    if (
        computeTotalIntervalDurations(contiguousPeriodicBinPeriodsWithinOtherInterval)
        <= min(
            Params.noOfPreviousBreathstoEnforceBeforeApnea,
            Params.noOfBreathstoEnforceAfterApnea,
        )
        * Params.minBreathingDistanceInSeconds
    ):
        contiguousPeriodicBinPeriodsWithinOtherInterval = []

    return contiguousPeriodicBinPeriodsWithinOtherInterval


def retrievePitchesForTimeRegion(
    pitchTimeAxis: np.ndarray,
    pitchFrequencies: np.ndarray,
    timeRegion,
):
    """
    Function used to select pitches / dominant fundamental frequencies
    within a specific time interval from detections

    :param pitchTimeAxis: the time axis of the pitches, for each time there is a pitch detection (or nan)
    :type pitchTimeAxis: 1D numpy array (preferably of floats, or of ints)
    :param pitchFrequencies: the dominant fundamental frequencies / pitches for which
    :type pitchFrequencies: 1D numpy array (preferably of floats, or of ints)
    :param timeRegion: time interval (in same unit as time axis) for which to restrict / sift out relevant pitches
    :type timeRegion: 2-element list or pair-tuple or similar
    :return: pitches / dominant frequencies restricted to the chosen interval (sub-portion of input array)
    :rtype: 1D numpy array (of floats or of ints depending on input)
    """
    timeRegionIndices = timeToIndex(pitchTimeAxis, timeRegion)
    timeRegionPitches = pitchFrequencies[timeRegionIndices[0] : timeRegionIndices[1]]

    return timeRegionPitches


def contiguousPitchesHaveAbnormalFrequencyJump(
    contiguousRegionPitches,
    contiguousTimeRegion,
):
    """Function used to check whether pitch detections over a contiguous time period have abnormal jumps

    :param contiguousRegionPitches: pitch / dominant fundamental frequency detections over a contiguous time region
    :type contiguousRegionPitches: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param contiguousTimeRegion: time interval (in seconds) on which the pitches are given
    :type contiguousTimeRegion: 2-element lists or pair-tuples or similar
    :return: True if a too-big frequency/pitch jump is detected, False otherwise
    :rtype: bool
    """
    # computing the derivative of the pitch curves to detect jumps
    timeRegionPitchDerivatives = sigAPI.convolve(contiguousRegionPitches, [1, -1])

    # cancelling the first/last derivative as these are ill-defined
    timeRegionPitchDerivatives[0] = np.nan
    timeRegionPitchDerivatives[-1] = np.nan

    timeRegionDuration = computeTotalIntervalDurations([contiguousTimeRegion])

    maxPitchJump = np.nanmax(abs(timeRegionPitchDerivatives))

    if maxPitchJump / timeRegionDuration >= Params.maxAllowableFrequencyJumpStrict:
        return True
    else:
        return False


def detectValidBreathingRegionsInOnsetSignal(
    breathingOnsetSignal,
    breathingOnsetSignalTimeAxisInSeconds,
    useSilenceValidityFlag: bool = True,
    restrictBreathingFrequencyJumps: bool = True,
):
    """Function which analyzes a breathing onset signal for time intervals where there's a valid form of breathing

    A "valid" form of breathing is essentially periodic breathing, though the periodicity has a certain tolerance.
    Im addition, the function detects silent periods (where only thermal noise or so is audible). A tolerence for serial
    apneas is also built-in: in those intervals the breathing is less periodic adjacent to the the silent period (apnea)
    but these are smartly-treated and the regions are retained as valid

    :param breathingOnsetSignal: breathing onset bio-signal (or a sort of DC-corrected breathing flow signal)
    :type breathingOnsetSignal: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param breathingOnsetSignalTimeAxisInSeconds: time axis corresponding to the breathing onset signal
    :type breathingOnsetSignalTimeAxisInSeconds: list, numpy array or similar (of floats)
    :param useSilenceValidityFlag: True if special treatment / recognition of silent periods -- whether short
        (e.g. apneas ajdacent to periodic breathing) or long (e.g. un-audible breathing drowned in noise) -- is
        desired (recommended), or False otherwise. Defaults to True
    :type useSilenceValidityFlag: bool, optional
    :param restrictBreathingFrequencyJumps: True if want to enable pitch-frequency jump supression, False otherwise.
        Defaults to True
    :type: bool
    :return:
        validBreathingPeriodsStartEnds: time intervals for which the bio-signal can be considered valid
        invalidBreathingPeriodsStartEnds: time intervals for which the bio-signal is considered invalid.
            These represent the time-complement of the intervals in validBreathingPeriodsStartEnds
        silentPauses: time intervals where only low-volume noise is audible (e.g. silent breathing etc)
        fineFeqSTFTTimeAxis: the time instants of the time-frames for which a
            periodicity/validity analysis was performed
        dominantBreathingFrequencies: the dominant frequencies found at specific time instants.
            When none are found, zero is returns
    :rtype:
        validBreathingPeriodsStartEnds: list of 2-element lists of pair-tuples (of floats)
        invalidBreathingPeriodsStartEnds: list of 2-element lists of pair-tuples (of floats)
        silentPauses: list of 2-element lists of pair-tuples (of floats)
        fineFeqSTFTTimeAxis: numpy array of floats
        dominantBreathingFrequencies: numpy array of floats
    """

    logger.info(
        "Detecting valid (breathing-periodic or small silent) regions\
             in breathing onset signal using fine-frequency STFT",
    )

    onsetSignalSamplingFrequency = 1 / getResolutionFromAxis(
        breathingOnsetSignalTimeAxisInSeconds,
    )

    n_fft_fine_freq_resolution_onset_signal = int(
        np.ceil(onsetSignalSamplingFrequency / Params.minBreathingFrequencyResolution),
    )
    spectrum_hop_length__fine_freq_resolution_onset_signal = int(
        np.round(
            Params.periodicityAnalysisTimeResolution * onsetSignalSamplingFrequency,
        ),
    )

    # handling nan values due to the breathing onset signal not being computed for all
    # time bins (due to invalid filtering regions and start/end of recording)
    oldBreathingOnsetSignalLength = len(breathingOnsetSignal)
    breathingOnsetSignal = breathingOnsetSignal[
        ~np.isnan(breathingOnsetSignal)
    ]  # removing the nans (these are only at the start and end of the signal due to filtering)
    noOfStartingNaNs = int(
        (oldBreathingOnsetSignalLength - len(breathingOnsetSignal)) / 2,
    )
    noOfStartingNaNsInSeconds = noOfStartingNaNs / onsetSignalSamplingFrequency
    if noOfStartingNaNs > 0:
        breathingOnsetSignalTimeAxisInSeconds = breathingOnsetSignalTimeAxisInSeconds[
            noOfStartingNaNs:-noOfStartingNaNs
        ]

    # computing low-frequency STFT on original onset signal without the regions filled with NaNs
    fineFreqSTFT, _, fineFeqSTFTTimeAxis, fineFeqSTFTFreqAxis = compute_STFT(
        onset_normalization(breathingOnsetSignal),
        sample_rate=onsetSignalSamplingFrequency,
        n_fft=n_fft_fine_freq_resolution_onset_signal,
        hop_length=spectrum_hop_length__fine_freq_resolution_onset_signal,
        reAdaptSTFTParamsFlag=False,
        computeSTFTPhaseFlag=False,
    )

    fineFeqSTFTTimeAxis = (
        fineFeqSTFTTimeAxis + breathingOnsetSignalTimeAxisInSeconds[0]
    )  # re-aligning the fine-frequency STFT with the start of the non-NaN portion of the breathing onset signal
    dominantBreathingFrequencies = np.zeros(fineFeqSTFTTimeAxis.shape)

    # restricting the frequency rage used (in the fine-frequency STFT) to the breathing range
    fineFrequencyRangeOfInterest = [
        n
        for n, i in enumerate(fineFeqSTFTFreqAxis)
        if i <= Params.maxBreathingFrequency and i >= Params.minBreathingFrequency
    ]
    fineFeqSTFTFreqAxis = fineFeqSTFTFreqAxis[fineFrequencyRangeOfInterest]
    fineFreqSTFT = fineFreqSTFT[fineFrequencyRangeOfInterest, :]

    if useSilenceValidityFlag:
        energyFilterDuration = Params.maxBreathingInhaleExhaleLengthInSeconds

        logger.info("Detecting (small) silent portions")
        energyFilter = prepareMovingAverageFilter(
            energyFilterDuration,
            1 / onsetSignalSamplingFrequency,
        )
        energySignal = sigAPI.convolve(
            np.expand_dims(np.power(breathingOnsetSignal, 2), axis=1),
            energyFilter,
            mode="same",
        )

        energySignalHighPercentile = np.percentile(energySignal, 95)
        energySignal[energySignal >= energySignalHighPercentile] = energySignalHighPercentile

        energyIndicator = np.zeros(energySignal.shape)
        energyIndicator[energySignal <= Params.silenceAbsoluteEnergyThreshold] = 1

        energyIndicatorExpander = prepareMovingAverageFilter(
            energyFilterDuration,
            1 / onsetSignalSamplingFrequency,
            normalizeFlag=False,
        )
        energyIndicator = sigAPI.convolve(
            energyIndicator,
            energyIndicatorExpander,
            mode="same",
        )
        energyIndicator = convertIndicatorToBoolean(energyIndicator)

        silentPeriods = computeStartAndEndTimesFromIndicator(
            energyIndicator,
            breathingOnsetSignalTimeAxisInSeconds,
        )

        del energyIndicator, energySignal
        gc.collect()

        shortSilentPeriods = [
            silentPeriod
            for silentPeriod in silentPeriods
            if silentPeriod[1] - silentPeriod[0] <= Params.maxSilentPeriodDuration
        ]
        shortSignificantSilentPeriods = [
            silentPeriod
            for silentPeriod in shortSilentPeriods
            if silentPeriod[1] - silentPeriod[0] >= Params.minSignificantSilentPeriodDuration
        ]

        silentPauses = [
            silentPeriod
            for silentPeriod in silentPeriods
            if silentPeriod[1] - silentPeriod[0] > Params.minSilentPauseDuration
        ]

        majorityVoteProportionToUseForValidity = (
            Params.majorityVoteProportionForPeriodicity + Params.majorityVoteProportionBumpForValidity
        )
        logger.info("Finished detecting small silent portions")
    else:
        majorityVoteProportionToUseForValidity = Params.majorityVoteProportionForPeriodicity

    fineSTFTTimeResolution = getResolutionFromAxis(fineFeqSTFTTimeAxis)
    fineSTFTFrequencyResolution = getResolutionFromAxis(fineFeqSTFTFreqAxis)

    logger.info("Detecting fine-frequency STFT bins with single dominant frequencies")
    maxFrequenciesInFrame = np.argmax(fineFreqSTFT, axis=0)
    maxFrequenciesInFrame = fineFeqSTFTFreqAxis[maxFrequenciesInFrame]

    # to avoid counting adjacent breathing frequencies as 2 distinct
    # dominant freqquencies -- these can both correspond to 1 frequency due to limited resolution
    # detecting which frames in he fine-frequency STFT have a single dominant frequency
    minDominantFreqeucnyPeakDistance = 2

    validityIndicator = np.array([0] * len(fineFeqSTFTTimeAxis))
    for n, currentSTFTBinNo in enumerate(range(len(fineFeqSTFTTimeAxis))):
        currentFineSTFTFrame = fineFreqSTFT[:, currentSTFTBinNo]
        currentAbsolutePeriodicityThreshold = Params.relativePeriodicityFrequencySelectionThreshold * max(
            currentFineSTFTFrame
        )
        currentFreqPeakIndices = sigAPI.find_peaks(
            currentFineSTFTFrame,
            height=currentAbsolutePeriodicityThreshold,
            distance=minDominantFreqeucnyPeakDistance,
        )[0]

        # need to manually add-back the first and last bins as peaks if they satisfy the condition
        # -- find_peaks otherwise doesn't detect it as of 2021-08-27
        currentFreqPeakIndices = addDominantFrequencyIndexToSelection(
            currentFineSTFTFrame,
            currentAbsolutePeriodicityThreshold,
            currentFreqPeakIndices,
            minDominantFreqeucnyPeakDistance,
            0,
        )

        currentFreqPeakIndices = addDominantFrequencyIndexToSelection(
            currentFineSTFTFrame,
            currentAbsolutePeriodicityThreshold,
            currentFreqPeakIndices,
            minDominantFreqeucnyPeakDistance,
            len(currentFineSTFTFrame) - 1,
        )

        # re-sorting by amplitude level before harmonic suppression
        currentFreqPeakAmplitudes = currentFineSTFTFrame[currentFreqPeakIndices]
        currentFreqPeakAmplitudesSorting = np.argsort(currentFreqPeakAmplitudes)
        currentFreqPeakAmplitudesSorting = np.flip(currentFreqPeakAmplitudesSorting)
        currentFreqPeakIndices = [currentFreqPeakIndices[i] for i in currentFreqPeakAmplitudesSorting]
        del currentFreqPeakAmplitudes, currentFreqPeakAmplitudesSorting

        # detecting and discarding harmonics
        currentDominantFrequencies = fineFeqSTFTFreqAxis[currentFreqPeakIndices]
        if len(currentDominantFrequencies) > 0:
            currentDominantFrequencies = checkForAndRemoveHarmonicsFromFreqencySelection(
                currentDominantFrequencies,
                currentDominantFrequencies[0],
                fineSTFTFrequencyResolution,
                Params.maxBreathingFrequency,
            )

        if len(currentDominantFrequencies) == 1:
            validityIndicator[currentSTFTBinNo] = 1
            dominantBreathingFrequencies[n] = currentDominantFrequencies[0]

    logger.info(
        "Finished detecting fine-frequency STFT bins with single dominant frequencies",
    )

    if useSilenceValidityFlag or restrictBreathingFrequencyJumps:
        logger.info(
            "Enforcing non-validity of any silent portions before treating short silent periods specifically",
        )

        if useSilenceValidityFlag:
            # flagging all silent period as non-valid / non-period at first, to
            # avoid having them accidentally enable other short valid periods later,
            # and to avoid noisy/spurious breaths in large pauses with mainly noise
            for n, silentPeriod in enumerate(silentPeriods):
                validityIndicator = switchFlagsInIntervalOnValidityIndicator(
                    validityIndicator,
                    fineFeqSTFTTimeAxis,
                    silentPeriod,
                    0,
                )

        contiguousPeriodicBinIntervals = computeStartAndEndTimesFromIndicator(
            validityIndicator,
            fineFeqSTFTTimeAxis,
        )

        if restrictBreathingFrequencyJumps:
            logger.info(
                "Discarding validity of ajdacent and/or contiguous periodic regions with breathing frequency jumps",
            )

            # prelininary search for serial apnea regions, they need special treatment

            shortApneaCandidateCenters = computeCenterIndicatorSignalFromIntervals(
                shortSignificantSilentPeriods,
                breathingOnsetSignal.size,
                timeAxis=breathingOnsetSignalTimeAxisInSeconds,
            )

            # preparing a counting filter
            serialApneaFinderFilterApneaNumber = Params.noOfShortSilencesForSerialApneaPredetection
            serialApneaFinderFilter = prepareMovingAverageFilter(
                serialApneaFinderFilterApneaNumber * Params.maxApneaLengthInSecondsInSerialRegions,
                1 / onsetSignalSamplingFrequency,
                normalizeFlag=False,
            )

            serialApneaIndicatorSignal = sigAPI.convolve(
                shortApneaCandidateCenters,
                serialApneaFinderFilter,
                mode="same",
            )
            # checking when there are enough short significant silent periods after another (with numerical tolerance)
            # and leaving some more margin in case some apnead don't have short silent period detections
            serialApneaIndicators = serialApneaIndicatorSignal > serialApneaFinderFilterApneaNumber - 0.01
            del serialApneaIndicatorSignal

            serialApneaIndicators = expandTooShortActiveRegionsInIndicator(
                serialApneaIndicators,
                len(serialApneaFinderFilter),
            )

            serialApneaIndicatorsRegions = computeStartAndEndTimesFromIndicator(
                serialApneaIndicators,
                breathingOnsetSignalTimeAxisInSeconds,
            )
            del serialApneaIndicators, shortApneaCandidateCenters

            gc.collect()

            #######

            for n in range(len(contiguousPeriodicBinIntervals) - 1):
                currentContiguousPeriodicRegion = contiguousPeriodicBinIntervals[n]
                currentContiguousPeriodicRegionDuration = computeTotalIntervalDurations(
                    [currentContiguousPeriodicRegion],
                )

                nextContiguousPeriodicRegion = contiguousPeriodicBinIntervals[n + 1]
                nextContiguousPeriodicRegionDuration = computeTotalIntervalDurations(
                    [nextContiguousPeriodicRegion],
                )

                currentNextRegionGap = (
                    currentContiguousPeriodicRegion[1],
                    nextContiguousPeriodicRegion[0],
                )
                currentNextRegionGapInSerialApneaRegions = siftIntervalsForThoseOverlappingAnyOtherInterval(
                    [currentNextRegionGap],
                    serialApneaIndicatorsRegions,
                )

                if (
                    computeTotalIntervalDurations(
                        currentNextRegionGapInSerialApneaRegions,
                    )
                    > 0
                ):
                    # in this case the "jump" is occuring in a serial apnea region
                    # , so increase jump cuttoff threshold / be more lenient
                    currentmaxAllowableFrequencyJump = Params.maxAllowableFrequencyJumpLenient
                else:
                    # otherwise decrease the jump cutoff threshold / be more strict
                    currentmaxAllowableFrequencyJump = Params.maxAllowableFrequencyJumpStrict

                if (
                    nextContiguousPeriodicRegion[0] - currentContiguousPeriodicRegion[0]
                    <= Params.maxFrequencyJumpCheckupTime
                    and currentContiguousPeriodicRegionDuration
                    >= Params.minTimeBinsForFreqJumpSupression * fineSTFTTimeResolution
                    and nextContiguousPeriodicRegionDuration
                    >= Params.minTimeBinsForFreqJumpSupression * fineSTFTTimeResolution
                ):
                    currentContiguousPeriodicRegionPitches = retrievePitchesForTimeRegion(
                        fineFeqSTFTTimeAxis,
                        dominantBreathingFrequencies,
                        currentContiguousPeriodicRegion,
                    )

                    nextContiguousPeriodicRegionPitches = retrievePitchesForTimeRegion(
                        fineFeqSTFTTimeAxis,
                        dominantBreathingFrequencies,
                        nextContiguousPeriodicRegion,
                    )

                    # if the dominant frequency jumps way too much between the end of the
                    # current contiguous valid bin region and the next
                    if (
                        abs(
                            nextContiguousPeriodicRegionPitches[0] - currentContiguousPeriodicRegionPitches[-1],
                        )
                        / (nextContiguousPeriodicRegion[0] - currentContiguousPeriodicRegion[1])
                        >= currentmaxAllowableFrequencyJump
                    ):
                        # then throw both contiguous bins out the window -- they become invalid
                        # as this is most probably a noise period

                        if (
                            currentContiguousPeriodicRegionDuration
                            > Params.imbalancedRegionFactor * nextContiguousPeriodicRegionDuration
                        ):
                            cancelNextContiguousPeriod = True
                            cancelCurrentContiguousPeriod = False
                        elif (
                            nextContiguousPeriodicRegionDuration
                            > Params.imbalancedRegionFactor * currentContiguousPeriodicRegionDuration
                        ):
                            cancelNextContiguousPeriod = False
                            cancelCurrentContiguousPeriod = True
                        else:
                            cancelNextContiguousPeriod = True
                            cancelCurrentContiguousPeriod = True

                        if cancelCurrentContiguousPeriod:
                            validityIndicator = switchFlagsInIntervalOnValidityIndicator(
                                validityIndicator,
                                fineFeqSTFTTimeAxis,
                                currentContiguousPeriodicRegion,
                                0,
                            )
                        if cancelNextContiguousPeriod:
                            validityIndicator = switchFlagsInIntervalOnValidityIndicator(
                                validityIndicator,
                                fineFeqSTFTTimeAxis,
                                nextContiguousPeriodicRegion,
                                0,
                            )

                    # or if any of the individual contiguous valid regions has an abnormal pitch jump in it, discard it
                    if contiguousPitchesHaveAbnormalFrequencyJump(
                        currentContiguousPeriodicRegionPitches,
                        currentContiguousPeriodicRegion,
                    ):
                        validityIndicator = switchFlagsInIntervalOnValidityIndicator(
                            validityIndicator,
                            fineFeqSTFTTimeAxis,
                            currentContiguousPeriodicRegion,
                            0,
                        )

                    if contiguousPitchesHaveAbnormalFrequencyJump(
                        nextContiguousPeriodicRegionPitches,
                        nextContiguousPeriodicRegion,
                    ):
                        validityIndicator = switchFlagsInIntervalOnValidityIndicator(
                            validityIndicator,
                            fineFeqSTFTTimeAxis,
                            nextContiguousPeriodicRegion,
                            0,
                        )

        if useSilenceValidityFlag:
            logger.info(
                "Enforcing validity of small silent portions adjacent to periodic breathing",
            )
            contiguousPeriodicBinIntervals = computeStartAndEndTimesFromIndicator(
                validityIndicator,
                fineFeqSTFTTimeAxis,
            )

            # filling the validity "gaps" for short silence periods between otherwise valid (periodic) regions
            for n, silentPeriod in enumerate(shortSilentPeriods):
                if n == 0:
                    previousSilentPeriod = (0, 0)
                else:
                    previousSilentPeriod = shortSilentPeriods[n - 1]

                if n == (len(shortSilentPeriods) - 1):
                    nextSilentPeriod = (
                        breathingOnsetSignalTimeAxisInSeconds[-1],
                        breathingOnsetSignalTimeAxisInSeconds[-1],
                    )
                else:
                    nextSilentPeriod = shortSilentPeriods[n + 1]

                # start and end of non-silent neighborhood on the left of the current silence period
                currentImmediateNonSilentNeighborHoodLeft = (
                    previousSilentPeriod[1],
                    silentPeriod[0],
                )
                currentContiguousPeriodicBinPeriodsWithinImmediateNonSilentNeighborhoodLeft = (
                    locateSignificantPeriodicIntervalsInOtherInterval(
                        contiguousPeriodicBinIntervals,
                        currentImmediateNonSilentNeighborHoodLeft,
                    )
                )

                # start and end of non-silent neighborhood on the right of the current silence period
                currentImmediateNonSilentNeighborHoodRight = (
                    silentPeriod[1],
                    nextSilentPeriod[0],
                )
                currentContiguousPeriodicBinPeriodsWithinImmediateNonSilentNeighborhoodRight = (
                    locateSignificantPeriodicIntervalsInOtherInterval(
                        contiguousPeriodicBinIntervals,
                        currentImmediateNonSilentNeighborHoodRight,
                    )
                )

                if (
                    len(
                        currentContiguousPeriodicBinPeriodsWithinImmediateNonSilentNeighborhoodLeft,
                    )
                    >= 1
                    and len(
                        currentContiguousPeriodicBinPeriodsWithinImmediateNonSilentNeighborhoodRight,
                    )
                    >= 1
                ):
                    validityIndicator = switchFlagsInIntervalOnValidityIndicator(
                        validityIndicator,
                        fineFeqSTFTTimeAxis,
                        silentPeriod,
                        1,
                    )

            logger.info(
                "Finished enforcing validity of small silent portions adjacent to periodic breathing",
            )

    logger.info("Computing majoritarily-valid regions")
    # computing a "vote" where a region of a certain length around a fine STFT time bin is flagged as periodic
    # if a certain percentage (signifying majority) of bins in it are flagged as periodic,
    # and preparing vote time-smoothing
    periodicityRegionMinLengthInSeconds = (
        Params.noOfBreathstoEnforceAfterApnea + Params.noOfPreviousBreathstoEnforceBeforeApnea
    ) * (Params.maxBreathingLengthInSeconds + fineSTFTTimeResolution) + Params.maxApneaLengthInSeconds

    periodicityRegionVoteFilter = prepareMovingAverageFilter(
        periodicityRegionMinLengthInSeconds,
        fineSTFTTimeResolution,
    )
    periodicityPeriodFilterLength = len(periodicityRegionVoteFilter)
    periodicityRegionVoteSmoothingFilter = prepareMovingAverageFilter(
        periodicityRegionMinLengthInSeconds / 6,
        fineSTFTTimeResolution,
    )
    periodicityRegionVoteFilterWithSmoothing = sigAPI.convolve(
        periodicityRegionVoteFilter,
        periodicityRegionVoteSmoothingFilter,
        mode="full",
    )

    expandedValidityIndicator = np.expand_dims(validityIndicator, 1)
    validityRegionVote = sigAPI.convolve(
        expandedValidityIndicator,
        periodicityRegionVoteFilterWithSmoothing,
        mode="same",
    )
    logger.info("Finished computing majoritarily-valid regions")

    logger.info("Expanding too-small valid regions to minimum size")
    # identifying centers of regions where the vote is majoritarily for periodicity (a pass), and computing meta-regions
    # where the center-vote is contiguously passed
    validBreathingPeriodsIndicator = np.array(
        validityRegionVote >= majorityVoteProportionToUseForValidity,
    )

    validBreathingPeriodsIndicator = expandTooShortActiveRegionsInIndicator(
        validBreathingPeriodsIndicator,
        periodicityPeriodFilterLength,
    )

    invalidBreathingPeriodsIndicator = convertIndicatorToBoolean(
        validBreathingPeriodsIndicator,
        lookForOnOrOff="off",
    )

    validBreathingPeriodsIndicator = convertIndicatorToBoolean(
        validBreathingPeriodsIndicator,
        lookForOnOrOff="on",
    )

    logger.info(
        "Finished expanding too-small valid regions to minimum size",
    )

    validBreathingPeriodsStartEnds = computeStartAndEndTimesFromIndicator(
        validBreathingPeriodsIndicator,
        fineFeqSTFTTimeAxis,
    )
    invalidBreathingPeriodsStartEnds = computeStartAndEndTimesFromIndicator(
        invalidBreathingPeriodsIndicator,
        fineFeqSTFTTimeAxis,
    )

    # for the periods where the original onset signal is full of NaNs, these are appended as invalid -- no signal there
    invalidBreathingPeriodsStartEnds = invalidBreathingPeriodsStartEnds + [
        (
            breathingOnsetSignalTimeAxisInSeconds[0] - noOfStartingNaNsInSeconds,
            breathingOnsetSignalTimeAxisInSeconds[0],
        ),
        (
            breathingOnsetSignalTimeAxisInSeconds[-1],
            breathingOnsetSignalTimeAxisInSeconds[-1] + noOfStartingNaNsInSeconds,
        ),
    ]

    return (
        validBreathingPeriodsStartEnds,
        invalidBreathingPeriodsStartEnds,
        silentPauses,
        fineFeqSTFTTimeAxis,
        dominantBreathingFrequencies,
    )


def detectOnsets(onsetSignal, onsetSignalTimeAxis, heightConstraint: float = None):
    """Function used to detect onsets on a breathing, snoring or other onset signal

    An onset is for instance a positive peak in the positive part of an onset signal, or a negative peak in the negative
    part of an onset signal. For such signals these would correspond respectively to the start/end of breathing, snoring
    and similar events (depending on the type of onset signal that's used)

    :param onsetSignal: breathing, snoring or onset signal on which to detect onsets
    :type onsetSignal: list or 1D numpy arrays of (preferably of floats, or of ints)
    :param onsetSignalTimeAxis: time axis (in seconds) corresponding to the onset signal
    :type onsetSignalTimeAxis: list or 1D numpy arrays of (preferably of floats, or of ints)
    :param heightConstraint: minimum height threshold (in absolute not relative terms) to enforce on the onsets
        to be detected -- discards any onsets with less than this. Set to None if want no such threhsolding is desired.
        Defaults to None
    :type heightConstraint: float, optional
    :return:
        onsetSignalDetections: indices for which onsets are detected on the onset signal
        onsetSignalDetectionsInSeconds: the onsetSignalDetections translated to seconds / time instants
    :rtype:
        onsetSignalDetections: numpy array of integers
        onsetSignalDetectionsInSeconds: numpy array of floats
    """
    if heightConstraint is not None:
        logger.info("Detecting onsets with a height constraint")
    else:
        logger.info("Detecting onsets (WITHOUT a height constraint)")
    onsetSignalTimeStep = getResolutionFromAxis(onsetSignalTimeAxis)
    onsetSignalDetections = sigAPI.find_peaks(
        onsetSignal,
        height=heightConstraint,
        distance=Params.minBreathingDistanceInSeconds / (onsetSignalTimeStep),
    )[0]
    onsetSignalDetectionsInSeconds = onsetSignalTimeAxis[onsetSignalDetections]

    return onsetSignalDetections, onsetSignalDetectionsInSeconds


def enforceValidityOnDetections(
    detectionsInSeconds,
    detections,
    validBreathingPeriodsStartsEnds,
):
    """Function used to sift onset detections and only keep those that fall in valid periods

    :param detectionsInSeconds: time instants corresponding to the onset detections
    :type detectionsInSeconds: list or 1D numpy array of (preferably of floats, or of ints)
    :param detections: incides corresponding to the onset detections
    :type detections: list or 1D numpy array of integers
    :param validBreathingPeriodsStartsEnds: time intervals for which the bio-signal can be considered valid
    :type validBreathingPeriodsStartsEnds: list of 2-element lists or pair-tuples, or 2D numpy array
        (preferably of floats, or of ints)
    :return:
        detections: indices for which onsets are detected on the onset signal, after sifting
        detectionsInSeconds: the detections translated to seconds / time instants, after sifting
    :rtype:
        detections: numpy array of integers
        detectionsInSeconds: numpy array of floats (or ints depending on input)
    """
    logger.info("Enforcing / sifting detections to lie within valid intervals")
    if len(validBreathingPeriodsStartsEnds) > 0:
        (
            detectionsInSeconds,
            retainedTimesIndicesInOriginalSet,
            _,
        ) = retainTimesWithinPredefinedWindows(
            detectionsInSeconds,
            validBreathingPeriodsStartsEnds,
        )
        detections = detections[retainedTimesIndicesInOriginalSet]
    else:
        detectionsInSeconds = []
        detections = []

    return detectionsInSeconds, detections


def detectValidOnsetsOfSufficientHeight(
    onsetSignal,
    onsetSignalTimeAxis,
    minimumHeight: float,
    validBreathingPeriodsStartsEnds=None,
):
    """Function used to detect onsets in an onset signal while restricting them to a minimum height and valid regions

    :param onsetSignal: breathing, snoring or onset signal on which to detect onsets
    :type onsetSignal: list or numpy array of floats
    :param onsetSignalTimeAxis: time axis in seconds corresponding to the onset signal
    :type onsetSignalTimeAxis: list or numpy array of floats
    :param minimumHeight: minimum height (in absolute not relative terms) a peak/onset needs to have to be detected
    :type minimumHeight: float
    :param validBreathingPeriodsStartsEnds: time intervals for which the bio-signal can be considered valid,
        onsets are only detected within these intervals, defaults to None
    :type validBreathingPeriodsStartsEnds: list of 2-element lists of pair-tuples, or 2D numpy array
        (preferably of floats, or of ints), optional
    :return:
        onsetDetections: indices for which onsets are detected on the onset signal
        onsetDetectionsInSeconds: the onsetDetections translated to seconds / time instants
        onsetDetectionStrengths: strength / salience of the detected onsets
    :rtype:
        onsetDetections: numpy array of integers
        onsetDetectionsInSeconds: numpy array of floats
        onsetDetectionStrengths: numpy array of floats
    """
    onsetDetections, onsetDetectionsInSeconds = detectOnsets(
        onsetSignal,
        onsetSignalTimeAxis,
        heightConstraint=minimumHeight,
    )

    if validBreathingPeriodsStartsEnds is not None:
        (
            onsetDetectionsInSeconds,
            onsetDetections,
        ) = enforceValidityOnDetections(
            onsetDetectionsInSeconds,
            onsetDetections,
            validBreathingPeriodsStartsEnds,
        )
    onsetDetectionStrengths = onsetSignal[onsetDetections]

    return (
        onsetDetections,
        onsetDetectionsInSeconds,
        onsetDetectionStrengths,
    )


def pairPositiveAndNegativeOnsetsAndTreatCollisions(
    positiveOnsetDetectionsInSeconds,
    positiveOnsetDetectionStrengths,
    negativeOnsetDetectionsInSeconds,
    negativOnsetDetectionStrengths,
):
    """
    Function used to pair positive onset detections with negative onset
    detections while repairing ambiguities/collisions

    The pairing is needed to identify breathing and snoring events, as each of those consists of a start
    (positive onset) and an end (negative onset).

    :param positiveOnsetDetectionsInSeconds: set of positive onset detections (in seconds)
    :type positiveOnsetDetectionsInSeconds: list or 1D numpy array of floats
    :param positiveOnsetDetectionStrengths: the "strength" or salience corresponding to each of the positive onsets
    :type positiveOnsetDetectionStrengths: list or 1D numpy array of floats
    :param negativeOnsetDetectionsInSeconds: set of negative onset detections (in seconds)
    :type negativeOnsetDetectionsInSeconds: list or 1D numpy array of floats
    :param negativOnsetDetectionStrengths: the "strength" or salience corresponding to each of the negative onsets
    :type negativOnsetDetectionStrengths: list or 1D numpy array of floats
    :return:
        eventStartEndsInSeconds: start and end time instants of the resulting events in seconds
        eventStrengths: combined/composite strength/salience of each of the resulting events
    :rtype:
        eventStartEndsInSeconds: list of pair-tuples (of floats)
        eventStrengths: list of floats
    """
    logger.info("Pairing starts (onsets) and ends of events while treating collisions")
    eventStartEndsInSeconds = []  # refined list of onset/end pairs
    eventStrengths = []  # refined list of onset/end pair strengths

    negativeOnsetDetectionsInSeconds = list(negativeOnsetDetectionsInSeconds)
    negativOnsetDetectionStrengths = list(negativOnsetDetectionStrengths)

    positiveOnsetsToExcludeFromFurtherProcessing = []

    for n, positiveOnsetDetection in enumerate(positiveOnsetDetectionsInSeconds):
        positiveOnsetDetectionStrength = positiveOnsetDetectionStrengths[n]
        if n not in positiveOnsetsToExcludeFromFurtherProcessing:
            # searching for end/negative peaks that occur AFTER at least 1 event length from the onset,
            # and within the maximum inhale + exhale breathing length -- only then can it be an actual
            # breath/snoring and not an impulsive noise or bed movement etc
            followupNegativeOnsetDetections = [
                (k, i)
                for k, i in enumerate(negativeOnsetDetectionsInSeconds)
                if i > positiveOnsetDetection + Params.minBreathingLengthInSeconds
                and i <= positiveOnsetDetection + Params.maxBreathingInhaleExhaleLengthInSeconds
            ]
            followupOnsetDetectionsInSeconds = [i[1] for i in followupNegativeOnsetDetections]
            followupOnsetDetectionsIndices = [i[0] for i in followupNegativeOnsetDetections]

            del followupNegativeOnsetDetections

            if len(followupOnsetDetectionsInSeconds) > 0:
                firstFollowupNegativeOnsetDetectionsInSeconds = followupOnsetDetectionsInSeconds[0]
                firstFollowupNegativeOnsetDetectionsIndex = followupOnsetDetectionsIndices[0]

                collidingPositiveOnsetDetections = [
                    (k, i, positiveOnsetDetectionStrengths[k])
                    for k, i in enumerate(positiveOnsetDetectionsInSeconds)
                    if i > positiveOnsetDetection + Params.minBreathingLengthInSeconds
                    and i <= firstFollowupNegativeOnsetDetectionsInSeconds
                    and k not in positiveOnsetsToExcludeFromFurtherProcessing
                ]
                collidingPositiveOnsetDetectionsIndices = [i[0] for i in collidingPositiveOnsetDetections]
                collidingPositiveOnsetDetectionsInSeconds = [i[1] for i in collidingPositiveOnsetDetections]
                collidingPositiveOnsetDetectionsStrengths = [i[2] for i in collidingPositiveOnsetDetections]
                del collidingPositiveOnsetDetections

                strongerCollidingPositiveOnsetDetectionsTempIndices = [
                    k
                    for k, i in enumerate(collidingPositiveOnsetDetectionsStrengths)
                    if i > positiveOnsetDetectionStrength
                ]

                if len(collidingPositiveOnsetDetectionsIndices) == 0:
                    currentEventToAdd = (
                        positiveOnsetDetection,
                        firstFollowupNegativeOnsetDetectionsInSeconds,
                    )
                    currentEventStrengthToAdd = (
                        abs(positiveOnsetDetectionStrength)
                        + abs(
                            negativOnsetDetectionStrengths[firstFollowupNegativeOnsetDetectionsIndex],
                        )
                    ) / 2
                    currentPositiveOnsetsToExcludeFromProcessing = [n]
                    currentNegativeOnsetIndicesToDelete = firstFollowupNegativeOnsetDetectionsIndex

                else:
                    if len(strongerCollidingPositiveOnsetDetectionsTempIndices) > 0:
                        # if one of the colliding onsets is stronger than the basis / current breathing/snoring onset

                        collidingStrongestPositiveOnsetDetectionTempIndex = (
                            strongerCollidingPositiveOnsetDetectionsTempIndices[
                                np.argmax(
                                    [
                                        collidingPositiveOnsetDetectionsStrengths[i]
                                        for i in strongerCollidingPositiveOnsetDetectionsTempIndices
                                    ],
                                )
                            ]
                        )

                        currentEventToAdd = (
                            collidingPositiveOnsetDetectionsInSeconds[
                                collidingStrongestPositiveOnsetDetectionTempIndex
                            ],
                            firstFollowupNegativeOnsetDetectionsInSeconds,
                        )
                        currentEventStrengthToAdd = (
                            abs(
                                collidingPositiveOnsetDetectionsStrengths[
                                    collidingStrongestPositiveOnsetDetectionTempIndex
                                ],
                            )
                            + abs(
                                negativOnsetDetectionStrengths[firstFollowupNegativeOnsetDetectionsIndex],
                            )
                        ) / 2

                        currentNegativeOnsetIndicesToDelete = firstFollowupNegativeOnsetDetectionsIndex

                        currentPositiveOnsetsToExcludeFromProcessing = [
                            collidingPositiveOnsetDetectionsIndices[collidingStrongestPositiveOnsetDetectionTempIndex],
                        ]
                        currentPositiveOnsetsToExcludeFromProcessing = (
                            currentPositiveOnsetsToExcludeFromProcessing
                            + list(
                                set(collidingPositiveOnsetDetectionsIndices).difference(
                                    set(
                                        [
                                            collidingPositiveOnsetDetectionsIndices[
                                                collidingStrongestPositiveOnsetDetectionTempIndex
                                            ],
                                        ],
                                    ),
                                ),
                            )
                        )
                        currentNegativeOnsetIndicesToDelete = None
                    else:
                        # if NONE of the colliding onsets is stronger than the basis / current breathing/snoring onset
                        currentEventToAdd = (
                            positiveOnsetDetection,
                            firstFollowupNegativeOnsetDetectionsInSeconds,
                        )
                        currentEventStrengthToAdd = (
                            abs(positiveOnsetDetectionStrength)
                            + abs(
                                negativOnsetDetectionStrengths[firstFollowupNegativeOnsetDetectionsIndex],
                            )
                        ) / 2
                        currentPositiveOnsetsToExcludeFromProcessing = collidingPositiveOnsetDetectionsIndices
                        currentNegativeOnsetIndicesToDelete = None

                eventStartEndsInSeconds.append(
                    currentEventToAdd,
                )  # to be included in the refined list of onset/end pairs
                eventStrengths.append(
                    currentEventStrengthToAdd,
                )  # to be included in the refined list of onset/end pair strengths
                if currentNegativeOnsetIndicesToDelete is not None:
                    negativeOnsetDetectionsInSeconds.pop(
                        firstFollowupNegativeOnsetDetectionsIndex,
                    )  # to be excluded form further processing
                    negativOnsetDetectionStrengths.pop(
                        firstFollowupNegativeOnsetDetectionsIndex,
                    )  # to be excluded form further processing
                positiveOnsetsToExcludeFromFurtherProcessing = (
                    positiveOnsetsToExcludeFromFurtherProcessing + currentPositiveOnsetsToExcludeFromProcessing
                )  # to be excluded form further processing

    return eventStartEndsInSeconds, eventStrengths


def computeAbsoluteOnsetThresholdAnchor(onsetSignal, onsetDetections):
    """
    Funciton used to compute the absolute anchor reference
    (absolute amplutude level) to be used for onset thresholding

    The snorefox algorithm uses a relative, signal-adapted threshold for detecting breathing, snoring and similar events
    This threshold being in percent it's always relative to an "anchor" or absolute ampltiude value on which the
    percentage is actually taken. This function computes that absolute level anchor.

    :param onsetSignal: breathing, snoring or onset signal on which to detect onsets
    :type onsetSignal: list or 1D numpy array (preferably of floats, or of ints)
    :param onsetDetections: preliminary set of ondset detections indices without any thresholding
    :type onsetDetections: list of integers
    :return: absolute level anchor
    :rtype: float
    """
    logger.info("Computing absolute anchor in onset signal")

    # re-peak picking using Median Absolute Deviation for sifting
    currentAbsoluteOnsetThresholdAnchor = find_peak_without_outliers(
        np.nan_to_num(
            onsetSignal[onsetDetections],
            copy=False,
            nan=0,
            posinf=0,
            neginf=0,
        ),
        outlier_threshold=10,
    )

    return currentAbsoluteOnsetThresholdAnchor


def find_peak_without_outliers(data, outlier_threshold=3):
    """
    Find the peak value in the given dataset, excluding outliers.

    Parameters:
    - data (array-like): The input dataset.
    - outlier_threshold (float, optional): The threshold for identifying outliers.
      Values beyond this threshold times the Median Absolute Deviation (MAD)
      from the median are considered outliers. Default is 3.

    Returns:
    - peak_value_no_outliers: The peak value in the dataset after excluding outliers.

    Notes:
    - Outliers are identified using the Median Absolute Deviation (MAD) method,
      and values beyond the specified threshold are considered outliers.
    - The peak value is then computed from the dataset after excluding the identified outliers.

    Example:
    data = [1, 2, 3, 100, 4, 5, 6]
    find_peak_without_outliers(data)
    6
    """
    # Compute the Median
    median_value = np.nanmedian(data)

    # Calculate the Median Absolute Deviation (MAD)
    mad = np.nanmedian(np.abs(data - median_value))

    # Identify Outliers
    outliers = (np.abs(data - median_value) / mad) > outlier_threshold

    # Compute the Peak Value Excluding Outliers
    data_no_outliers = data[~outliers]
    peak_value_no_outliers = np.nanmax(data_no_outliers)

    return peak_value_no_outliers


def detectBreathingOrSnoringEvents(
    breathingOrSnoringOnsetSignal,
    onsetSignalTimeAxisInSeconds,
    impulseOnsetSignal=None,
    validBreathingPeriodsStartsEnds=None,
    invalidBreathingPeriodsStartsEnds=None,
    relativeBreathingOrSnoringThreshold: float = Params.relativeBreathThreshold,
):
    """Function used to detected events on different types of onset signals, e.g. breathing, snoring events etc

    event (start, end) detection for breathing or snoring events. An event is for example a full breathing: inhale
    (potentially followed by an exhale). The function assumes it's used for both - breathing or snoring.
    Adapts to breathing+snoring event detection by default, but can be used for snoring event detection specifically
    while (largely) excluding breathing

    :param breathingOrSnoringOnsetSignal: onset signal to detect events on, usually breathing or snoring onset signal
    :type breathingOrSnoringOnsetSignal: list or 1D numpy array (preferably of floats, or of ints)
    :param onsetSignalTimeAxisInSeconds: time axis in seconds corresponding to the provided onset signal
    :type onsetSignalTimeAxisInSeconds: list or 1D numpy array (preferably of floats, or of ints)
    :param impulseOnsetSignal: an impulse onset signal for detecting impulse noise events on. Keep to None if no such
        impulsive noise detection is desired. Assumes same time axis as main onset signal. Defaults to None
    :type impulseOnsetSignal: list or 1D numpy array (preferably of floats, or of ints), optional
    :param validBreathingPeriodsStartsEnds: time intervals of the onset signal considered valid. The event detection is
        restricted to these intervals. Keep to None if you want no such restriction / you want to detect events on
        the whole onset signal. Defaults to None
    :type validBreathingPeriodsStartsEnds: list of 2-element lists of pair-tuples, or 2D numpy array
        (preferably of floats, or of ints), optional
    :param invalidBreathingPeriodsStartsEnds: time intervals of the onset signal is considered invalid. Supplied or not
        always together with validBreathingPeriodsStartsEnds. Defaults to None
    :type invalidBreathingPeriodsStartsEnds: list of 2-element lists of pair-tuples, or 2D numpy array
        (preferably of floats, or of ints), optional
    :param relativeBreathingOrSnoringThreshold: relative threshold in percent to use for the detection.
        Defaults to Params.relativeBreathThreshold
    :type relativeBreathingOrSnoringThreshold: float, optional
    :return:
        eventStartEndsInSeconds: start and end time instants (in seconds) of the resulting events
        eventStrengths: combined/composite strength/salience of each of the resulting events
        impulseOnsetDetectionsInSeconds: the impulsive noise event detections (time instants,
            only returned if impulsive onset signal was supplied)
    :rtype:
        eventStartEndsInSeconds: list of pair-tuples (of floats)
        eventStrengths: 1D numpy array of floats
        impulseOnsetDetectionsInSeconds: 1D numpy array of floats
    """
    logger.info("Detecting events (snoring, breathing etc)")
    logger.info(
        "First detecting preliminary onsets (for anchoring) from main onset signal",
    )

    logger.info("Copying/extracting and treating positive onset part")
    usedSTFThopLength = getResolutionFromAxis(onsetSignalTimeAxisInSeconds)

    positiveBreathingOnsetSignal = np.copy(breathingOrSnoringOnsetSignal)
    positiveBreathingOnsetSignal[positiveBreathingOnsetSignal < 0] = 0

    # detect first set of positive peaks with no threshold
    (
        positiveBreathingOnsetDetections,
        positiveBreathingOnsetDetectionsInSeconds,
        _,
    ) = detectValidOnsetsOfSufficientHeight(
        positiveBreathingOnsetSignal,
        onsetSignalTimeAxisInSeconds,
        0,
        validBreathingPeriodsStartsEnds,
    )
    del _
    logger.info(
        "Finished copying/extracting and treating positive onset part",
    )

    logger.info("Coying/extracting and treating negative onset part")
    negativeBreathingOnsetSignal = np.copy(breathingOrSnoringOnsetSignal)
    del breathingOrSnoringOnsetSignal  # can just reconstruct this back by addition, better save memory
    negativeBreathingOnsetSignal[negativeBreathingOnsetSignal >= 0] = 0

    # detect first set of negative peaks with no threshold
    (
        negativeBreathingOnsetDetections,
        negativeBreathingOnsetDetectionsInSeconds,
        _,
    ) = detectValidOnsetsOfSufficientHeight(
        -negativeBreathingOnsetSignal,
        onsetSignalTimeAxisInSeconds,
        0,
        validBreathingPeriodsStartsEnds,
    )
    del _
    logger.info(
        "Finished coying/extracting and treating negative onset part",
    )

    # if impulsive noise onset signal is given, check for impulsive noise bursts / events
    #  and use them for sorting spurious breathing peaks
    if impulseOnsetSignal is not None:
        logger.info("Detecting impulsive events/noise")
        # ok to use X% of the maxmimum onset here, as the impulsive noise onsets are naturally the loudest
        # very unlikely to have a spurious peak in the onset signal not corresponding to impulsive noise
        currentLowerImpulseThreshold = Params.relativeImpulseThreshold * np.nanmax(
            impulseOnsetSignal,
        )
        impulseOnsetDetections, impulseOnsetDetectionsInSeconds = detectOnsets(
            impulseOnsetSignal,
            onsetSignalTimeAxisInSeconds,
            heightConstraint=currentLowerImpulseThreshold,
        )
    else:
        logger.info("Proceeding without impulsive events/noise detection")
        impulseOnsetDetections, impulseOnsetDetectionsInSeconds = (None, None)

    if len(positiveBreathingOnsetDetections) > 0 and len(negativeBreathingOnsetDetections) > 0:
        # # remove as any outliers as possible from rudimentary initial breathing detections
        if impulseOnsetDetections is not None:
            logger.info(
                "Sifting out positive/negative onsets (starts and ends) near impulsive noise / outliers",
            )
            (
                positiveBreathingOnsetDetections,
                negativeBreathingOnsetDetections,
            ) = siftOutDetectionsCloseToOutliers(
                (positiveBreathingOnsetDetections, negativeBreathingOnsetDetections),
                impulseOnsetDetections,
                Params.impulsiveNoiseSupressionNeighborhoodStrictInSeconds / usedSTFThopLength,
                Params.impulsiveNoiseSupressionNeighborhoodLeniantInSeconds / usedSTFThopLength,
                Params.noOfImpulsiveOutliersToAllowInLenaintNeighborhood,
                Params.impulseCausalityTolerance / usedSTFThopLength,
            )
            positiveBreathingOnsetDetectionsInSeconds = onsetSignalTimeAxisInSeconds[positiveBreathingOnsetDetections]
            negativeBreathingOnsetDetectionsInSeconds = onsetSignalTimeAxisInSeconds[negativeBreathingOnsetDetections]

        logger.info(
            "Determining signal-adapted / relative threshold for positive/negative onset detection",
        )
        # if the measurement is not completely empty of breathing or other events

        currentLowerBreathingOrSnoringThresholdPositive = (
            relativeBreathingOrSnoringThreshold
            * computeAbsoluteOnsetThresholdAnchor(
                positiveBreathingOnsetSignal,
                positiveBreathingOnsetDetections,
            )
        )
        currentLowerBreathingOrSnoringThresholdNegative = (
            relativeBreathingOrSnoringThreshold
            * computeAbsoluteOnsetThresholdAnchor(
                -negativeBreathingOnsetSignal,
                negativeBreathingOnsetDetections,
            )
        )

        logger.info("Detecting positive onsets (event starts) to be used for events")
        # detecting positive onsets (breathing/snoring starts) using the determined threshold
        (
            _,
            positiveBreathingOnsetDetectionsInSeconds,
            positiveBreathingOnsetDetectionStrengths,
        ) = detectValidOnsetsOfSufficientHeight(
            positiveBreathingOnsetSignal,
            onsetSignalTimeAxisInSeconds,
            currentLowerBreathingOrSnoringThresholdPositive,
            validBreathingPeriodsStartsEnds,
        )
        del positiveBreathingOnsetSignal

        logger.info("Detecting negative onsets (event ends) to be used for events")
        # detecting negative onsets (breathing/snoring ends) using the SAME determined threshold and parameters
        (
            _,
            negativeBreathingOnsetDetectionsInSeconds,
            negativeBreathingOnsetDetectionStrengths,
        ) = detectValidOnsetsOfSufficientHeight(
            -negativeBreathingOnsetSignal,
            onsetSignalTimeAxisInSeconds,
            currentLowerBreathingOrSnoringThresholdNegative,
            validBreathingPeriodsStartsEnds,
        )
        del negativeBreathingOnsetSignal

        if impulseOnsetDetections is not None:
            logger.info(
                "Sifting out positive/negative event onsets near impulsive noise / outliers",
            )
            # sift out breathing detections that are too close to the impulsive noise bursts
            (
                positiveBreathingOnsetDetectionsInSeconds,
                negativeBreathingOnsetDetectionsInSeconds,
            ) = siftOutDetectionsCloseToOutliers(
                (
                    positiveBreathingOnsetDetectionsInSeconds,
                    negativeBreathingOnsetDetectionsInSeconds,
                ),
                impulseOnsetDetectionsInSeconds,
                Params.impulsiveNoiseSupressionNeighborhoodStrictInSeconds,
                Params.impulsiveNoiseSupressionNeighborhoodLeniantInSeconds,
                Params.noOfImpulsiveOutliersToAllowInLenaintNeighborhood,
                Params.impulseCausalityTolerance,
            )

        (
            eventStartEndsInSeconds,
            eventStrengths,
        ) = pairPositiveAndNegativeOnsetsAndTreatCollisions(
            positiveBreathingOnsetDetectionsInSeconds,
            positiveBreathingOnsetDetectionStrengths,
            negativeBreathingOnsetDetectionsInSeconds,
            negativeBreathingOnsetDetectionStrengths,
        )
    else:
        del negativeBreathingOnsetSignal, positiveBreathingOnsetSignal
        eventStartEndsInSeconds = []
        eventStrengths = []
        impulseOnsetDetectionsInSeconds = []

    logger.info("Detected a total of " + str(len(eventStartEndsInSeconds)) + " events")

    return eventStartEndsInSeconds, eventStrengths, impulseOnsetDetectionsInSeconds


def siftBreathingEventsForSpecificPastOrFutureNumberAllowance(
    breathingEvents,
    cutoffTime: float,
    noOfBreathsToKeep: int,
    pastOrFuture: str = "past",
):
    """Function used to look for and keep a select number of breathing or similar events before/after a cutoff time

    :param breathingEvents: list of breathing events with start/end times in seconds and event strength
    :type breathingEvents: list of dictionaries with 2-element lists or pair-tuples, or 2D numpy array (of floats)
    :param cutoffTime: time (in seconds) after/before which to look for breathing events
    :type cutoffTime: float
    :param noOfPastBreathsToKeep: number of breaths to keep before/after the cutoff time
    :type noOfPastBreathsToKeep: int
    :param pastOrFuture: whether to sift for events before (past) or after (future) the cutoff time,
        the corresponding modes are stores as strings in the variables "past" and
        "future". defaults to "past".
    :type pastOrFuture: str, optional
    :return: a subset of breathing events (limited in number by noOfBreathsToKeep) falling within the search window
    :rtype: list of pair-tuples (of floats)
    """
    if pastOrFuture == "past":
        operatorToUseFirstCheck = operator.ge
        operatorToUseSecondCheck = operator.le
        funToUse = max
        limitToCompareWith = 0
        signToUse = -1
        breathStartOrEndToCompareAgainstCutoff = 1  # breath end
    elif pastOrFuture == "future":
        operatorToUseFirstCheck = operator.le
        operatorToUseSecondCheck = operator.ge
        funToUse = min
        limitToCompareWith = np.inf
        signToUse = +1
        breathStartOrEndToCompareAgainstCutoff = 0  # breath start

    # Taking the ceiling of 1.02 times that duration just to leave a 2% error margin in the duration.
    return [
        breathingEvent
        for breathingEvent in breathingEvents
        if operatorToUseFirstCheck(
            breathingEvent["startEnd"][breathStartOrEndToCompareAgainstCutoff],
            funToUse(
                cutoffTime
                + signToUse
                * np.ceil(
                    1.02 * noOfBreathsToKeep * Params.maxBreathingDistanceInSeconds,
                ),
                limitToCompareWith,
            ),
        )
        and operatorToUseSecondCheck(
            breathingEvent["startEnd"][breathStartOrEndToCompareAgainstCutoff],
            cutoffTime,
        )
    ]


def computeAverageAbsoluteOnsetAmplitude(
    onsetSignal,
    onsetSignalTimeAxis,
    startTime: float = None,
    endTime: float = None,
    onsetSignalTimeAxisMinimum=None,
    onsetSignalTimeAxisMaximum=None,
    onsetSignalTimeAxisIsUniform=True,
):
    """Function used to compute average onset amplitude in a time interval on an onset signal

    :param onsetSignal: onset signal on which to compute the average amplitudes
    :type onsetSignal: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param onsetSignalTimeAxis: time axis corresponding to the breathing onset signal
    :type onsetSignalTimeAxis: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param startTime: start time of the desired interval, leave to None to simply use the starting time
    :type startTime: float, optional
    :param endTime: end time of the desired interval, leave to None to simply use the end time
    :type endTime: float, optional
    :return: average/composite amplitude used to describe the "strength" of the signal in the desired interval
    :rtype: float
    """
    if startTime is None:
        startTime = onsetSignalTimeAxis[0]
    if endTime is None:
        endTime = onsetSignalTimeAxis[-1]

    startTimeIndex, endTimeIndex = timeToIndex(
        onsetSignalTimeAxis,
        [startTime, endTime],
        onsetSignalTimeAxisMinimum=onsetSignalTimeAxisMinimum,
        onsetSignalTimeAxisMaximum=onsetSignalTimeAxisMaximum,
        uniformTimeAxis=onsetSignalTimeAxisIsUniform,
    )

    maxPreviousBreathOnsetAmplitude = np.nanmax(
        np.abs(onsetSignal[startTimeIndex:endTimeIndex]),
    )

    return maxPreviousBreathOnsetAmplitude


def lookForwardForApneaClosureWithEnoughBreaths(
    nextBreaths,
    currentApneaBoundaries: tuple[float],
    previousApneaBoundaries: tuple[float],
    newApneaDetections: list[tuple[float]],
    newApneaAmplitudeDeviations: list[float] = None,
    enforceApneaAmplitudeReduction: bool = True,
    enforceApneaAmplitudeIncrease: bool = True,
    breathingOnsetSignal=None,
    breathingOnsetSignalTimeAxis=None,
    breathingOnsetSignalTimeAxisIsUniform=True,
    breathingOnsetSignalTimeAxisMinimum=None,
    breathingOnsetSignalTimeAxisMaximum=None,
    averageOnsetAmplitudeInPreviousNeighborhood: Union[float, int] = None,
):
    """Function used to check which breathing/snoring event should be taken as an end for an apnea candidate

    Sometimes an apnea candidate -- a gap between a breathing event's end and a next breathing event's start --
    should be extended / concatenated to a future apnea candidate / breathing gap, namely if only 1 single
    breath (or n breaths where n is a judiciously chosen number) separates them.
    This logic was confirmed with Prof. Heiser.

    :param nextBreaths: breathing/snoring events after apnea
    :type nextBreaths: list of dictionary containing 2-element lists or pair-tuples (of floats)
        each 2-element pair can be retrieved by the same key, denoting a breath start and end time
        and accessible in the variable "startEnd"
    :param currentApneaBoundaries: boundaries (start / end times in secondss) of the current apnea candidate
    :type currentApneaBoundaries: pair-tuple (of floats)
    :param previousApneaBoundaries: boundaries (start / end times in secondss) of the previous apnea candidate
    :type previousApneaBoundaries: pair-tuple (of floats)
    :param newApneaDetections: running list of apnea detection canddiates
    :type newApneaDetections: list of pair-tuples (of floats)
    :param enforceApneaAmplitudeReduction: True if you want to enable enforcement of breathing/snoring amplitude
        (strength/salience) reduction for detected apneas (recommended). Defaults to True
    :param newApneaAmplitudeDeviations: running list of amplitude deviations observed for apneas, to be appended
        and edited as far as the look-forward searches operate. Leave to None if you don't want this outputted / edited
        (not really needed unless you're using apnea amplitude deviation factors or so). Defaults to None
    :type newApneaAmplitudeDeviations: list of floats, optional
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
    :param averageOnsetAmplitudeInPreviousNeighborhood: The average/composite amplitude on the "baseline" neighborhood
        preceding the current apnea start. This is used as a basline if using apnea amplitude reduction enforcement.
        Needs to be supplied if the latter is used. Defaults to None
    :type averageOnsetAmplitudeInPreviousNeighborhood: float or int
    :return:
        currentApneaBoundaries: new boundaries of the current apnea after potential concatenation with another
        previousApneaBoundaries: boundaries of the previous apnea updated to the current apnea boundaries
            this is needed for code automation reasons.
        newApneaDetections: set of apnea candidates modified after the current apnea is processed, potentially after
            being "closed" concatenated with another anea candidate
    :rtype:
        currentApneaBoundaries: pair-tuple (of floats)
        previousApneaBoundaries: pair-tuple (of floats)
        newApneaDetections: list of pair-tuples (of floats)
    """
    # locating any upcooming breath occurring right after the maximum apnea length from the apnea
    # start + the margin for the minimum breath afterwards,
    nextBreathDetectionsInLongerNeighborhood = siftBreathingEventsForSpecificPastOrFutureNumberAllowance(
        nextBreaths,
        currentApneaBoundaries[0],
        int(
            np.ceil(
                Params.maxApneaLengthInSeconds / Params.maxBreathingDistanceInSeconds,
            ),
        )
        + max(
            Params.noOfBreathstoEnforceAfterApnea,
            Params.noOfBreathsToUseAsApneaAmplitudeReference,
        ),
        pastOrFuture="future",
    )

    i = 0
    currentApneaStart = currentApneaBoundaries[0]

    if enforceApneaAmplitudeReduction or enforceApneaAmplitudeIncrease:
        previouslyFoundApneaOnsetAmplitude = np.inf
    else:
        newApneaAmplitudeDeviations = None

    if enforceApneaAmplitudeReduction:
        previousNextBreathDetectionsInImmediateNeighborhood = np.inf

    apneaClosingBreathStart = None
    while i < len(nextBreathDetectionsInLongerNeighborhood):
        nextBreathCandidateStart = nextBreathDetectionsInLongerNeighborhood[i]["startEnd"][0]

        currentApneaCandidateBoundaries = (
            currentApneaBoundaries[0],
            nextBreathCandidateStart,
        )
        currentCandidateApneaDuration = computeTotalIntervalDurations(
            [currentApneaCandidateBoundaries],
        )

        if (
            currentCandidateApneaDuration <= Params.maxApneaLengthInSeconds
            and currentCandidateApneaDuration >= Params.minApneaLengthInSeconds
        ):
            if enforceApneaAmplitudeIncrease or enforceApneaAmplitudeReduction:
                # need to measure the onset signal amplitude within the apnea after some filtering
                # offset from the start and until before a filtering offset from the end. Because otherwise
                # when measuring the amplitude of silence, the long filter decay will be measured within the
                # silence, making the silence measure an abnormously high amplitude

                innerApneaStart = min(
                    currentApneaStart + Params.minBreathingInhaleExhaleLengthInSeconds,
                    nextBreathCandidateStart,
                )
                innerApneaEnd = max(
                    nextBreathCandidateStart - Params.minBreathingInhaleExhaleLengthInSeconds,
                    currentApneaStart,
                )

                if innerApneaStart > innerApneaEnd:
                    # when takaing the offset is not possible cuz the apnea candidate is too small, re-flip the
                    # order due to the limits above otherwise going out of the apnea candidate interval
                    temp = innerApneaStart
                    innerApneaStart = innerApneaEnd
                    innerApneaEnd = temp
                    del temp

                averageCurrentApneaOnsetAmplitude = computeAverageAbsoluteOnsetAmplitude(
                    breathingOnsetSignal,
                    breathingOnsetSignalTimeAxis,
                    startTime=innerApneaStart,
                    endTime=innerApneaEnd,
                    onsetSignalTimeAxisMinimum=breathingOnsetSignalTimeAxisMinimum,
                    onsetSignalTimeAxisMaximum=breathingOnsetSignalTimeAxisMaximum,
                    onsetSignalTimeAxisIsUniform=breathingOnsetSignalTimeAxisIsUniform,
                )

                if enforceApneaAmplitudeReduction:
                    if previousNextBreathDetectionsInImmediateNeighborhood < Params.noOfBreathstoEnforceAfterApnea:
                        averageCurrentApneaOnsetAmplitudeReducingWrtBefore = True
                    else:
                        averageCurrentApneaOnsetAmplitudeReducingWrtBefore = (
                            averageCurrentApneaOnsetAmplitude <= previouslyFoundApneaOnsetAmplitude
                        )

                    averageApneaAmplitudeReductionWrtPreviousNeighborhood = (
                        averageOnsetAmplitudeInPreviousNeighborhood - averageCurrentApneaOnsetAmplitude
                    ) / averageOnsetAmplitudeInPreviousNeighborhood
                    currentApneaAmplitudeDeviation = averageApneaAmplitudeReductionWrtPreviousNeighborhood

                    amplitudeReductionCheck = (
                        averageApneaAmplitudeReductionWrtPreviousNeighborhood >= Params.apneaBreathRelativeDropThreshold
                        and averageCurrentApneaOnsetAmplitudeReducingWrtBefore
                    )

                    if not averageCurrentApneaOnsetAmplitudeReducingWrtBefore:
                        # should make sure to stop search going too far and re-increasing amplitude after the actual
                        # apnea because otherwise it can erroneously merge two apnea detections into a bigger one
                        # but should not break if the previous closure only failed due to there not being enough
                        # breath after the previous candidate closure -- in that case should merge the previous apnea
                        # candidate with the bigger gap to allow for maintaining the assumption on the minimum
                        # number of breaths after each apnea detection
                        break
                else:
                    amplitudeReductionCheck = True

                if enforceApneaAmplitudeIncrease:
                    averageOnsetAmplitudeInNextNeighborhood = computeOnsetSignalBaselineAmplitude(
                        nextBreathDetectionsInLongerNeighborhood,
                        nextBreathCandidateStart,
                        breathingOnsetSignal,
                        breathingOnsetSignalTimeAxis,
                        pastOrFuture="future",
                        onsetSignalTimeAxisMinimum=breathingOnsetSignalTimeAxisMinimum,
                        onsetSignalTimeAxisMaximum=breathingOnsetSignalTimeAxisMaximum,
                        onsetSignalTimeAxisIsUniform=breathingOnsetSignalTimeAxisIsUniform,
                    )
                    averageApneaAmplitudeIncreaseWrtNextNeighborhood = (
                        averageOnsetAmplitudeInNextNeighborhood - averageCurrentApneaOnsetAmplitude
                    ) / averageOnsetAmplitudeInNextNeighborhood
                    amplitudeIncreaseCheck = (
                        averageApneaAmplitudeIncreaseWrtNextNeighborhood >= Params.apneaBreathRelativeDropThreshold
                    )
                else:
                    amplitudeIncreaseCheck = True

                if enforceApneaAmplitudeReduction:
                    currentApneaAmplitudeDeviation = (
                        currentApneaAmplitudeDeviation + averageApneaAmplitudeIncreaseWrtNextNeighborhood
                    ) / 2
                else:
                    currentApneaAmplitudeDeviation = averageApneaAmplitudeIncreaseWrtNextNeighborhood

            if amplitudeReductionCheck and amplitudeIncreaseCheck:
                nextBreathDetectionsInImmediateNeighborhood = siftBreathingEventsForSpecificPastOrFutureNumberAllowance(
                    nextBreathDetectionsInLongerNeighborhood,
                    nextBreathCandidateStart,
                    Params.noOfBreathstoEnforceAfterApnea,
                    pastOrFuture="future",
                )

                if len(nextBreathDetectionsInImmediateNeighborhood) >= Params.noOfBreathstoEnforceAfterApnea:
                    apneaClosingBreathStart = nextBreathCandidateStart
                    if enforceApneaAmplitudeReduction:
                        previousNextBreathDetectionsInImmediateNeighborhood = np.inf
                        previouslyFoundApneaOnsetAmplitude = averageCurrentApneaOnsetAmplitude
                    lastRetainedApneaAmplitudeDeviation = currentApneaAmplitudeDeviation
                elif enforceApneaAmplitudeReduction:
                    previousNextBreathDetectionsInImmediateNeighborhood = len(
                        nextBreathDetectionsInImmediateNeighborhood,
                    )
                    previouslyFoundApneaOnsetAmplitude = np.inf

        i += 1

    if apneaClosingBreathStart is not None:
        currentApneaBoundaries = (currentApneaBoundaries[0], apneaClosingBreathStart)

        previousApneaBoundaries = currentApneaBoundaries
        newApneaDetections.append(currentApneaBoundaries)
        if enforceApneaAmplitudeIncrease or enforceApneaAmplitudeReduction:
            newApneaAmplitudeDeviations.append(lastRetainedApneaAmplitudeDeviation)

    return (
        currentApneaBoundaries,
        previousApneaBoundaries,
        newApneaDetections,
        newApneaAmplitudeDeviations,
    )


def computeAHIFromDetections(
    apneasAndHypopneaDetections: Iterable,
    totalTime: Union[int, float],
    reFactorAHIAccordingToClusters: bool = False,
    reFactorAHIAccordingToApneaAmplitudeDeviations: bool = False,
    apneasAndHypopneaAmplitudeDeviations=None,
    reFactorAHIAccordingToApneaDurations: bool = False,
):
    """Function used to compute an apnea-hyponea (AHI) or apnea (AI) index
    based on a number of hypopnea/apnea detections

    :param apneasAndHypopneaDetections: apnea and/or hypopnea detections (their times instants /intervals)
    :type apneasAndHypopneaDetections: list, set or 1D numpy array of floats, ints or intervals - anything with a length
    :param totalTime: total time (in seconds) on which the apneas are computed / time basis to use for the AHI/AI
    :type totalTime: float
    :param reFactorAHIAccordingToClusters: True if you want to re-apply weights to apnea detections according to whether
        they fall near other apneas (preference for clusters). Defaults to False
    :type reFactorAHIAccordingToClusters: bool, optional
    :param reFactorAHIAccordingToApneaAmplitudeDeviations: True if you want to re-apply weights to apnea detections
        according to how  much their onset amplide deviates from neighborhood. Defaults to False
    :type reFactorAHIAccordingToApneaAmplitudeDeviations: bool, optional
    :param apneasAndHypopneaAmplitudeDeviations: amplitude deviations observed for each apnea. Supply if you
        want to use apnea amplitude deviation based weighting. Otherwise leave to None. Defaults to None.
    :type apneasAndHypopneaAmplitudeDeviations: list 1D numpy array of floats, optional
    :param reFactorAHIAccordingToApneaDurations: True if you want to re-apply weights to apnea detections
        according to their durations. CAREFUL: Only works if you supply intervals (not individual timeastamps)
        for the apnea detections (for duration computation). Defaults to False
    :type reFactorAHIAccordingToApneaDurations: bool, optional
    :return: apnea-hyponea (AHI) or apnea (AI) index, depending on which detections were supplied
    :rtype: float
    """
    if reFactorAHIAccordingToApneaAmplitudeDeviations:
        apneasAndHypopneaAmplitudeDeviations = np.array(
            apneasAndHypopneaAmplitudeDeviations,
        )

    if len(apneasAndHypopneaDetections) > 0:
        temp = apneasAndHypopneaDetections[0]
        if isinstance(temp, Iterable):
            areApneasIntervals = len(apneasAndHypopneaDetections[0]) == 2
            areApneasTimestamps = len(apneasAndHypopneaDetections[0]) == 1
        else:
            areApneasTimestamps = True
            areApneasIntervals = False

        if (
            reFactorAHIAccordingToClusters
            or reFactorAHIAccordingToApneaAmplitudeDeviations
            or reFactorAHIAccordingToApneaDurations
        ):
            if areApneasIntervals:
                intervalsAreTimeStamps = False
            elif areApneasTimestamps:
                intervalsAreTimeStamps = True

            maxApneaTime = np.max(np.max(np.array(apneasAndHypopneaDetections)))
            apneaIndicatorFs = 0.5
            apneaIndicatorMaxTime = maxApneaTime + 10 * 60  # leaving a 10s margin for the signaling analysis
            apneaIndicatorTimeAxis = np.linspace(
                0,
                apneaIndicatorMaxTime,
                int(np.ceil(apneaIndicatorFs * apneaIndicatorMaxTime)),
                endpoint=False,
            )

            apneaIndicator = computeCenterIndicatorSignalFromIntervals(
                apneasAndHypopneaDetections,
                len(apneaIndicatorTimeAxis),
                apneaIndicatorTimeAxis,
                intervalsAreTimeStamps=intervalsAreTimeStamps,
            )

            weightedApneaIndicator = np.ones(apneaIndicator.shape)
            if reFactorAHIAccordingToClusters:
                apneaClusterFilter = prepareMovingAverageFilter(
                    Params.apneaClusterFactorTimeBasis,
                    1 / apneaIndicatorFs,
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

            if reFactorAHIAccordingToApneaAmplitudeDeviations:
                apneasAndHypopneaAmplitudeDeviations = np.array(
                    apneasAndHypopneaAmplitudeDeviations,
                )
                apneaAmplitudeWeights = (
                    apneasAndHypopneaAmplitudeDeviations / Params.fullWeightedApneaAmplitudeDeviation
                )
                # resetting the weighting for the apnea detections which are likely to be apneas to 1
                # these should not be re-weighted as the AHI definition of 90% reduction
                # suffices -- no O2 signal whose behavior is to be predicted here
                apneaSepcificAmplitudeDeviationFlags = (
                    apneasAndHypopneaAmplitudeDeviations >= Params.apneaDistinguishingAmplitudeDeviation
                )
                apneaAmplitudeWeights[apneaSepcificAmplitudeDeviationFlags] = 1

                amplitudeDeviationWeightedApneaIndicator = computeCenterIndicatorSignalFromIntervals(
                    apneasAndHypopneaDetections,
                    len(apneaIndicatorTimeAxis),
                    apneaIndicatorTimeAxis,
                    intervalsAreTimeStamps=intervalsAreTimeStamps,
                    intervalIndicatorWeights=apneaAmplitudeWeights,
                )
                weightedApneaIndicator = weightedApneaIndicator * amplitudeDeviationWeightedApneaIndicator
                del amplitudeDeviationWeightedApneaIndicator

            if reFactorAHIAccordingToApneaDurations:
                apneaDapneasAndHypopneaDurations = [
                    computeTotalIntervalDurations([i]) for i in apneasAndHypopneaDetections
                ]

                apneaDurationWeights = np.array(apneaDapneasAndHypopneaDurations) / Params.fullWeightedApneaDuration

                if reFactorAHIAccordingToApneaAmplitudeDeviations:
                    # in case amplitude deviations are also used, then neutralize the processing for
                    # apneotic events with amplitude deviations indicative of apneas -- no O2 signal
                    # behavior is relevant according to AASM rules here / apneas always count as 1
                    # inthe definition of the AHI
                    apneaDurationWeights[apneaSepcificAmplitudeDeviationFlags] = 1

                amplitudeDurationWeightedApneaIndicator = computeCenterIndicatorSignalFromIntervals(
                    apneasAndHypopneaDetections,
                    len(apneaIndicatorTimeAxis),
                    apneaIndicatorTimeAxis,
                    intervalsAreTimeStamps=intervalsAreTimeStamps,
                    intervalIndicatorWeights=apneaDurationWeights,
                )
                weightedApneaIndicator = weightedApneaIndicator * amplitudeDurationWeightedApneaIndicator
                del (
                    apneaDapneasAndHypopneaDurations,
                    amplitudeDurationWeightedApneaIndicator,
                )

            del apneaIndicator, apneaIndicatorTimeAxis

            apneaCount = np.sum(weightedApneaIndicator)
            del weightedApneaIndicator

        else:
            apneaCount = len(apneasAndHypopneaDetections)
    else:
        apneaCount = 0

    if totalTime == 0:
        return np.nan
    else:
        return apneaCount / (totalTime / 60 / 60)


def computeOnsetSignalBaselineAmplitude(
    events,
    cutoffTime: float,
    onsetSignal,
    onsetSignalTimeAxis,
    pastOrFuture: str = "past",
    onsetSignalTimeAxisMinimum=None,
    onsetSignalTimeAxisMaximum=None,
    onsetSignalTimeAxisIsUniform=True,
):
    """Function used to compute an onset signal amplitude on a "baseline" or other neighborhood

    :param events: breathing, snoring etc events on the onset signsl (depending on type of signal)
    :type events: list of 2-element lists of pair-tuples (preferably of floats, or of ints)
    :param cutoffTime: cutoff time marking 1 endpont of the neighborhood on which to compute the baseline
    :type cutoffTime: float
    :param onsetSignal: onset signal on which to compute the average amplitudes
    :type onsetSignal: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param onsetSignalTimeAxis: time axis corresponding to the breathing onset signal
    :type onsetSignalTimeAxis: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param pastOrFuture: whether to sift for events / neighborhood before (past) or after (future) the cutoff time,
        the corresponding modes are stores as strings in the variables "past" and
        "future". defaults to "past".
    :type pastOrFuture: str, optional
    :return: average/composite amplitude of the baseline or other neighborhood on the onset signal
    :rtype: float
    """

    siftedBreathingEvents = siftBreathingEventsForSpecificPastOrFutureNumberAllowance(
        events,
        cutoffTime,
        Params.noOfBreathsToUseAsApneaAmplitudeReference,
        pastOrFuture=pastOrFuture,
    )

    if pastOrFuture == "past":
        siftedBreathingEvents = siftedBreathingEvents[
            (
                max(
                    0,
                    len(siftedBreathingEvents) - 1 - Params.noOfBreathsToUseAsApneaAmplitudeReference,
                )
            ) : (len(siftedBreathingEvents))
        ]
    else:
        siftedBreathingEvents = siftedBreathingEvents[
            0 : min(
                len(siftedBreathingEvents),
                Params.noOfBreathsToUseAsApneaAmplitudeReference,
            )
        ]

    startOfBreathingInterval = siftedBreathingEvents[0]["startEnd"][0]
    endOfBreathingInterval = siftedBreathingEvents[-1]["startEnd"][1]

    averageOnsetAmplitudeInSiftedInterval = computeAverageAbsoluteOnsetAmplitude(
        onsetSignal,
        onsetSignalTimeAxis,
        startTime=startOfBreathingInterval,
        endTime=endOfBreathingInterval,
        onsetSignalTimeAxisMinimum=onsetSignalTimeAxisMinimum,
        onsetSignalTimeAxisMaximum=onsetSignalTimeAxisMaximum,
        onsetSignalTimeAxisIsUniform=onsetSignalTimeAxisIsUniform,
    )

    return averageOnsetAmplitudeInSiftedInterval


def computeApneasAndMetricsOnBreathingDetections(
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
    enforeMinApneaDurationInPresifting: bool = True,
    reduceHypopneaChecks: bool = False,
    reFactorAHIAccordingToClusters: bool = True,
    reFactorAHIAccordingToApneaAmplitudeDeviations: bool = False,
    reFactorAHIAccordingToApneaDurations: bool = False,
):
    """Function used to detect apneas/hypopneas and asses AHI / other metrics on pre-detected breathing/snoring events

    :param breathingEventStartEnds: start and end time instants (in seconds) of the breathing/snoring events
    :type breathingEventStartEnds: list of 2-element lists or pair-tuples, or 2D numpy array
        (preferably of floats, or of ints)
    :param breathingEventStrengths: combined/composite strength/salience of each of the events
    :type breathingEventStrengths: list or 1D numpy array (preferably of floats, or of ints)
    :param validBreathingPeriodsStartEnds: time intervals for which the bio-signal can be considered valid. Leave set
        to None if no restriction to valid   periods in the bio-signal is desired. Defaults to None
    :type validBreathingPeriodsStartEnds: list of 2-element lists of pair-tuples, or 2D numpy array
        (preferably of floats, or of ints), optional
    :param invalidBreathingPeriodsStartEnds: time intervals for which the bio-signal is considered invalid. Left to None
        simultaneously with validBreathingPeriodsStartEnds. Defaults to None
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
    :param enforeMinApneaDurationInPresifting: Set to True whenever (for performance reasons) it is desirable
        to already apply the minimum apnea duration constraint to all initial apnea candidates before
        any look-forward tries to concatenate too-short apneas. For most accurate results, this should be left to False,
        but this results in many apnea-checking look-forward searches operating with many overlaps, so for performance
        reasons it defaults to True
    :type enforeMinApneaDurationInPresifting: bool, optional
    :param reduceHypopneaChecks: set to True if you want to reduce the number of hypopnea candidates to check, makes
        the apnea/hypopnea detection algorithm run faster. Recommended be left to False. Defaults to False
    :type reduceHypopneaChecks: bool, optional
    :param reFactorAHIAccordingToClusters: True if you want to re-apply weights to apnea detections according to whether
        they fall near other apneas (preference for clusters). Defaults to True
    :type reFactorAHIAccordingToClusters: bool, optional
    :param reFactorAHIAccordingToApneaAmplitudeDeviations: True if you want to re-apply weights to apnea detections
        according to how  much their onset amplide deviates from neighborhood. Defaults to False
    :type reFactorAHIAccordingToApneaAmplitudeDeviations: bool, optional
    :param reFactorAHIAccordingToApneaDurations: True if you want to re-apply weights to apnea detections
        according to their durations. Defaults to False
    :type reFactorAHIAccordingToApneaDurations: bool, optional
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
        totalSilentInvalidPauses: total time (in s) in which signal was found to be both invalid and also largely silent
        totalNoisyInvalidPauses: total time (in s) in which signal was found to be both invalid
            AND also loud / non-silent / noisy
        AHI: apnea-hypopnea index as estimated based on the number of detected apneas/hypopneas
        apneaToTotalAHITimeBasis: ratio (in %, form 0 to 100) of total time suffering from apnea (totalApneaDuration)
            to the total duration of the "valid" regions in the signal + the total duration of any silent invalid pauses
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
        breathingOnsetSignalTimeAxisMinimum = np.nanmin(breathingOnsetSignalTimeAxis)
        breathingOnsetSignalTimeAxisMaximum = np.nanmax(breathingOnsetSignalTimeAxis)

    logger.info("Detecting apnea and other metrics based on breathing events")

    logger.info("Computing candidate apneas")
    # this needs to be forced to be a sorted list, not in random order, and needs to be based
    # on the same breathing event order as breathingEventStartEnds
    breathingEventGapBoundaries = [
        (breathingEventStartEnds[n][1], breathingEventStartEnds[n + 1][0])
        for n in range(len(breathingEventStartEnds) - 1)
    ]

    if validBreathingPeriodsStartsEnds is not None:
        logger.info(
            "Restricting candidate apneas to those lying in pre-given valid regions",
        )
        # discarding breathing gaps if they have an invalid breathing region inside of them
        # -- to avoid detecting apneas in invalid regions
        breathingEventGapBoundaries = [
            (startTime, endTime)
            for (startTime, endTime) in breathingEventGapBoundaries
            if sum(
                [1 for i in invalidBreathingPeriodsStartsEnds if i[0] >= startTime and i[1] <= endTime],
            )
            == 0
        ]
        validPeriods = validBreathingPeriodsStartsEnds
    else:
        logger.info(
            "Proceeding without restricting candidate apneas to those lying in pre-given valid regions",
        )
        validPeriods = None

    breathingEventGapDurations = [(i[1] - i[0]) for i in breathingEventGapBoundaries]

    if invalidBreathingPeriodsStartsEnds is not None:
        invalidPeriods = invalidBreathingPeriodsStartsEnds
    if invalidBreathingPeriodsStartsEnds is None:
        logger.info(
            "Detecting rudimentary invalid regions based on those with long breathing gaps",
        )
        # computing total of periods (in s) where no breaths are detected for long
        # -- these are additionally re-flagged as invalid

        invalidPeriods = [
            breathingEventGapBoundaries[n]
            for n, i in enumerate(breathingEventGapDurations)
            if i > Params.maxApneaLengthInSeconds
        ]

    logger.info("Computing total span/duration of invalid intervals")
    totalInvalidPeriods = computeTotalIntervalDurations(invalidPeriods)
    if validBreathingPeriodsStartsEnds is not None:
        # checking which silent pauses fall under invalid regions
        silentInvalidPauses = siftIntervalsForThoseOverlappingAnyOtherInterval(
            silentPauses,
            invalidPeriods,
        )
    else:
        silentInvalidPauses = []

    # computing total of periods (in s) where enough breaths / periodic breathing are detected -- these are valid
    if validBreathingPeriodsStartsEnds is not None:
        logger.info("Computing total span/duration of valid intervals")
        totalValidPeriods = computeTotalIntervalDurations(validPeriods)
        logger.info("Computing total span/duration of silent invalid pause intervals")
        totalSilentInvalidPauses = computeTotalIntervalDurations(silentInvalidPauses)
    else:
        totalValidPeriods = (
            breathingOnsetSignalTimeAxis[-1] - breathingOnsetSignalTimeAxis[0]  # total signal duration
        ) - totalInvalidPeriods  # minus ad-hoc detected invalid time
        totalSilentInvalidPauses = 0

    totalNoisyInvalidPauses = totalInvalidPeriods - totalSilentInvalidPauses

    nonEmptySignal = totalValidPeriods > 0 and len(breathingEventStartEnds) > 0

    if nonEmptySignal:
        # it should check everything if it's desirable to detect both hypopneas and apneas,
        # but it can also explicitly exclude most hypopneas by pre-sifting the candidates for
        # those with a minimum duration
        if enforeMinApneaDurationInPresifting:
            minApneaLimitForPresifting = 0.3 * Params.minApneaLengthInSeconds
        else:
            minApneaLimitForPresifting = 0

        logger.info("Enforcing (some of) basic apnea assumptions on apnea candidates")
        apneaStartEnds = [
            breathingEventGapBoundaries[n]
            for n, i in enumerate(breathingEventGapDurations)
            if minApneaLimitForPresifting <= breathingEventGapDurations[n]
            and breathingEventGapDurations[n] <= Params.maxApneaLengthInSeconds
        ]
        del breathingEventGapDurations

        if (not enforeMinApneaDurationInPresifting) and reduceHypopneaChecks:
            hypopneaTriggerIndices = [
                i
                for i, _ in enumerate(apneaStartEnds)
                if computeTotalIntervalDurations([apneaStartEnds[i]]) <= Params.minApneaLengthInSeconds
            ]
            apneaTriggerIndices = [i for i, _ in enumerate(apneaStartEnds) if i not in hypopneaTriggerIndices]
            hypopneaTriggerIndices = hypopneaTriggerIndices[::3]

            apneaOrHypopneaTriggerIndices = sorted(
                list(set(apneaTriggerIndices).union(set(hypopneaTriggerIndices))),
            )

            del apneaTriggerIndices, hypopneaTriggerIndices

            apneaStartEnds = [
                apneaStartEnds[i] for i, _ in enumerate(apneaStartEnds) if i in apneaOrHypopneaTriggerIndices
            ]

            del apneaOrHypopneaTriggerIndices

        logger.info(
            "Enforcing minimum number of breathing/snoring events before and\
        after each apnea, and relative amplitude reduction",
        )
        newApneaStartEnds = []
        if enforceApneaAmplitudeReduction or enforceApneaAmplitudeIncrease:
            apneaAmplitudeDeviations = []
        else:
            apneaAmplitudeDeviations = None

        latestApneaBoundaries = (0, 0)
        for currentApneaBoundaries in apneaStartEnds:
            if not currentApneaBoundaries[1] <= latestApneaBoundaries[1]:
                # checking for each apnea, how many breaths are detected in the maximal seconds taken by
                # the defined Params.noOfPreviousBreathstoEnforceBeforeApnea, the duration is computed based
                #  on the maximal time those would take. Restricting these "previous" breaths to those occuring
                # after the previous apnea detection too
                previousBreathDetectionsSinceLastApnea = [
                    {
                        "startEnd": breathingEventStartEnd,
                        "strength": breathingEventStrengths[k],
                    }
                    for k, breathingEventStartEnd in enumerate(breathingEventStartEnds)
                    if breathingEventStartEnd[0] <= currentApneaBoundaries[0]
                    and breathingEventStartEnd[0] >= latestApneaBoundaries[1]
                ]
                # since the apneas start and end at the starts of the inhales, comparing them
                # against the START of breathing events here

                previousBreathDetectionsInImmediateNeighborhood = (
                    siftBreathingEventsForSpecificPastOrFutureNumberAllowance(
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
                    ) = retainTimesWithinPredefinedWindows(
                        [currentApneaBoundaries[0]],
                        validBreathingPeriodsStartsEnds,
                    )

                    # assuming the apnea start -- e.g. the last breath end before it --
                    # can only occur in one valid region since they're distinct
                    whichValidRegionsApneaBelongsTo = whichValidRegionsApneaBelongsTo[0][0]

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
                        averageOnsetAmplitudeInPreviousNeighborhood = computeOnsetSignalBaselineAmplitude(
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
                    ) = lookForwardForApneaClosureWithEnoughBreaths(
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

        apneaStartEnds = newApneaStartEnds
        del newApneaStartEnds

        apneaDurations = [i[1] - i[0] for i in apneaStartEnds]
        logger.info(
            "Finished enforcing all apnea assumptions on candidates",
        )

        totalApneaDuration = sum(apneaDurations)
        apneaDetections = [((apneaBoundaries[0] + apneaBoundaries[1]) / 2) for apneaBoundaries in apneaStartEnds]

        if len(apneaStartEnds) != 0:
            logger.info(
                "Detected a total of " + str(len(apneaStartEnds)) + " apneotic events",
            )
        else:
            logger.info("Could not detect any apneotic events")

    if nonEmptySignal:
        totalAHITimeBasis = totalValidPeriods + totalSilentInvalidPauses
        usable_by_algorithm = bool(totalAHITimeBasis >= Params.minimumAHITimeBasis)

        AHI = computeAHIFromDetections(
            apneaStartEnds,
            totalAHITimeBasis,
            reFactorAHIAccordingToClusters=reFactorAHIAccordingToClusters,
            reFactorAHIAccordingToApneaAmplitudeDeviations=reFactorAHIAccordingToApneaAmplitudeDeviations,
            apneasAndHypopneaAmplitudeDeviations=apneaAmplitudeDeviations,
            reFactorAHIAccordingToApneaDurations=reFactorAHIAccordingToApneaDurations,
        )
        apneaToTotalAHITimeBasis = 100 * totalApneaDuration / totalAHITimeBasis
    else:
        apneaDetections = []
        apneaStartEnds = []
        apneaDurations = []
        totalApneaDuration = -1
        AHI = -1
        apneaToTotalAHITimeBasis = -1
        usable_by_algorithm = False

    return (
        apneaDetections,
        apneaStartEnds,
        apneaAmplitudeDeviations,
        apneaDurations,
        totalApneaDuration,
        invalidPeriods,
        totalInvalidPeriods,
        validPeriods,
        totalValidPeriods,
        silentInvalidPauses,
        totalSilentInvalidPauses,
        totalNoisyInvalidPauses,
        AHI,
        usable_by_algorithm,
        apneaToTotalAHITimeBasis,
    )


def remove_constant_freq_bins(stft_array, threshold=0.01):
    """
    Set constant frequency bins in the STFT array to zero.

    Parameters:
    - stft_array: numpy array, shape (num_frequency_bins, num_time_frames)
      The input STFT array.
    - threshold: float, optional
      The threshold below which the standard deviation is considered low.

    Returns:
    - stft_array_modified: numpy array, shape (num_frequency_bins, num_time_frames)
      The modified STFT array with constant frequency bins set to zero.
    """
    # Calculate the standard deviation along the time axis (axis=1)
    std_deviation = np.nanstd(stft_array, axis=1)

    # Identify frequency bins with low standard deviation
    constant_freq_bins = np.where(std_deviation < threshold)[0]

    # Create a boolean mask where True indicates the rows to keep
    stft_array_modified = np.copy(stft_array)
    mask = np.ones(stft_array_modified.shape[0], dtype=bool)
    mask[constant_freq_bins] = False

    # Use boolean indexing to select rows that satisfy the mask
    stft_array_modified = stft_array_modified[mask]

    return stft_array_modified


def filterSTFTForCustomOnsets(
    STFTtoFilter: np.ndarray,
    sample_rate: int,
    customOnsetActiveDurationInSeconds: float,
    customOnsetPassiveDurationInSeconds: float = None,
    spectrum_hop_length: int = Params.spectrum_hop_length_breathing,
    n_fft: int = Params.n_fft,
    gradualOnsetFlag: bool = False,
    halfWaveRectificationFlag: bool = True,
    frequencyLossFlag: bool = False,
):
    """Function used to filter an STFT magnitude across time (and frequencies potentially) for onset signal extraction

    essentially applies image filtering (mostly smoothed derivatives) on the STFT magnitude coefficients
    from sleep recordings for emphasizing the onsets of breathing, impulsise noise, snoring and other types of events

    :param STFTtoFilter: the STFT magnitude coefficients on which to do the filtering.  Given in LINEAR decibel scale
        as it's changed to decibel scale hereafter
    :type STFTtoFilter: 2D numpy array (of floats)
    :param sample_rate: sample rate/frequency in Hz of the original audio used to compute the STFT
    :type sample_rate: int, optional
    :param customOnsetActiveDurationInSeconds: the custom active duration paratemer in seconds passed on to the
        image filter creation function.
    :type customOnsetActiveDurationInSeconds: float
    :param customOnsetPassiveDurationInSeconds: the custom passive duration paratemer in seconds passed on to the
        image filter creation function. Leave to None to simply re-use the same duration as
        customOnsetActiveDurationInSeconds. Defaults to None
    :type customActiveDurationInSeconds: float, optional
    :param spectrum_hop_length: number of samples (step) by which to move each window, taken at the reference
        sample rate (SAMPLE_RATE)), defaults to Params.spectrum_hop_length_breathing
    :type spectrum_hop_length: int, optional
    :param n_fft: frame/window size in samples at the reference sample rate (SAMPLE_RATE), defaults to Params.n_fft
    :type n_fft: int, optional
    :param gradualOnsetFlag: Gradual onset detection parameter passed on to the image filter creation function.
        Defaults to False
    :type gradualOnsetFlag: bool, optional
    :param halfWaveRectificationFlag: Set to True to apply half-wave rectification to the resulting onset signal.
        Half-wave rectification means cutting out the negative part and leaving only the positive part of the signal.
        Not recommended for onset types that should have two-sided waveforms (e.g. breathing flow etc)
        Set to False if you want to keep the whole signal. Defaults to True
    :type halfWaveRectificationFlag: bool, optional
    :param frequencyLossFlag: frequency loss detection parameter passed on to the image filter creation function.
        Defaults to False
    :type frequencyLossFlag: bool, optional
    :return: the onset signal resulting from fitlering the STFT magnitudes after applying the filtering
    :rtype: 1D numpy array (of floats)
    """

    logger.info("Processing and filtering STFT for custom onsets")

    customOnsetFilter = computeSTFTCustomOnsetFilter(
        sample_rate,
        customOnsetActiveDurationInSeconds,
        minPassiveDurationBetweenEventsInSeconds=customOnsetPassiveDurationInSeconds,
        zeroInCenter=False,
        frequencyDomainAveraging=frequencyLossFlag,
        spectrum_hop_length=spectrum_hop_length,
        n_fft=n_fft,
        gradualOnsetFlag=gradualOnsetFlag,
        gradualFrequencyLossFlag=frequencyLossFlag,
    )

    filterDimensions = customOnsetFilter.shape
    is1DFilter = filterDimensions[0] == 1

    customFilterHalfWidth = round(filterDimensions[1] / 2)

    # Clean constant noise from STFT
    STFTtoFilter = remove_constant_freq_bins(STFTtoFilter, threshold=4.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore runtime warnings about empty slices
        if is1DFilter:
            # in this case it's actually a 1D filter embedded in a 2D filter, can do the filtering on the
            # other dimension before applying the filter on its own relevant dimension -- since the filtering
            # on the other dimension is averaging, this allows for drastically saving memory usage

            filteredSTFT = np.nanmean(
                STFTtoFilter,
                axis=0,
            )  # averaging over the frequency range

            customOnsetSignal = sigAPI.convolve(
                filteredSTFT,
                customOnsetFilter[0, :].squeeze(),
                mode="same",
            )  # filtering across the time-dimension

        else:
            filteredSTFT = sigAPI.convolve2d(
                STFTtoFilter,
                customOnsetFilter,
                mode="same",
            )  # filtering across the time-dimension
            customOnsetSignal = np.nanmean(
                filteredSTFT,
                axis=0,
            )  # averaging over the frequency range

        del filteredSTFT
        gc.collect()

    if halfWaveRectificationFlag:
        logger.info("Performing half-wave rectification on filtered image")
        customOnsetSignal[
            customOnsetSignal < 0
        ] = 0  # if needed, isolate the onsets of the events (avoiding output at the end of the event)
        logger.info(
            "Finished performing half-wave rectification on filtered image",
        )

    noOfSTFTBins = len(customOnsetSignal)
    # these periods are ill-defined -- but keeping them in the result for axis purposes
    customOnsetSignal[0:customFilterHalfWidth] = np.nan
    customOnsetSignal[(noOfSTFTBins - customFilterHalfWidth) : noOfSTFTBins] = np.nan

    return customOnsetSignal


def computeSTFTCustomOnsetFilter(
    sample_rate: float,
    minEventActiveDurationInSeconds: float,
    minPassiveDurationBetweenEventsInSeconds: float = None,
    zeroInCenter: bool = True,
    frequencyDomainAveraging: bool = True,
    spectrum_hop_length: int = Params.spectrum_hop_length_breathing,
    n_fft: int = Params.n_fft,
    gradualOnsetFlag: bool = False,
    gradualFrequencyLossFlag: bool = False,
):
    """Function for designing image filters for extracting breathing, snoring and other onsets on STFT magnitude images

    the filter is basically designed as some sort of variant on a smoothed-derivative filter, designed
    to make the onsets salient than on the original audio waveform yielding the STFT.

    :param sample_rate: sample rate/frequency in Hz of the audio signal on which the STFT was computed
    :type sample_rate: int, optional
    :param minEventActiveDurationInSeconds: Duration paratemer (in seconds) which sets the temporal duration of the
        filter on the STFT. This is is usually chosen as a minimum value, it corresponds to the amount of
        smoothing/smudging/averaging that the filtering is allowed to do.
    :type minEventActiveDurationInSeconds: float
    :param minPassiveDurationBetweenEventsInSeconds: similar to minEventActiveDurationInSeconds, but corresponding to
        the duration of silence that can be expected before the start of each event whose onset is to be detected.
        For example this can be the minimum duration in seconds between different breaths (exhale to inhale).
        Leave to None to simply re-use the same duration as minEventActiveDurationInSeconds.
        Defaults to None.
    :type minPassiveDurationBetweenEventsInSeconds: float, optional
    :param zeroInCenter: Set to true if you want to make the filter symmetric around its center by introducing
        a zero value/column in the middle. Defaults to True
    :type zeroInCenter: bool, optional
    :param frequencyDomainAveraging: Set to true if you want the filter to average the STFT magnitudes over the
        frequency axis. Not usually needed / redundant if you're anyway averagint the STFT after filtering over
        the frequency axis. Defaults to True
    :type frequencyDomainAveraging: bool, optional
    :param spectrum_hop_length: the hop_length parameter with which the STFT to be filtered computed.
        Defaults to Params.spectrum_hop_length_breathing
    :type spectrum_hop_length: int, optional
    :param n_fft: the n_fft parameter with which the STFT to be filtered is computed. Defaults to Params.n_fft
    :type n_fft: int, optional
    :param gradualOnsetFlag: Set to true if you want the filter to emphasize events which gradually fade in,
        for example breathing events, as opposed to events which suddently begin (like impulsive noise or snoring)
        -- for those set it to False. Defaults to False
    :type gradualOnsetFlag: bool, optional
    :param gradualFrequencyLossFlag: Set to True if you want the filter to apply a gradual-loss detection over
        the frequency-axis of the STFT. This attempts to emphasize events which feature a loss -- on average --
        of STFT magnitude as the frequency increases. Defaults to False
    :type gradualFrequencyLossFlag: bool, optional
    :return: image filter to be used for filtering STFT magnitudes according to the desired functioning/parameters
    :rtype: 2D numpy array (of floats)
    """
    logger.info("Preparing custom (image) filter to be used on STFT")
    # used to map the STFT parameters from those used at 16 kHz
    n_fft = int(np.ceil(n_fft * sample_rate / Params.SAMPLE_RATE_STFT))
    spectrum_hop_length = int(
        np.ceil(spectrum_hop_length * sample_rate / Params.SAMPLE_RATE_STFT),
    )

    if frequencyDomainAveraging:
        frequncyBandwidthToAverage = Params.defaultOnsetFrequencyBandwidthToAverage
        noOfFreqBinsToAverage = int(
            round(frequncyBandwidthToAverage / (sample_rate / n_fft)),
        )
    else:
        noOfFreqBinsToAverage = 1

    # need to use the hop_length not n_fft parameter cuz the hop length is what determines the number of frames
    # corresponding to a specific time span on an STFT, not the n_fft
    minEventLengthInSTFTFrames = int(
        round(minEventActiveDurationInSeconds * sample_rate / spectrum_hop_length),
    )

    if not gradualOnsetFlag:
        activeFilterRegion = np.ones(
            (noOfFreqBinsToAverage, minEventLengthInSTFTFrames),
        )

        if minPassiveDurationBetweenEventsInSeconds is None:
            passiveFilterRegion = activeFilterRegion
        else:
            minPauseEventLengthInSTFTFrames = int(
                round(
                    minPassiveDurationBetweenEventsInSeconds * sample_rate / spectrum_hop_length,
                ),
            )
            passiveFilterRegion = np.ones((1, minPauseEventLengthInSTFTFrames))
            passiveFilterRegion = (
                passiveFilterRegion
                * np.sum(activeFilterRegion.reshape(activeFilterRegion.size))
                / np.sum(passiveFilterRegion.reshape(passiveFilterRegion.size))
            )

        if not zeroInCenter:
            imageFilter = np.concatenate(
                [activeFilterRegion, -passiveFilterRegion],
                axis=1,
            )
        else:
            imageFilter = np.concatenate(
                [
                    activeFilterRegion,
                    np.zeros((noOfFreqBinsToAverage, 1)),
                    -passiveFilterRegion,
                ],
                axis=1,
            )

    else:
        imageFilter = np.linspace(-6, 6, num=int(round(2 * minEventLengthInSTFTFrames)))
        imageFilter = 2 * (1 - 1 / (1 + np.exp(-imageFilter))) - 1
        imageFilter = np.expand_dims(imageFilter, axis=1).T

    if gradualFrequencyLossFlag:
        # assumes the lower rows in the filter correspond to the higher frequencies in the STFT
        noOfFrequenciesToTakeasPositive = round(noOfFreqBinsToAverage / 2)
        frequencyDecayFilter = np.ones((noOfFreqBinsToAverage, 1))
        frequencyDecayFilter[0:(noOfFrequenciesToTakeasPositive)] = -1
        imageFilter = sigAPI.convolve2d(frequencyDecayFilter, imageFilter, mode="full")

    return imageFilter


def extractCustomOnsets(
    inputAudioSignal,
    sample_rate: int,
    STFT_n_fft: dict[int] = {
        "breathing": Params.n_fft,
        "snoring": Params.n_fft_snoring,
        "snoring_low": Params.n_fft_snoring,
    },
    STFT_hop_length: dict[int] = {
        "breathing": Params.spectrum_hop_length_breathing,
        "snoring": Params.spectrum_hop_length_snoring,
        "snoring_low": Params.spectrum_hop_length_snoring,
    },
    filterActiveDurationsToUse: dict[float] = {
        "breathing": Params.minBreathingInhaleExhaleLengthInSeconds,
        "snoring": Params.minBreathingInhaleExhaleLengthInSeconds,
        "snoring_low": Params.minBreathingInhaleExhaleLengthInSeconds,
    },
    filterPassiveDurationsToUse: dict[float] = {
        "breathing": Params.breathingPassiveDuratonToUse,
        "snoring": Params.snoringPassiveDuratonToUse,
        "snoring_low": Params.snoringPassiveDuratonToUse,
    },
    filterGradualOnsetFlags: dict[bool] = {
        "breathing": Params.breathingGradualOnsetUse,
        "snoring": Params.snoringGradualOnsetUse,
        "snoring_low": Params.snoringGradualOnsetUse,
    },
    filterHWRectificationFlags: dict[bool] = {
        "breathing": Params.breathingHWRectificationUse,
        "snoring": Params.snoringHWRectificationUse,
        "snoring_low": Params.snoringHWRectificationUse,
    },
    filterFrequencyLossFlags: dict[bool] = {
        "breathing": Params.breathingFrequencyLossDetection,
        "snoring": Params.snoringFrequencyLossDetection,
        "snoring_low": Params.snoringFrequencyLossDetection,
    },
    STFToutlierSiftingPercentiles: dict[float] = {
        "breathing": Params.breathingOnsetSiftingPercentile,
        "snoring": Params.snoringOnsetSiftingPercentile,
        "snoring_low": Params.snoringOnsetSiftingPercentile,
    },
    filterFreqRangeBoundaries: dict[tuple[float]] = {
        "breathing": (
            Params.lowPassCutoffFreqChebySnorefox,
            Params.highPassCuttoffFreqChebySnorefox,
        ),
        "snoring": (
            Params.snoringLowPassCuttoffFreqCheby,
            Params.snoringHighPassCuttoffFreqCheby,
        ),
        "snoring_low": (
            Params.snoringLowPassCuttoffFreqChebyLow,
            Params.snoringHighPassCuttoffFreqChebyLow,
        ),
    },
):
    """Function used to extract onset signals (breathing, snoring, impulsive noise etc) from an audio signal

    An "onset" signal is essentially a heavily-processed version of another signal, which tries to focus on the "onsets"
    (beginnings) of specific types of events (breathing events, snoring events). The consept of "onset" is generalized
    here to also mean "negative onsets" -- essentially the onset of the *end* of an event (end of breathing exhale). One
    good example of an onset signal this is able to compute from the audio is a "breathing onset signsal" -- a sort
    of double-sided breathing flow signal artificially-recreated from the audio.
    The function is actually generic: you can make it prepare / compute a single onset signal, two, three, a dozen.
    The parameters for the different onset signals are provided in dictionaries, and are stored under the same keys.
    The function launches the STFT computation, filters the STFT, and generates the onset signals, all
    done with the relevant parameters as provided in the dictionaries.

    :param inputAudioSignal: input audio or other 1D signal
    :type inputAudioSignal: list or 1D numpy array (preferably of floats, or of ints),
        or generator (stream) returning those
    :param sample_rate: sample rate/frequency in Hz
    :type sample_rate: int
    :param STFT_n_fft: n_fft parameters to pass on to the STFT computation for each onset signal.
        Defaults to {"breathing": Params.n_fft,
                     "snoring": Params.n_fft_snoring
                    },
    :type STFT_n_fft: dict[int], optional
    :param STFT_hop_length: hop_length parameters to pass on to the STFT computation for each onset signal.
        Defaults to { "breathing": Params.spectrum_hop_length_breathing,
                      "snoring": Params.spectrum_hop_length_snoring}
    :type STFT_hop_length: dict[int], optional
    :param filterActiveDurationsToUse: The active duration parameters when filtering STFTs for each onset signal.
        Defaults to { "breathing": Params.minBreathingInhaleExhaleLengthInSeconds,
                      "snoring": Params.minBreathingInhaleExhaleLengthInSeconds}
    :type filterActiveDurationsToUse: dict[float], optional
    :param filterPassiveDurationsToUse: The passive duration parameters (if any) when filtering STFTs for each onset
        signal. Defaults to { "breathing": Params.breathingPassiveDuratonToUse,
                              "snoring": Params.snoringPassiveDuratonToUse}
    :type filterPassiveDurationsToUse: dict[float], optional
    :param filterGradualOnsetFlags: Flags setting whether to look for gradual or abrupt onsets for each signal.
        Defaults to { "breathing": Params.breathingGradualOnsetUse,
                      "snoring": Params.snoringGradualOnsetUse}
    :type filterGradualOnsetFlags: dict[bool], optional
    :param filterHWRectificationFlags: Flags setting whether to apply half-wave rectification for each onset signal.
        Defaults to { "breathing": Params.breathingHWRectificationUse,
                      "snoring": Params.snoringHWRectificationUse}
    :type filterHWRectificationFlags: dict[bool], optional
    :param filterFrequencyLossFlags: Flags setting whether to look for frequency-loss for each onset signal.
        Defaults to { "breathing": Params.breathingFrequencyLossDetection,
                      "snoring": Params.snoringFrequencyLossDetection, }
    :type filterFrequencyLossFlags: dict[bool], optional
    :param STFToutlierSiftingPercentiles: Percentile parameters for outlier sifting to pass on to the STFT filtering
        It's a value (between 0 and 100) useful for discarding high-amplitude outliers from the STFT,
        and focus more on low-volume sounds (such as breathing).
        Defaults to { "breathing": Params.breathingOnsetSiftingPercentile,
                                "snoring": Params.snoringOnsetSiftingPercentile}
    :type STFToutlierSiftingPercentiles: dict[float], optional
    :param filterFreqRangeBoundaries: frequency range to focus in the STFT for each onset signal, i.e.,
        the minimum and maximum frequencies (in Hz) for which to trim the STFT (only keeping bins within these).
        Defaults to {"breathing": (Params.lowPassCutoffFreqChebySnorefox,
                                                     Params.highPassCuttoffFreqChebySnorefox,
                                                    ),
                     "snoring": (Params.snoringLowPassCuttoffFreqCheby,
                                                   Params.snoringHighPassCuttoffFreqCheby,
                                                    ),
                    },
    :type filterFreqRangeBoundaries: dict[tuple[float]], optional
    :return:
        customOnsetSignals: onset signals for each type of parameter combination (dictionary key)
        customOnsetSignalTimeAxesInSeconds: time axes corresponding to the onset signals
    :rtype:
        customOnsetSignals: dictionary of 1D numpy arrays (of floats), with the same structure as the dictionaries given
            to the function in arguments
        customOnsetSignalTimeAxesInSeconds: dictionary of 1D numpy arrays (of floats), with the same structure as the
            dictionaries given to the function in arguments
    """
    customOnsetSignals = {}
    customOnsetSignalTimeAxesInSeconds = {}

    logger.info("Extracting custom onset signals from audio")

    i = 0
    for (
        (onsetType, onsetLengthInSeconds),
        (_, STFT_n_fft),
        (_, STFT_hop_length),
        (_, pauseLengthInSeconds),
        (_, onsetGradualFlag),
        (_, onsetHWRectificationFlag),
        (_, onsetFreqRangeBoundaries),
        (_, onsetFreqLossFlag),
        (_, STFToutlierSiftingPercentile),
    ) in zip(
        filterActiveDurationsToUse.items(),
        STFT_n_fft.items(),
        STFT_hop_length.items(),
        filterPassiveDurationsToUse.items(),
        filterGradualOnsetFlags.items(),
        filterHWRectificationFlags.items(),
        filterFreqRangeBoundaries.items(),
        filterFrequencyLossFlags.items(),
        STFToutlierSiftingPercentiles.items(),
    ):
        if callable(inputAudioSignal):
            logger.info("Computing " + str(onsetType) + " onsets using streamed file")

        else:
            logger.info("Computing " + str(onsetType) + " onsets using fully-read file")

        (
            currentSTFT,
            _,
            customOnsetSignalTimeAxisInSeconds,
            STFTFreqAxis,
        ) = compute_STFT(  # computing the STFT with NO overlap, to not not waste memory
            inputAudioSignal,
            n_fft=STFT_n_fft,
            hop_length=STFT_hop_length,
            sample_rate=sample_rate,
            computeSTFTPhaseFlag=False,
        )

        logger.info("Discarding STFT frequency bins outside relevant frequency range")
        frequencyRangeOfInterest = [
            n
            for n, i in enumerate(STFTFreqAxis)
            if i >= onsetFreqRangeBoundaries[1] and i <= onsetFreqRangeBoundaries[0]
        ]
        # finding the start of the frequency bin range to keep on the STFT

        # better use min-to-max slice indexing if switching to more efficient clipping later
        # lept like this because it achieves a better memory profile with np.clip()
        currentSTFT = currentSTFT[
            frequencyRangeOfInterest,
            :,
        ]  # only keeping a sub-STFT -- other parts of the spectrum are useless here

        logger.info(
            "Detecting and removing outliers in STFT using percentile computation",
        )
        # sifting out outliers to not overstress the decibel-scale dynamic range of the STFT

        siftedMaxSTFTAmp = np.percentile(
            currentSTFT.reshape(
                currentSTFT.size,
            ),  # flattening the array without creating an array copy
            STFToutlierSiftingPercentile,
            axis=0,
        )

        # sifting out / capping the outliers
        currentSTFT = np.clip(currentSTFT, a_min=None, a_max=siftedMaxSTFTAmp)

        logger.info(
            "Finished detecting and removing outliers in STFT using percentile computation",
        )

        currentSTFT = mag2db(
            currentSTFT,
            threshold=Params.STFT_cutoff_dB_threshold,
            renormalizeFlag=False,
        )

        customOnsetSignal = filterSTFTForCustomOnsets(
            currentSTFT,
            sample_rate,
            onsetLengthInSeconds,
            customOnsetPassiveDurationInSeconds=pauseLengthInSeconds,
            spectrum_hop_length=STFT_hop_length,
            n_fft=STFT_n_fft,
            gradualOnsetFlag=onsetGradualFlag,
            halfWaveRectificationFlag=onsetHWRectificationFlag,
            frequencyLossFlag=onsetFreqLossFlag,
        )

        del currentSTFT
        gc.collect()

        customOnsetSignals[onsetType] = customOnsetSignal
        customOnsetSignalTimeAxesInSeconds[onsetType] = customOnsetSignalTimeAxisInSeconds

        logger.info(
            "Finished computing " + str(onsetType) + " onsets",
        )
        del customOnsetSignal
        gc.collect()

        i += 1

    gc.collect()

    return customOnsetSignals, customOnsetSignalTimeAxesInSeconds


def create_onset_from_events(times, strengths, STFTTimeAxisAll):
    """
    Given a set of events with their timestamps (start and end times) and strengths, generate a continuous
    onset signal.

    Parameters:
        times (np.ndarray): timestamps of events
        strengths (np.ndarray): strength of the signal for each timestamp (1 timestamp has start and end value)
        STFTTimeAxisAll (list): time axis corresponding to the breathing and snoring onset signal

    Returns:
        np.ndarray: amplitude of the onset signal
        np.ndarray: times of the onset signal
    """

    modified_array = []
    amplitudes = []

    # for i, sub_array in enumerate(times):
    for i in range(0, len(times)):
        # Calculate the difference between the two values in the sub-array
        difference = (times[i][1] - times[i][0]) / 10

        amplitud = 2000
        if len(strengths) > 0:
            amplitud = strengths[i]

        # Add values at the beginning and end of the sub-array
        if (i < (len(times) - 1) and ((times[i + 1][0] - times[i][1]) > 40.0)) or (
            i >= 0 and ((times[i][0] - times[i - 1][1]) > 40.0)
        ):
            modified_sub_array = np.insert(times[i], 0, times[i][0] - difference)
            modified_sub_array = np.append(modified_sub_array, times[i][1] + difference)
            amplitudes_sub_array = [0, amplitud, -amplitud, 0]
        else:
            modified_sub_array = times[i]
            amplitudes_sub_array = [amplitud, -amplitud]

        # Assign the modified sub-array to the modified array
        modified_array.append(modified_sub_array)
        amplitudes.append(amplitudes_sub_array)

    times = np.concatenate(modified_array, axis=0)
    onsets = np.concatenate(amplitudes, axis=0)
    times = np.insert(times, 0, STFTTimeAxisAll[0])
    onsets = np.insert(onsets, 0, 0)

    return onsets, times


def update_length_and_interpolate(onsets, times, length):
    """
    Update an onset signal with its amplitude and timestamps to a given length by interpolating its values.

    Parameters:
        onsets (np.ndarray): amplitude of the onset signal
        times (np.ndarray): times of the onset signal
        length (int): desired length to update the onset signal

    Returns:
        np.ndarray: amplitude of the onset signal
        np.ndarray: times of the onset signal
    """

    # Check for Non-increasing sequence and handle consecutive duplicates
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            low_val = times[i]
            high_val = times[i - 1]

            # Replace the repeated value with the average of the neighbors
            times[i] = 0.5 * (low_val + high_val)

    # Check for consecutive duplicate values after updating
    zero_diff_indices = np.where(np.diff(times) == 0)[0]
    for i in zero_diff_indices:
        if (len(times) - 1) > i > 0:
            times[i] = (times[i - 1] + times[i + 1]) * 0.5

    # For interpolation `x` must be strictly increasing sequence.
    # This fixes possible floating errors
    times = sorted(times)

    # Update length and interpolate
    times_new = np.linspace(times[0], times[-1], length)
    interpolator = scipy.interpolate.PchipInterpolator(times, onsets)
    onsets_new = interpolator(times_new)

    return onsets_new, times_new


def detect_valid_breathing_regions(
    STFTTimeAxisAll,
    allCurrentBreathingOnsets,
    STFTTimeAxisBreathing,
    allCurrentSnoringOnsets,
    STFTTimeAxisSnoring,
    offsetToStartAt,
):
    """Function that analyzes the breathing and snoring onset signals for time intervals where there's a valid form of
    breathing. This function combines the old invalidity detector logic (detectValidBreathingRegionsInOnsetSignal())
    and a new analysis of onset signals generated from all breathing and snoring events.

    :param STFTTimeAxisAll: time axis corresponding to the breathing and snoring onset signal
    :type STFTTimeAxisAll: list, numpy array or similar (of floats)

    :param allCurrentBreathingOnsets: breathing onset bio-signal (or a sort of DC-corrected breathing flow signal)
    :type allCurrentBreathingOnsets: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param STFTTimeAxisBreathing: time axis corresponding to the breathing onset signal
    :type STFTTimeAxisBreathing: list, numpy array or similar (of floats)

    :param allCurrentSnoringOnsets: snoring onset bio-signal (or a sort of DC-corrected breathing flow signal)
    :type allCurrentSnoringOnsets: list, 1D numpy array or similar (preferably of floats, or of ints)
    :param STFTTimeAxisSnoring: time axis corresponding to the snoring onset signal
    :type STFTTimeAxisSnoring: list, numpy array or similar (of floats)

    :return:
        validBreathingPeriodsStartEnds: time intervals for which the bio-signal can be considered valid
        invalidBreathingPeriodsStartEnds: time intervals for which the bio-signal is considered invalid.
            These represent the time-complement of the intervals in validBreathingPeriodsStartEnds
        silentPauses: time intervals where only low-volume noise is audible (e.g. silent breathing etc)
        fineFeqSTFTTimeAxis: the time instants of the time-frames for which a
            periodicity/validity analysis was performed
        dominantBreathingFrequencies: the dominant frequencies found at specific time instants.
            When none are found, zero is returns
    :rtype:
        validBreathingPeriodsStartEnds: list of 2-element lists of pair-tuples (of floats)
        invalidBreathingPeriodsStartEnds: list of 2-element lists of pair-tuples (of floats)
        silentPauses: list of 2-element lists of pair-tuples (of floats)
        fineFeqSTFTTimeAxis: numpy array of floats
        dominantBreathingFrequencies: numpy array of floats
    """

    # Breathing periods
    (
        validBreathingPeriodsStartsEnds_old,
        _,
        silentPauses,
        breathingFineFeqSTFTTimeAxis,
        dominantBreathingFrequencies,
    ) = detectValidBreathingRegionsInOnsetSignal(
        allCurrentBreathingOnsets,
        STFTTimeAxisBreathing,
    )

    # Snoring events with no invalid sections
    (
        currentSnoringEventStartEnds,
        currentSnoringEventStrengths,
        _,
    ) = detectBreathingOrSnoringEvents(
        onset_normalization(allCurrentSnoringOnsets),
        STFTTimeAxisSnoring,
        validBreathingPeriodsStartsEnds=None,
        invalidBreathingPeriodsStartsEnds=None,
        relativeBreathingOrSnoringThreshold=Params.relativeSnoringThreshold,
    )

    # Generate onset signal from snoring events
    snore_onsets, snore_times = create_onset_from_events(
        np.array(currentSnoringEventStartEnds),
        np.array(currentSnoringEventStrengths),
        STFTTimeAxisAll["breathing"],
    )
    if len(snore_onsets) < len(snore_times):
        snore_onsets = np.append(snore_onsets, 0)

    # Match snoring onsets signal length and interpolate
    snore_onsets, snore_times = update_length_and_interpolate(
        snore_onsets,
        snore_times,
        len(allCurrentBreathingOnsets),
    )

    # Get valid, invalid and silent periods from the generated snoring onset signal
    (
        validSnoringPeriodsStartsEnds,
        invalidSnoringPeriodsStartsEnds,
        _,
        _,
        _,
    ) = detectValidBreathingRegionsInOnsetSignal(snore_onsets, snore_times)

    validBreathingSnoringPeriodsStartsEnds_arr = np.array(validSnoringPeriodsStartsEnds)

    validBreathingSnoringPeriodsStartsEnds_arr = validBreathingSnoringPeriodsStartsEnds_arr.flatten()
    validBreathingSnoringPeriodsStartsEnds_arr = np.sort(
        validBreathingSnoringPeriodsStartsEnds_arr,
    )
    validBreathingSnoringPeriodsStartsEnds_result = validBreathingSnoringPeriodsStartsEnds_arr.reshape(-1, 2)
    validBreathingPeriodsStartsEnds = validBreathingSnoringPeriodsStartsEnds_result.tolist()

    invalidBreathingSnoringPeriodsStartsEnds_arr = validBreathingSnoringPeriodsStartsEnds_arr[1:-1]
    if len(validBreathingPeriodsStartsEnds) and validBreathingPeriodsStartsEnds[0][0] > STFTTimeAxisAll["snoring"][0]:
        invalidBreathingSnoringPeriodsStartsEnds_arr = np.append(
            invalidBreathingSnoringPeriodsStartsEnds_arr,
            [STFTTimeAxisAll["snoring"][0], validBreathingPeriodsStartsEnds[0][0]],
        )
    invalidBreathingSnoringPeriodsStartsEnds_result = invalidBreathingSnoringPeriodsStartsEnds_arr.reshape(-1, 2)
    invalidBreathingPeriodsStartsEnds = invalidBreathingSnoringPeriodsStartsEnds_result.tolist()

    # Merge valid periods and replace them
    valid_merged = (
        merge_ranges(
            validBreathingPeriodsStartsEnds_old,
            validBreathingPeriodsStartsEnds,
        )
        if len(validBreathingPeriodsStartsEnds_old) or len(validBreathingPeriodsStartsEnds)
        else []
    )
    validBreathingPeriodsStartsEnds = valid_merged

    # Create invalid periods
    last_timestamp = STFTTimeAxisAll["snoring"][-1] - Params.enforcedSilencePeriodLengthInSeconds
    valid_merged_arr = np.array(valid_merged)
    valid_merged_arr = valid_merged_arr.flatten()
    invalid_merged_arr = np.insert(valid_merged_arr, 0, offsetToStartAt)
    if invalid_merged_arr[-1] <= last_timestamp:
        invalid_merged_arr = np.append(invalid_merged_arr, last_timestamp)
    else:
        invalid_merged_arr = invalid_merged_arr[:-1]
    invalid_merged_result = invalid_merged_arr.reshape(-1, 2)
    invalidBreathingPeriodsStartsEnds = invalid_merged_result.tolist()

    if len(validBreathingPeriodsStartsEnds) and validBreathingPeriodsStartsEnds[-1][1] > last_timestamp:
        validBreathingPeriodsStartsEnds[-1][1] = last_timestamp

    return (
        validBreathingPeriodsStartsEnds,
        invalidBreathingPeriodsStartsEnds,
        silentPauses,
        breathingFineFeqSTFTTimeAxis,
        dominantBreathingFrequencies,
    )


def add_events_outside_invalid_sections(range_to_check, strengths, ranges):
    """
    Remove events that fall on an existing invalid section

    Parameters:
        range_to_check (np.ndarray): times of the onset signal
        strengths (np.ndarray): amplitude of the onset signal
        ranges (np.ndarray): invalid section

    Returns:
        np.ndarray: times of the onset signal
        np.ndarray: amplitude of the onset signal
    """

    loop_start = 0
    new_ranges = []
    new_strengths = []

    for i in range(0, len(range_to_check)):
        for j in range(loop_start, len(ranges)):
            if range_to_check[i][0] >= ranges[j][0] and range_to_check[i][1] <= ranges[j][1]:
                new_ranges.append(range_to_check[i])
                new_strengths.append(strengths[i])
                loop_start = j

    return np.array(new_ranges), np.array(new_strengths)


def merge_ranges(list1, list2):
    """
    Combine two lists with timestamps and sort it

    Parameters:
        list1 (list): timestamps of list 1
        list2 (list): timestamps of list 2

    Returns:
        list: list of timestamps
    """

    # Combine the two lists
    combined_list = list1 + list2

    # Sort the combined list based on the start time
    combined_list.sort(key=lambda x: x[0])

    merged_list = []
    current_range = list(combined_list[0])

    # Iterate over the sorted list and merge overlapping ranges
    for start, end in combined_list[1:]:
        if start <= current_range[1]:
            # Overlapping range, update the end time
            current_range[1] = max(current_range[1], end)
        else:
            # Non-overlapping range, add the current range to the merged list
            merged_list.append(current_range)
            current_range = [start, end]

    # Add the last range to the merged list
    merged_list.append(current_range)

    return merged_list


def minuteMovingAVGFilter(apneaDetections, totalMinutes, window=30):
    """Moving averge filter for Apnea Detections

    :param apneaDetections: All time stamps where the apnea is detected
    :type apneaDetections: tuple array
    :param totalMinutes: Total duration of the stream
    :type totalMinutes: int
    :param window: window size of moving avergae filter, defaults to 30
    :type window: int, optional
    :return: the values after filtering
    :rtype: float array
    """
    apneaMinutes = np.array(list(map(int, np.array(apneaDetections) / 60)))
    filteredValue = [
        len(
            np.where(
                (apneaMinutes > i - (window / 2)) & (apneaMinutes < (i + (window / 2))),
            )[0],
        )
        / window
        for i in range(totalMinutes)
    ]
    return filteredValue


def ranges(nums) -> tuple:
    """Creating tuple array for finding adjacent numbers

    :param nums: array of integers
    :type nums: int array
    :return: tuple array of adjacent numbers
    :rtype: tuple array
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def reAdaptSTFTParams(
    n_fft: int = Params.n_fft,
    hop_length: int = Params.spectrum_hop_length,
    sample_rate: int = Params.SAMPLE_RATE_STFT,
):
    """Function used to translate time-dependent spectral analysis parameters from reference to working sample rate

    :param n_fft: frame/window size in samples at the reference sample rate (Params.SAMPLE_RATE_STFT).
        Defaults to Params.n_fft
    :type n_fft: int, optional
    :param hop_length: number of samples (step) by which to move each window, taken at the reference
        sample rate (Params.SAMPLE_RATE_STFT)), defaults to Params.spectrum_hop_length
    :type hop_length: int, optional
    :param sample_rate: sample rate/frequency in Hz, defaults to Params.SAMPLE_RATE_STFT
    :type sample_rate: int, optional
    :return:
        n_fft: original n_fft translated to correspond to provided sample rate/frequency
        hop_length: original hop_length translated to correspond to provided sample rate/frequency
    :rtype:
        n_fft: int
        hop_length: int
    """

    logger.info("Re-adapting STFT parameters")
    # used to map the STFT parameters from those used at 16 kHz
    n_fft = int(np.ceil(n_fft * sample_rate / Params.SAMPLE_RATE_STFT))
    hop_length = int(np.ceil(hop_length * sample_rate / Params.SAMPLE_RATE_STFT))

    return n_fft, hop_length


def openAudioStreamAtSnorefoxParameters(
    filePath,
    offsetToStartAt: float = 0.0,
    durationToRead: float = None,
):
    """Function used to "stream" an audio file to "open" it without loading it completely into memory

    The function is supposed to "open" the file in blocks, essentially yielding a generator that allows for reading
    a well-chosen portion of the audio file, potentially until a specific duration, and only load that into memory.
    Later portions of the file are read in the same way, but only on-demand. This allows for loading an audio file
    in blocks, instead of loading it in one fell-swoop.

    :param filePath: path of the audio file to read
    :type filePath: str, int, or Path object
    :param offsetToStartAt: offset in seconds to delay the reading by -- only the file portion after this much
        time in seconds is read, defaults to 0.0
    :type offsetToStartAt: float, optional
    :param durationToRead: duration in seconds for how long to allow reading the file -- the file is read starting
        from zero seconds (or "offsetToStartAt" seconds if given) until "durationToRead" seconds
        (or "durationToRead" + "offsetToStartAt" seconds if the latter is given). Can be set to None to allow
        reading the file until the last sample / until the end. Defaults to None
    :type durationToRead: float, optional
    :return:
        audioSignal: audio signal (read at the supplied or its native sample rate) to be read in blocks
        sample_rate: sample rate at which the file is set up to be read
        effectiveLibrosaStreamingBlockLengthInSeconds: effectively-resulting audio block duration (in seconds)
        readDuration: the total duration (in seconds) for which audio is read
    :rtype:
        audioSignal: generator returning 1D numpy arrays (of floats)
        sample_rate: int
        effectiveLibrosaStreamingBlockLengthInSeconds: float
        readDuration: float
    """
    maxReadingTimeCounter = 0
    if offsetToStartAt != 0:
        maxReadingTimeCounter += offsetToStartAt

    if durationToRead is not None:
        maxReadingTimeCounter += durationToRead

    readDuration = maxReadingTimeCounter - offsetToStartAt

    logger.info("Opening audio stream")

    sample_rate = librosa.get_samplerate(filePath)

    # needs to use Params.n_fft_big for compatibility with
    # Params.librosaStreamingBlockLengthInFrames, otherwise update the latter to re-use
    # the n_fft you program this with
    n_fft_big_re_adapted, _ = reAdaptSTFTParams(
        n_fft=Params.n_fft_big,
        sample_rate=sample_rate,
    )

    effectiveLibrosaStreamingBlockLengthInSeconds = (
        Params.librosaStreamingBlockLengthInFrames * n_fft_big_re_adapted / sample_rate
    )

    # loads the audio stream using blocks of a couple of minutes each (or whatever chosen duration)
    audioSignal = librosa.stream(
        filePath,
        block_length=Params.librosaStreamingBlockLengthInFrames,
        frame_length=n_fft_big_re_adapted,
        hop_length=n_fft_big_re_adapted,
        offset=offsetToStartAt,
        duration=durationToRead,
    )

    return (
        audioSignal,
        sample_rate,
        effectiveLibrosaStreamingBlockLengthInSeconds,
        readDuration,
    )


def onset_normalization(onset):
    """
    Apply a series of normalization techniques to the given onset signal.

    Parameters:
    - onset (array-like): The input onset signal.

    Returns:
    - normalized_onset: The onset signal after applying peak, RMS, limiter, and expander normalization.

    Notes:
    - Peak Normalization scales the onset signal to a maximum amplitude of 500.0.
    - RMS normalization adjusts the amplitude to achieve an RMS level of 120.0.
    - Limiter caps the amplitude at a threshold of 500.0 to prevent clipping.
    - Expander lowers the amplitude of small values below a threshold while preserving larger values.
    """
    # Peak Normalization
    onset = 500.0 * (onset / np.nanmax(np.abs(onset)))

    # RMS normalization
    rms = np.sqrt(np.nanmean(onset**2))
    onset *= 120.0 / rms

    # Limiter
    threshold = 500.0
    onset = np.where(
        np.abs(onset) > threshold,
        threshold * np.sign(onset),
        onset,
    )

    # Expander
    expander_threshold = 500.0 * 0.15
    expander_gain = 0.15
    onset = np.where(
        np.abs(onset) < expander_threshold,
        expander_gain * onset,
        onset,
    )

    return onset

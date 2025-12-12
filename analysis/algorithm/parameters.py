import numpy as np

# numerical noise tolerance around 1 to use when checking indicator values
indicatorNoiseTolerance = 0.1

# audio-internal parameters, select one and comment out the rest

# sample rate of all clips in the somnofox ACTLE database -- reference sample rate on which parameters
# were chosen -- JUST DON'T CHANGE THIS UNLESS KNOW WHAT YOU'RE DOING -- RISK OF NUCLEAR MELTDOWN
SAMPLE_RATE_STFT = 16000

# ######### general FFT/STFT/MFCC parameters

basicNfftToSamplerateRatio = 256 / 16000  # the original parameters were chosen based on 16 kHz

n_fft = round(basicNfftToSamplerateRatio * SAMPLE_RATE_STFT)
spectrum_hop_length = round(n_fft / 4)

# these windows are big enough to feature multiple cycles of snoring impulsive trains etc
# second factor is to automatically re-adapt in case reference fs changed
n_fft_big = round(2048 * (basicNfftToSamplerateRatio * SAMPLE_RATE_STFT) / n_fft)
spectrum_hop_length_big = round(n_fft_big / 4)


# ######### F0 / pitch parameters (snoring-specific)

minSnoringF0 = 5
maxSnoringF0 = 300
minPitchStrength = 0.3
voicingThreshold = 0.15


# ######### hand-picked thresholds for use when calling mag2dB
dB_standard_cutoff_threshold = -50
STFT_cutoff_dB_threshold = dB_standard_cutoff_threshold

filterOrderToUse = 15

lowPassCutoffFreqChebySomnofox = 6000  # in Hz for stricter chebychev filter

snorefoxAudioFrequencyBandwidth = 4400
highPassCuttoffFreqChebySnorefox = 600  # in Hz for stricter chebychev filter
desiredLibrosaStreamingBlockLengthInSeconds = 5 * 60
librosaStreamingBlockLengthInFrames = int(
    np.ceil(desiredLibrosaStreamingBlockLengthInSeconds / (n_fft_big / SAMPLE_RATE_STFT))
)
lowPassCutoffFreqChebySnorefox = snorefoxAudioFrequencyBandwidth + highPassCuttoffFreqChebySnorefox

# in Hz for stricter chebychev filter, used for snoring specifically (not breathing + snoring)
snoringHighPassCuttoffFreqCheby = 250
snoringHighPassCuttoffFreqChebyLow = 50

# in Hz for stricter chebychev filter, used for snoring specifically (not breathing + snoring)
snoringLowPassCuttoffFreqCheby = 1500
snoringLowPassCuttoffFreqChebyLow = 200

# ######## breathing description parameters

minBreathingLengthInSeconds = 0.45
# observed on the 0016th 15-s window of 200731-22050 -- can take that since this is a minimal amount
# observed on the stream_39 around 13824s -- this is exceptional though / not a rule to follow
maxBreathingLengthInSeconds = 3  # based on- discussion with Christoph right before 2021-07-14 17:43:18

minInhaleExhaleLengthMargin = 0.2
# observed on the 0016th 15-s window of 200731-22050 -- can take that since this is a minimal amount

maxInhaleExhaleLengthMargin = 0.37  # observed on stream_27 around 9925s
minExhaleInhaleMargin = 0.9  # 0.9-1s observed around 9927s and 0.9s on 9941s on stream_27

maxBreathingInhaleExhaleLengthInSeconds = 2 * maxBreathingLengthInSeconds + maxInhaleExhaleLengthMargin
# 1 max length for inhale, 1 for exhale + pause in betweeen
minBreathingInhaleExhaleLengthInSeconds = 2 * minBreathingLengthInSeconds + minInhaleExhaleLengthMargin

# )  # between onset of one breath and another -- using other assumptions
minBreathingDistanceInSeconds = 60 / 30  # using 30 bpm here to leave a margin wrt the observed/normal range
# between onset of one breath and another -- using sleep foundation observations as of 2022-01-26
maxBreathingDistanceInSeconds = 8
# between onset of one breath and another, period can go up to 8s as observed on 210409-22050 around 1h44m

# ######## apnea definition parameters

minApneaLengthInSeconds = 10  # 10s based on the AASM rules

# based on 95 percentile ovserved on greek database with some 13-15s margin added
maxApneaLengthInSeconds = 60

# ######## serial apnea pre-detection parameters (for serial apnea episodes)

# maxApneaLengthInSecondsInSerialRegions = 87  # only in serial apnea episodes (for prelininary search)
# # to allow maximal serial apnea as observed for the maximal case on 200810-22050 between 6023-6110s

# only in serial apnea episodes (for prelininary search)
maxApneaLengthInSecondsInSerialRegions = maxApneaLengthInSeconds

# minimum number of short silence periods happening in sequence for validity detector to
# treat their region as a potential serial apnea region
noOfShortSilencesForSerialApneaPredetection = 3

# ######## breath detection parameters

relativeBreathThreshold = 10 / 100
relativeSnoringThreshold = 45 / 100
# chosen explicitly to be higher than the relative breathing detection threshold

# ######## apnea amplitude reduction/increase enforcement parameters

# allow more hypopneas
apneaBreathRelativeDropThreshold = 52.5 / 100

# ######## general AHI estimation reliability parameters

enforcedSilencePeriodLengthInSeconds = 2 * 60  # period to trim out / silence at beginning / end of file

# the minimum duration of the audio used for computing onset signals,
# to make sure trimming is only applied if the() audio data is at least 1.5x as long as the portion
# to be cut out to leave SOMETHING after trimming
minUsableAudioDuration = 3 * enforcedSilencePeriodLengthInSeconds

# the minimum amount of time an AHI can be based on, to avoid high AHI quantization errors
minimumAHITimeBasis = 1.5 * 60 * 60

# ######## onset signal filtering parameters

spectrum_hop_length_breathing = n_fft
n_fft_snoring = n_fft * 4
spectrum_hop_length_snoring = n_fft_snoring

breathingOnsetSiftingPercentile = 75
snoringOnsetSiftingPercentile = 95

breathingPassiveDuratonToUse = None
breathingGradualOnsetUse = True
breathingHWRectificationUse = False
breathingFrequencyLossDetection = False

snoringPassiveDuratonToUse = None
snoringGradualOnsetUse = False
snoringHWRectificationUse = False
snoringFrequencyLossDetection = False

defaultOnsetFrequencyBandwidthToAverage = 6 * (22050 / 353)
# default is at 6 frequency bins a 353-windowed STFT at 22.05 kHZ (approx. 368 Hz), re-adapted from there

# ######## apnea search parameters

# adopting an algorithmic choice -- shorter than 2mins to allow for better detection of serial apneas
noOfBreathsToUseAsApneaAmplitudeReference = 3

noOfPreviousBreathstoEnforceBeforeApnea = 2
noOfBreathstoEnforceAfterApnea = 2

# ######## impulsive noise parameters
relativeImpulseThreshold = 85 / 100

impulsiveNoiseSupressionNeighborhoodStrictInSeconds = 2 * minBreathingInhaleExhaleLengthInSeconds

impulseCausalityTolerance = maxBreathingInhaleExhaleLengthInSeconds
leniantSupressionNeighborhoodFactor = 5
noOfImpulsiveOutliersToAllowInLenaintNeighborhood = 2
impulsiveNoiseSupressionNeighborhoodLeniantInSeconds = (
    leniantSupressionNeighborhoodFactor * impulsiveNoiseSupressionNeighborhoodStrictInSeconds
)

# ######## periodicity detection parameters

minBreathingFrequency = 1 / maxBreathingDistanceInSeconds
maxBreathingFrequency = 1 / minBreathingDistanceInSeconds

# )  # fine frequency resolution
noOfFrequenciesToDistinguishInBreathingBand = 10  # fine frequency resolution
periodicityAnalysisTimeResolution = 1.15  # in seconds
relativePeriodicityFrequencySelectionThreshold = 80 / 100
majorityVoteProportionForPeriodicity = 55 / 100

# ######## frequency jumping prohibition parameter

maxFrequencyJumpCheckupTime = 30

minBreathingFrequencyResolution = (
    maxBreathingFrequency - minBreathingFrequency
) / noOfFrequenciesToDistinguishInBreathingBand

maxAllowableFrequencyJumpTimeStrict = 20
# chosen based on observation / experience given the periodicity parameter as of 2022-06-17
maxAllowableFrequencyJumpStrict = (
    maxBreathingFrequency - minBreathingFrequency  # this much frequency change allowed
) / maxAllowableFrequencyJumpTimeStrict  # in this much time in seconds, allowing this much to account
# for this edge case: the serial apneas in 200810-22050, which show such sharp changes can happen

maxAllowableFrequencyJumpTimeLenient = 3
# chosen based on observation / experience given the periodicity parameter as of 2022-06-17
maxAllowableFrequencyJumpLenient = (
    maxBreathingFrequency - minBreathingFrequency  # this much frequency change allowed
) / maxAllowableFrequencyJumpTimeLenient  # in this much time in seconds
# this one allows much more jumps and is used for regions with potential serial apneas

# serial apneas (as on 210418_22050 between 6740-7700s)
minTimeBinsForFreqJumpSupression = 4 - 0.1

# if one region is much longer than the other, it makes little sense to have the super-smaller
# one cancel one X times its size, so then only cancel the smaller one
imbalancedRegionFactor = 5

#  in terms of onset signal. It’s an observation, not a parameter
silenceAbsoluteEnergyThreshold = 20**2

# the silent period duration limits need to be smaller than their respective apneas
# cuz energy averaging will make energy leak into the silent periods, causing them
# to be detected effectively shorter than the actual silence, and also keeping it lower
# for min apnea duration to allow for better friendliness towards serial hypopneas
# (these don't involve full but only partial silence)
maxSilentPeriodDuration = maxApneaLengthInSecondsInSerialRegions
minSignificantSilentPeriodDuration = 0.4 * minApneaLengthInSeconds

# this needs ot be just above the maximal apnea duration, cuz otherwise those themselves
# would be confused with silent pauses. Leaving a margin above 100% of that
minSilentPauseDuration = 1.1 * maxApneaLengthInSeconds

majorityVoteProportionBumpForValidity = (
    5 / 100
)  # bump up the majority selection threshold if using silent period treatment

# ##### AHI cluster/amplitude weighting parameters
# clean apneaThreshold by cris
cleanApneasThreshold = 10

minApneaWeightOutsideClusters = 0.9

# Constants for computeAHIFromDetections method
apneaIndicatorFs = 0.5
maxApneaTimeMargin = 10 * 60  # leaving a 10s margin for the signaling analysis

# choosing the time basis for what to consider a "cluster" for the apnea cluster factor
minNoOfApneasInCluster = 3
apneaClusterFactorTimeBasis = minNoOfApneasInCluster * 100  # in seconds

# the apnea amplitude deviation from AARE/AAIE that should be considered to give a 100% apnea weight,
# apneas with higher deviations from their previous/next neighborhoods are weighted accordingly more,
# those with smaller deviations are weighted accordingly less.
fullWeightedApneaAmplitudeDeviation = 0.9

# the apnea amplitude deviation beyond which the apneotic event is considered an apnea, not a hypopnea
apneaDistinguishingAmplitudeDeviation = 0.7

# same logic as above but for apnea durations
fullWeightedApneaDuration = 35

# ########

negligibleToMildAHIThreshold = 5
mildToSevereAHIThreshold = 15
severeToVerySevereAHIThreshold = 30

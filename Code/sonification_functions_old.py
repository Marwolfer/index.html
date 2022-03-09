import sox
import numpy as np
from scipy import interpolate
import datetime
import scipy.special
from matplotlib import pyplot as plt
import scipy.signal
import librosa
from pydub import AudioSegment

# consider including sample rate as function key-word argument
sample_rate = 44100

def plot_samples(*args):
    # maybe include check so all drawn samples of distribution have same siue?
    fig = plt.figure(figsize=(10,8))
    #print(len(args[0]))
    for i in range(len(args[0])):
        plt.plot([i]*len(args[0][i]),args[0][i],'.',alpha=0.3)


# consider using in on whole array if possbile any way
def interpolate_without_outliers2(A, min_, max_):
    # note: scaling all distributions by the same scale only leads to all distributions being whitenoise
    C = np.array([])
    f = interpolate.interp1d((min_, max_), (400, 5000))
    for i in range(len(A)):
        # remove values below 0 and above 1 from array
        # np_arg_0 = A[i][0<=A[i]]
        # np_arg_1 = A[i][A[i]<=1]
        # print("length np_arg_0: {}".format(len(np_arg_0)))
        # print("length np_arg_1: {}".format(len(np_arg_1)))
        # print("Arrays equal {}".format(np.array_equal(np_arg_0, np_arg_1)))
        # equal_arrays = np.array_equal(np_arg_0, np_arg_1)
        # np_arg_intersect = np.intersect1d(np.round(np_arg_0,5),np.round(np_arg_1,5))
        # print("xor :{}".format((np.setxor1d(np_arg_0,np_arg_1))))
        # print("length np_arg_intersect: {}".format(len(np_arg_intersect)))
        # if equal_arrays:
        # B =np.array(f(A[i]))
        # else:
        B = np.array(f(A[i]))
        C = np.concatenate((C, B))
    # print(len(C))

    return C.reshape(len(A), -1)

def interpolate_sigmoid(A):
    C = np.array([])
    for i in range(len(A)):
        B = np.array(sigmoid(A[i]))
        C = np.concatenate((C, B))



    return C.reshape(len(A),-1)

def sigmoid(x):
    return 4602*scipy.special.expit(x*7.8)+400

def interpolate_sigmoid(A):
    C = np.array([])
    for i in range(len(A)):
        B = np.array(sigmoid(A[i]))
        C = np.concatenate((C, B))

    return C.reshape(len(A),-1)

def create_sine_signal(freq, duration=1.0, volume=0.8):
    # duration in seconds, volume=0.8 max to have a chance at preventing clipping
    return volume*np.sin(2 * np.pi * freq * np.arange(sample_rate * duration) / sample_rate)

def create_sine_signal_change(f_start, f_end, duration):
    # inp
    n_samples = int(sample_rate*duration)
    A = np.zeros(int(sample_rate*duration))
    for i in range(n_samples):
        delta = i/n_samples
        t = duration * delta
        phase = 2* np.pi * t * (f_start + (f_end - f_start)*delta /2)
        A[i] = np.sin(phase)
    return A


def make_sample_sounds1D(A1D, freq_lookup_table, duration=1.0, volume=0.8):
    # One array with frequencies is passed as input
    # output is array of arrays. each inner array is a sine waveform with the input frequency
    Sines = np.zeros((len(A1D), int(sample_rate * duration)))
    for i, freq in enumerate(A1D):
        # print(freq)
        Sine = freq_lookup_table[int(freq)]
        Sines[i] = Sine
    return Sines


# maybe rename to overlay sample sounds, overlay frequencies etc...
def make_sample_sounds2D(A2D, freq_lookup_table, duration=1.0, volume=0.8):
    # input is an Array of arrays of frequencies. later this will be of size batch x parameters
    # Data of each inner arrays need to be added, then normalized (+ adjustment for loudness/loundess contour curve)
    # for each frequency in inner Array, a whole Array is created

    # return should be
    Sounds = np.zeros((len(A2D), int(sample_rate * duration)))
    # print("Sounds Shape {}".format(Sounds.shape))
    tfm = sox.Transformer()
    tfm.norm()
    for i in range(len(A2D)):
        # simply add elements of inner arrays to overlay sines
        Sines = make_sample_sounds1D(A2D[i], freq_lookup_table, duration, volume=0.8)
        # print("Sines shape: {}".format(Sines.shape))
        overlaid_Sines = Sines[0]

        # overlay sines be adding arrays of sines element-wise
        for j in range(1, len(Sines)):
            overlaid_Sines = np.add(overlaid_Sines, Sines[j])

        # normalize input to prevent clipping
        overlaid_Sines_norm = tfm.build_array(input_array=overlaid_Sines, sample_rate_in=int(sample_rate * duration))
        Sounds[i] = overlaid_Sines_norm
        if np.mod(i, 10) == 0:
            print("Batch: {}, Time {}".format(i, datetime.datetime.now()))
    return Sounds


def concatenate_signals1D(A, duration=1.0, crossfade=0.001, sample_rate=44100):
    # takes an 1D Array of sine signal arrays as input an concatenates them.
    # given a single sine array as output

    tfm = sox.Transformer()
    tfm.fade(0.001, 0.001)
    for i in range(len(A)):
        A[i] = tfm.build_array(input_array=A[i], sample_rate_in=sample_rate)
    Concatenated = np.zeros(int(len(A) * sample_rate * duration))
    idx = 0
    for i in range(len(A)):
        Concatenated[idx:idx + len(A[i])] = A[i]
        idx = idx + len(A[i])

    return Concatenated

def get_frequency_from_sine(A):
    #https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
    # input: a single sine signal
    # output: frequency in that sine singal with highest power
    w = np.fft.fft(A)
    freqs = np.fft.fftfreq(len(A))
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * 44100)
    return int(freq_in_hertz)


# lenght of all bridges together needs to be computed beforehand. If that is the case, bridges can also be computed beforehand
def get_bridges(sinesarray, duration=0.002):
    # input either only two arrays and returs bridge array
    # or input array of arrays, returns array of arrays (bridges between input arrays)
    bridges = []
    length_bridges = 0

    # determine frequencies in sinesarray
    for sine in sinesarray:
        freqs.append(get_frequency_from_sine(sine))

    for i in range(len(freqs) - 1):

        freq_first = freqs[i]
        freq_second = freqs[i + 1]

        # max duration for frequeny change should be 1 second for a change from 400Hz to 5000Hz
        # compute duration for frequency change relative to that

        # consider keeping duration of bridge constant and small instead of doing it like that
        if freq_frist >= freq_second:
            duration = (freq_first - freq_second) / (5000 - 400)
        else:
            duration = (freq_second - freq_first) / (5000 - 400)
        bridge_signal = create_sine_signal_change(freq_first, freq_second, duration)

        length_bridges += len(bridge_signal)
        print("length Bridges: {}".format(length_bridges))
        # add them to output list of bridges
        bridges.append(bridge_signal.tolist())

    # here it probably makes more sense to store them in a linked list, as the length of the bridge is impossible+
    # to determine beforA = []
    return (bridges, length_bridges)


def concatenate_signals1D_bridges(A, duration=1.0, sample_rate=44100):
    A_bridges = get_bridges(A)
    bridges = A_bridges[0]
    length_bridges = A_bridges[1]
    # tfm = sox.Transformer()
    # tfm.fade(0.001,0.001)
    # for i in range(len(A)):
    # A[i] = tfm.build_array(input_array=A[i], sample_rate_in = sample_rate)
    Concatenated = np.zeros(int(len(A) * sample_rate * duration) + length_bridges)
    idx = 0
    for i in range(len(A)):
        Concatenated[idx:idx + len(A[i])] = A[i]
        idx = idx + len(A[i])
        if i < (len(bridges)):
            print("Bridge {}".format(i + 1))
            Concatenated[idx:idx + len(bridges[i])] = bridges[i]
            idx = idx + len(bridges[i])

        # print(idx)
    return Concatenated


def sound_avg_pool(A, num_frames):
    # this functions is supposed to split the array into frames(array needs to be sorted first)
    # then the mean frequency of each frame is supposed to be taken. The result should be a sound where
    # fewer frequencies need to be overlaid.
    # obviously len(A)<<num_frames

    # function takes 1D and 2D nd.arrays as inputs for A
    if len(A.shape) == 1:
        frequencies_per_frame = int(len(A) / num_frames)
        pooled_frequencies = np.array([])
        A = np.sort(A)
        for i in range(num_frames):
            if (i + 1) * frequencies_per_frame < len(A):
                pooled_frequencies = np.append(pooled_frequencies,
                                               A[i * frequencies_per_frame:(i + 1) * frequencies_per_frame].mean())
            else:
                pooled_frequencies = np.append(pooled_frequencies, A[i * frequencies_per_frame:-1].mean())
        return pooled_frequencies

    else:
        # quick fix, consider moving into for loop
        frequencies_per_frame = int(len(A[0]) / num_frames)
        # print(frequencies_per_frame)
        pooled_frequencies_2D = np.array([])
        A = np.sort(A)
        for j in range(len(A)):
            pooled_frequencies = np.array([])
            for i in range(num_frames):
                if (i + 1) * frequencies_per_frame < len(A[j]):
                    pooled_frequencies = np.append(pooled_frequencies, A[j][i * frequencies_per_frame:(
                                                                                                                  i + 1) * frequencies_per_frame].mean())
                else:
                    pooled_frequencies = np.append(pooled_frequencies, A[j][i * frequencies_per_frame:-1].mean())
                # print(pooled_frequencies)
            pooled_frequencies_2D = np.append(pooled_frequencies_2D, pooled_frequencies)
        return pooled_frequencies_2D.reshape(len(A), -1)


def create_sine_lookuptable(max_freq=5000, duration=1.0, volume=0.8):
    # should be +1 but sigmoid not fine tuned right now, has max of 5002
    # A = np.zeros((max_freq+5,int(duration*sample_rate)))
    # print(np.round(duration*sample_rate))
    A = np.zeros((max_freq + 5, int(np.round(duration * sample_rate))))
    for freq in range(0, max_freq + 5):
        A[freq] = volume * np.sin(2 * np.pi * freq * np.arange(int(np.round(sample_rate * duration))) / sample_rate)
    return A


def save_to_file(A, filename):
    # pydub does not support 24bit wave files, so use different encoding
    tfm = sox.Transformer()
    tfm.set_output_format(file_type="wav", rate=44100, bits=16)
    tfm.set_input_format(file_type="wav", rate=44100, bits=16)
    return tfm.build_file(
        input_array=A, sample_rate_in=sample_rate,
        output_filepath=filename, return_output=True,
    )


def log_scaled_freq(x):
    return 4602 * scipy.special.expit(x * 55000) + 400


def epoch_trend(Batch_Grads):
    # input needs to have format epoch x Batch_Nr x Gradient
    # input also already needs to be "absoluted" and "logarithmed"
    # mean_arr = np.zeros(len(Batch_Grads))
    # sigma_arr = np.zeros(len(Batch_Grads))
    # skew_arr = np.zeros(len(Batch_Grads))
    # kurt_arr = np.zeros(len(Batch_Grads))

    Moments_arr = np.zeros((len(Batch_Grads), 4))

    # 2,3,4,5
    for i in range(len(Batch_Grads)):
        BatchGrad_arg_log = np.log(np.abs(Batch_Grads[i]))
        is_inf_idx = np.isinf(BatchGrad_arg_log).flatten()
        BatchGrad_arg_log_wo_infty = np.delete(BatchGrad_arg_log, is_inf_idx)
        Moments_arr[i] = scipy.stats.describe(BatchGrad_arg_log_wo_infty)[2:]
    return Moments_arr


def logabs_transform(A):
    # input in form epoch x gradient
    # as it looks right now, the transformation has to be done for each Gradient individually, otherwise,
    # the correct ordering will be lost after flattening the Array for replace
    # x[x == -inf] = 0
    A_logabs = np.log(np.abs(A))
    A_logabs[np.isneginf(A_logabs)] = -3000
    return A_logabs


def sampleGrad_scalarprodMatrix(A):
    # input of form  batchsize x num_parameters
    dotMat = np.ones((A.shape[0], A.shape[0]))

    for i in range(len(dotMat)):
        for j in range(i + 1, len(dotMat)):
            # if (np.linalg.norm(A[i])*np.linalg.norm(A[j])) == 0:
            # print("zero")
            dotMat[i, j] = np.dot(A[i], A[j]) / (np.linalg.norm(A[i]) * np.linalg.norm(A[j]))
    return dotMat

    # should return a scalar product matrix for each epoch
    # only elements above, below diagonal need to be sonified, diag elements can be filled with 1(or rather 2)


def get_dotMat_sines(dotMat, freq_lookup_table, duration=1.0, sample_rate=44100):
    # goal of this function: sonify dotproduct Matrix by overlaying all dot products in an epoch
    # input # dotmat of one epoch
    dotMat_freq = linear_value_to_freq(dotMat)
    print(dotMat_freq)
    # here two options: seperate matrices for sines above/below 2280Hz? and play them on left/right ear or play them combined on stereo?
    # first: combined option

    # better implementation, get upper triangel with triu, count nonzero elements?

    dotMat_idx = dotMat != 2
    len_dotMat = len(dotMat[dotMat_idx])
    A = np.zeros((len_dotMat, int(duration * sample_rate)))
    idx = 0

    for i in range(len(dotMat)):
        for j in range(i + 1, len(dotMat)):
            # print(dotMat_freq[i,j])
            # let values of NaN be mapped to 0. Nan values to exist if the Norm of one of the two gradients in the denominator does
            # evaluates to 0. that happens, when if the all elements of the corresponding sample gradient are 0

            print(idx)
            print(i, ",", j)
            if np.isnan(dotMat_freq[i, j]):

                A[idx] = freq_lookup_table[0]
            else:
                A[idx] = freq_lookup_table[int(dotMat_freq[i, j])]
            idx += 1
            # print(idx)

    return A

    # first, get linear function which maps from range -1 -> 1 to frequencies (check)
    # question: -1 deep 0 middle 1 high freq?
    # or -1 and 1 high freq but different timbre?(different instrument for -1 and 1?), or different ears?


def overlay_sines(A, duration=1.0, sample_rate=44100):
    # input: matrix of sine matrices as input.
    # overlays matrices and returns normalized single sine
    tfm = sox.Transformer()
    tfm.norm()
    Sound = A[0]
    # Warning! adjust code to use torch instead of numpy
    for i in range(1, len(A)):
        Sound = np.add(Sound, A[i])

    Sound_norm = tfm.build_array(input_array=Sound, sample_rate_in=int(sample_rate * duration))
    return Sound_norm


def all_epochs(Sample_Grads, duration=1.0, sample_rate=44100):
    # input of form epoch x Samples_of_Batch x num_param

    # output should be overlaid sines for each epoch

    freq_lookup_table = create_sine_lookuptable(max_freq=5000, duration=duration, volume=1.0)
    overlaid_sines_per_epoch = np.zeros((len(Sample_Grads), int(duration * sample_rate)))

    for i in range(len(Sample_Grads)):
        print(i)
        dotMat = sampleGrad_scalarprodMatrix(Sample_Grads[i])
        dotMat_sines = get_dotMat_sines(dotMat, freq_lookup_table, duration=duration, sample_rate=sample_rate)
        overlaid_sines_per_epoch[i] = overlay_sines(dotMat_sines, duration=duration, sample_rate=sample_rate)

    conc = concatenate_signals1D(overlaid_sines_per_epoch)
    return conc


def overlay_sines_loudness(A, duration=1.0, sample_rate=44100):
    # input: matrix of sine matrices as input.
    # overlays matrices and returns normalized single sine
    tfm = sox.Transformer()
    # tfm.loudness(gain_db=-10.0, reference_level = 65)
    tfm.equalizer(5000, 0.5, 5)
    tfm.norm()
    Sound = A[0]
    # Warning! adjust code to use torch instead of numpy
    for i in range(1, len(A)):
        Sound = np.add(Sound, A[i])

    Sound_norm = tfm.build_array(input_array=Sound, sample_rate_in=int(sample_rate * duration))
    return Sound_norm


def epoch_trend_sampleGrad(SampleGrads):
    # input needs to have format epoch x Batch_Nr x Gradient
    # input also already needs to be "absoluted" and "logarithmed"
    # mean_arr = np.zeros(len(Batch_Grads))
    # sigma_arr = np.zeros(len(Batch_Grads))
    # skew_arr = np.zeros(len(Batch_Grads))
    # kurt_arr = np.zeros(len(Batch_Grads))

    dotMat_abssum = np.zeros(len(SampleGrads))
    Moments_arr = np.zeros((len(SampleGrads), 4))
    upper_indices = np.triu_indices(SampleGrads.shape[1], 1)
    # 2,3,4,5
    for i in range(len(SampleGrads)):
        dotMat = sampleGrad_scalarprodMatrix(SampleGrads[i])
        dotMat_abssum[i] = sum(np.abs(dotMat[upper_indices]))
        Moments_arr[i] = scipy.stats.describe(dotMat[upper_indices])[2:]
    return Moments_arr, dotMat_abssum


def get_dotMat_sines2(dotMat, freq_lookup_table, duration=1.0, sample_rate=44100):
    #
    # input # dotmat of one epoch

    # output: sines for this dotMat. sines are seperated into sines for positive and sines for negative scalar Product values

    # here two options: seperate matrices for sines above/below 2280Hz? and play them on left/right ear or play them combined on stereo?
    # first: combined option

    # better implementation, get upper triangel with triu
    upper_indices = np.triu_indices(dotMat.shape[1], 1)
    len_dotMat = len(dotMat[upper_indices])
    # print(len_dotMat)
    dotMat = dotMat[upper_indices]
    # print(dotMat.shape)
    # print(len_dotMat)

    # play exactly orthongal sampleGrads (scalarProduct of 0) one both earphones?
    dotMat_positive = dotMat[dotMat >= 0]
    dotMat_negative = dotMat[dotMat <= 0]
    # print(dotMat_positive.max())
    # print(dotMat_negative.min())

    # print(len(dotMat_positive))
    # print(len(dotMat_negative))

    # apply average pool in relation to size of negative and postive scalarProds.
    # split into 50 frames, each frame should have sound length of 20ms
    total_frames = len(dotMat_positive) + len(dotMat_negative)
    positve_frames = int(50 * len(dotMat_positive) / total_frames)
    negative_frames = int(50 * len(dotMat_negative) / total_frames)
    # print(total_frames)
    # print(positve_frames)
    # print(negative_frames)

    # choose different mapping functions? Options include x^2, other linear but with big frequency band between negative and positve frequencies
    dotMat_positive_freq = x_sqr(dotMat_positive)
    dotMat_negative_freq = x_sqr(dotMat_negative)
    # print(dotMat_positive_freq.max())

    positive_freq_avg_pool = sound_avg_pool(dotMat_positive_freq, positve_frames)
    negative_freq_avg_pool = sound_avg_pool(dotMat_positive_freq, negative_frames)

    dotMat_positive_sines = np.zeros((len(positive_freq_avg_pool), int(duration * sample_rate)))
    dotMat_negative_sines = np.zeros((len(negative_freq_avg_pool), int(duration * sample_rate)))

    for i in range(len(positive_freq_avg_pool)):
        dotMat_positive_sines[i] = freq_lookup_table[int(positive_freq_avg_pool[i])]

    for i in range(len(negative_freq_avg_pool)):
        dotMat_negative_sines[i] = freq_lookup_table[int(negative_freq_avg_pool[i])]

    print(dotMat_negative_sines.shape)
    print(dotMat_positive_sines.shape)

    # if problem with vanishing gradients reappear, map them to a frequency of zero again
    # if np.isnan(dotMat_freq[i,j]):
    #    A[idx] = freq_lookup_table[0]

    return (dotMat_positive_sines, dotMat_negative_sines)

    # first, get linear function which maps from range -1 -> 1 to frequencies (check)
    # question: -1 deep 0 middle 1 high freq?
    # or -1 and 1 high freq but different timbre?(different instrument for -1 and 1?), or different ears?


def all_dotMats(SampleGrads, filename, duration=0.02):
    # input of shape epochs x Batchsize x num_params,
    # filename needs to end with .wav
    # duration ~20 ms , so 50 frequencies(one epoch) will take 1 sec.
    # output of shape num_individual_scalarprods
    # calculate length of upper Triangle
    cbn = sox.Combiner()
    filenames = []
    freq_lookup_table = create_sine_lookuptable(max_freq=5500, duration=duration, volume=1.0)
    for i in range(len(SampleGrads)):
        dotMat = sampleGrad_scalarprodMatrix(SampleGrads[i])
        sines = get_dotMat_sines2(dotMat, freq_lookup_table, duration=duration)
        filenames.append(stereo_gain_overlay(sines[0], sines[1], str(i) + filename, duration=duration))
    print(filenames)
    cbn.build(filenames, "concatenated_" + filename, "concatenate")
    # consider appending silent segment after each epoch


def stereo_gain_overlay(posSines, negSines, filename, duration=0.02):
    # consider including crossfade function when coded
    # very ugly and probabaly inefficient function, but is done like that for lack of better option atm

    pos_conc = concatenate_signals1D(posSines, duration=duration)
    neg_conc = concatenate_signals1D(negSines, duration=duration)
    # print(pos_conc.shape)
    # print(neg_conc.shape)
    save_to_file(pos_conc, "positive" + filename)
    save_to_file(neg_conc, "negative" + filename)
    sound_pos = AudioSegment.from_file("positive" + filename, format="wav")
    sound_neg = AudioSegment.from_file("negative" + filename, format="wav")
    sound_pos = sound_pos.apply_gain_stereo(-120, 0)
    sound_neg = sound_neg.apply_gain_stereo(0, -120)
    # find out which segement is longer to overlay correctly(otherwise if will be truncated and no information (more postive)
    # /negative scalar Products) can be gained
    if len(pos_conc) > len(neg_conc):
        sound = sound_pos.overlay(sound_neg)
    else:
        sound = sound_neg.overlay(sound_pos)

    # sound = sound_neg.append(sound_pos, crossfade=5)
    file_handle = sound.export(filename, format="wav")
    return filename


def get_dotMat_sines3(dotMat, freq_lookup_table, duration=1.0, sample_rate=44100):
    #
    # input # dotmat of one epoch

    # output: sines for this dotMat. sines are seperated into sines for positive and sines for negative scalar Product values

    # here two options: seperate matrices for sines above/below 2280Hz? and play them on left/right ear or play them combined on stereo?
    # first: combined option

    # better implementation, get upper triangel with triu
    upper_indices = np.triu_indices(dotMat.shape[1], 1)
    len_dotMat = len(dotMat[upper_indices])
    # print(len_dotMat)
    dotMat = dotMat[upper_indices]
    # print(dotMat.shape)
    # print(len_dotMat)

    # play exactly orthongal sampleGrads (scalarProduct of 0) one both earphones?
    dotMat_positive = dotMat[dotMat >= 0]
    dotMat_negative = dotMat[dotMat <= 0]
    # print(dotMat_positive.max())
    # print(dotMat_negative.min())

    # print(len(dotMat_positive))
    # print(len(dotMat_negative))

    # apply average pool give same number of frames to positive/negative dotProducts
    # split into 50 frames, each frame should have sound length of 20ms
    # total_frames = len(dotMat_positive) + len(dotMat_negative)
    positve_frames = 25
    negative_frames = 25
    # print(total_frames)
    # print(positve_frames)
    # print(negative_frames)

    # choose different mapping functions? Options include x^2, other linear but with big frequency band between negative and positve frequencies
    dotMat_positive_freq = x_sqr(dotMat_positive)
    dotMat_negative_freq = x_sqr(dotMat_negative)
    # print(dotMat_positive_freq.max())

    positive_freq_avg_pool = sound_avg_pool(dotMat_positive_freq, positve_frames)
    negative_freq_avg_pool = sound_avg_pool(dotMat_positive_freq, negative_frames)

    dotMat_positive_sines = np.zeros((len(positive_freq_avg_pool), int(duration * sample_rate)))
    dotMat_negative_sines = np.zeros((len(negative_freq_avg_pool), int(duration * sample_rate)))

    for i in range(len(positive_freq_avg_pool)):
        dotMat_positive_sines[i] = freq_lookup_table[int(positive_freq_avg_pool[i])]

    for i in range(len(negative_freq_avg_pool)):
        dotMat_negative_sines[i] = freq_lookup_table[int(negative_freq_avg_pool[i])]

    # print(dotMat_negative_sines.shape)
    # print(dotMat_positive_sines.shape)

    # if problem with vanishing gradients reappear, map them to a frequency of zero again
    # if np.isnan(dotMat_freq[i,j]):
    #    A[idx] = freq_lookup_table[0]

    return (dotMat_positive_sines, dotMat_negative_sines)

    # first, get linear function which maps from range -1 -> 1 to frequencies (check)
    # question: -1 deep 0 middle 1 high freq?
    # or -1 and 1 high freq but different timbre?(different instrument for -1 and 1?), or different ears?


def all_epochs_sines(SampleGrads, frames, duration=0.02, sample_rate=44100):
    all_epochs_pos = np.zeros((len(SampleGrads), frames, int(duration * sample_rate)))
    all_epochs_neg = np.zeros((len(SampleGrads), frames, int(duration * sample_rate)))
    freq_lookup_table = create_sine_lookuptable(max_freq=5500, duration=duration, volume=1.0)
    for i in range(len(SampleGrads)):
        dotMat = sampleGrad_scalarprodMatrix(SampleGrads[i])
        pos_sines, neg_sines = get_dotMat_sines3(dotMat, freq_lookup_table, duration=duration)
        all_epochs_pos[i] = pos_sines
        all_epochs_neg[i] = neg_sines

    # concatenate first sine/frame of every epoch
    # pro epoche x frames.
    # Dann werden in t1 je die ersten frames jeder epoche konkateniert, also num_epoch frames.
    # also hat jeder abschnitt t1 num_epoch frames, es geht aber nur t_(num_frames) verschiedene abschnitte

    sines_conc_pos = np.zeros((frames, int(len(SampleGrads) * duration * sample_rate)))
    sines_conc_neg = np.zeros((frames, int(len(SampleGrads) * duration * sample_rate)))

    # use numpy.transpose to switch dimensions SampleGrad and frame
    # tranpose to shape frames x epochs x sines
    all_epochs_pos_perm = np.transpose(all_epochs_pos, (1, 0, 2))
    all_epochs_neg_perm = np.transpose(all_epochs_neg, (1, 0, 2))

    for i in range(frames):
        sines_conc_pos[i] = concatenate_signals1D(all_epochs_pos_perm[i], duration=duration)
        sines_conc_neg[i] = concatenate_signals1D(all_epochs_neg_perm[i], duration=duration)

    return (sines_conc_pos, sines_conc_neg)




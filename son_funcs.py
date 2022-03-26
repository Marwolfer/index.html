__version__ = 'dev'

from matplotlib import pyplot as plt
import datetime
import numpy as np
import matplotlib
import numpy as np
from numpy import inf
from scipy import interpolate
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib
import math
import platform
from pathlib import Path
import subprocess
import sys
import copy
import mingus
from mingus.containers import Note
from mingus.containers import NoteContainer
import mingus.midi.midi_file_out
from mingus.containers import Bar
from mingus.containers import Track
from mingus.midi import midi_file_out
from mingus.midi.midi_file_out import MidiFile
from mingus.containers.instrument import MidiInstrument
from mingus.midi.midi_track import MidiTrack
from mingus.containers.track import Track
from mingus.containers.composition import Composition
#import matplotlib.animation as animation
from mingus.core.chords import triad
from pydub import AudioSegment
import string
matplotlib.rcParams.update({'font.size': 18})

# legacy function? does what histograms_allepochs_samplegrads but expects averaged gradients already instead all gradients in a batch
def histograms_epoch(Grads_Epoch,lower_bound,upper_bound,bin_count=20):
    bins = np.linspace(lower_bound,upper_bound,bin_count+1)
    histograms = np.zeros((Grads_Epoch.shape[0],bin_count))
    for i in range(Grads_Epoch.shape[0]):
        histograms[i] = np.histogram(Grads_Epoch[i],bins)[0]
    return histograms

# logabs changed to replace infinity with nan instead of deleteing it(as nan can be ignored with np.nanmean and is automatically
# ignored by histograms
def logabs_transform(A):
    #
    A_logabs = np.log(np.abs(A))
    A_logabs[A_logabs == -inf] = np.NAN
    return A_logabs


def sampleGrad_scalarprodMatrix(A):
    # input of form  batch x num_parameters
    # calculates all pair-wise scalar Products of gradients in a batch
    dotMat = np.ones((A.shape[0], A.shape[0]))

    for i in range(len(dotMat)):
        for j in range(i + 1, len(dotMat)):
            # solve vanishing gradient Problem by ignoring them :)
            if (np.linalg.norm(A[i]) * np.linalg.norm(A[j])) == 0:
                dotMat[i, j] = np.NaN
            else:
                dotMat[i, j] = np.dot(A[i], A[j]) / (np.linalg.norm(A[i]) * np.linalg.norm(A[j]))
    return dotMat


def epoch_upper_triangles(SampleGrads):
    # input of shape num_epochs x Batchsize x num_parameters
    # computes pairwise scalar Products of all Gradients in a batch
    # as diagonal and redundant values will not be needed, only use the upper triangle of scalarProduct Matrix(dotMat)
    dotMats = np.zeros((SampleGrads.shape[0], SampleGrads.shape[1], SampleGrads.shape[1]))
    for i in range(SampleGrads.shape[0]):
        dotMats[i] = sampleGrad_scalarprodMatrix(SampleGrads[i])

    upper_indices = np.triu_indices(SampleGrads.shape[1], 1)
    length_upper_tri = len(dotMats[0][upper_indices])
    # print(length_upper_tri)
    upper_triangles = np.zeros((SampleGrads.shape[0], length_upper_tri))

    for i in range(SampleGrads.shape[0]):
        upper_triangles[i] = dotMats[i][upper_indices]
    return upper_triangles


# consider giving bins as input
def upper_triangles_to_histograms(upper_triangles, lower_bound, upper_bound, bin_count):
    # input of shape epochs x length_upper_triangles
    # upper_triangles is output of epoch_upper_triangles function
    # lower and upper bounds are edges of later linspace
    # bincount depends on the instrument you want to use to sonify, use Instrument_dict["bin_count"] as input
    bins = np.linspace(lower_bound, upper_bound ,bin_count+1)
    hists = np.zeros((upper_triangles.shape[0], bin_count))
    for i in range(upper_triangles.shape[0]):
        hist = np.histogram(upper_triangles[i],bins)[0]
        hists[i] = hist
    return hists


def Histogram_to_Track_OneInstrument(hists, Instrument_dict, note_length=4, channel=1):
    # Function takes inputs:
    # np.ndarray of ints: hists: Calculated histograms of form epoch x histogram bins. best computed by histograms_allepochs_samplegrads function
    # or upper_triangles_to_histogram function or difference_consecutive_histograms function.
    # dictionary: Instrument_dict: One of the preselected Dictionaries with two Instruments e.g.
    # int: note_length, to change speed/duration of the output Track. default are quarter notes. posstible range range from 1-16

    # Function gives mingus Track object as output

    inst = MidiInstrument()
    inst.instrument_nr = Instrument_dict["instrument1_nr"]
    t1 = Track(inst)

    hists_max = np.nanmax(hists)
    print("hist_max:{}".format(hists_max))
    relative_velocities = interpolate.interp1d((0, hists_max), (0, 120))
    for i in range(hists.shape[0]):
        nc_inst = NoteContainer()
        for j, key in enumerate(Instrument_dict["instrument1"]):
            if np.isnan(hists[i, j]):
                # print("key: {}, velocity: {}".format(key,int(relative_velocities(hists[i,j]))))
                continue
            else:
                nc_inst.add_note(Note(key, velocity=int(relative_velocities(hists[i, j])), channel=channel))
        t1.add_notes(nc_inst, note_length)

    return t1


def Histogram_to_Track_OneInstrument_logvelocity(hists, Instrument_dict, note_length=4, channel=1):
    # Function takes inputs:
    # np.ndarray of ints: hists: Calculated histograms of form epoch x histogram bins. best computed by histograms_allepochs_samplegrads function
    # or upper_triangles_to_histogram function or difference_consecutive_histograms function.
    # dictionary: Instrument_dict: One of the preselected Dictionaries with two Instruments e.g.
    # int: note_length, to change speed/duration of the output Track. default are quarter notes. range from 1-16

    # Function gives mingus Composition object as output

    inst = MidiInstrument()
    inst.instrument_nr = Instrument_dict["instrument1_nr"]
    t1 = Track(inst)
    hists_NaN = Replace_0_with_NaN_hists(hists)
    hists_max = ceil_decimal(np.nanmax(hists_NaN), 1)
    hists_min = np.nanmin(hists_NaN)
    min_ = hists_min / 10
    print(hists_min)
    steps = int(np.log10(hists_max / min_))
    x_vals = np.logspace(np.log10(min_), np.log10(hists_max), steps)
    y_vals = np.linspace(0, 120, steps)
    print(x_vals)
    print(y_vals)

    print("hist_max:{}".format(hists_max))
    relative_velocities = interpolate.interp1d(x_vals, y_vals, fill_value=(0, 120))
    for i in range(hists.shape[0]):
        # print(i)
        nc_inst = NoteContainer()
        for j, key in enumerate(Instrument_dict["instrument1"]):
            if np.isnan(hists_NaN[i, j]):
                # print("key: {}, velocity: {}".format(key,int(relative_velocities(hists[i,j]))))
                continue
            else:
                print(hists_NaN[i, j])
                nc_inst.add_note(Note(key, velocity=int(relative_velocities(hists_NaN[i, j])), channel=channel))
        t1.add_notes(nc_inst, note_length)

    return t1


def save_scalarProd_and_loss_logscaled(upper_triangles, bins, train_loss, val_loss, checkpoints=[], dir_="hist_loss/",
                                       file="hist_loss_plot.jpg"):
    # takes output of epoch_upper_triangles as input for upper_triangles
    # bins is np.linspace(0,1,Instrument_dict["bin_count"])) or np.linspace(-1,1,Instrument_dict["bin_count"])
    # shape[0] of train_loss and val_loss need to be the sampe as upper_triangles.shape[0]

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 20})
    # initalize values for histograms
    tri_hist = upper_triangles_to_histograms(upper_triangles, int(bins[0]), int(bins[-1]), len(bins))
    max_ = np.max(tri_hist)
    print(max_)

    # initialize values for Loss
    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)

    bins = np.linspace(bins[0], bins[-1], len(bins) + 1)
    for i in range(upper_triangles.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
        ax1.hist(upper_triangles[i], bins)
        ax1.set_ylim([0, max_])
        ax1.set_title('ScalarProduct Histograms')
        # ax1.set_xlabel("Epoch",fontfamily = "serif", fontsize ='medium')
        ax1.set_ylabel("Number of Scalar Products", fontfamily="serif", fontsize='medium')
        ax1.text(0.8, 0.95, "Epoch: {}, Batch: {}".format(checkpoints[i][0], checkpoints[i][1]),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize='medium')

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        # ax2.set_xlabel("Epoch",fontfamily ="serif", fontsize ='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file


def save_scalarProd_and_loss_heatmap(grads, bins, train_loss, val_loss, checkpoints, dir_="heatmap_gradslogabs_loss/",
                                     file="heatmap_loss_plot.jpg"):
    # takes output of epoch_upper_triangles as input for upper_triangles
    # bins is np.linspace(0,1,Instrument_dict["bin_count"])) or np.linspace(-1,1,Instrument_dict["bin_count"])
    # shape[0] of train_loss and val_loss need to be the sampe as upper_triangles.shape[0]

    # saves images of heatmap and loss ins supplied directory
    # output are dir_ and file of images, which need to be used as input for build_video_fps_adjusted
    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 18})
    # initalize values for histograms
    upper_triangles = epoch_upper_triangles(grads)
    upper_triangles_hists = upper_triangles_to_histograms(upper_triangles, bins[0], bins[-1], 20)

    max_ = np.max(upper_triangles_hists)
    img = np.zeros((upper_triangles_hists.shape))
    # initialize values for Loss
    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)

    y_axis = np.linspace(0, len(bins) - 1, len(bins))

    # print(bins)
    for i in range(upper_triangles_hists.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 7), gridspec_kw={'width_ratios': [1.5, 1]})
        img[i] = upper_triangles_hists[i]
        # print(img[i].shape)
        im = ax1.imshow(img.T, cmap='hot', vmax=max_, aspect="auto")
        ax1.set_ylim([0, len(bins) - 1])
        ytick_labels = list(np.round(bins[::3], decimals=2))
        ytick_labels = list(map(str, ytick_labels))
        ax1.set_yticks(y_axis[::3])
        # ax1.set_ylim(int(bins[0]))
        ax1.set_yticklabels(ytick_labels)
        ax1.set_title("Heatmap of Scalar Products Histograms")
        ax1.set_xlabel("Checkpoints", fontfamily="serif", fontsize='medium')
        # cax = ax1.axes([0.91, 0.2, 0.05, 0.6])
        # divider = make_axes_locatable(ax1)
        plt.colorbar(im, ax=ax1)

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file


def convert_midi_to_mp3(file):
    # input is a midi file that is converted to a mp3 file. This is required to later stich together the audio and the video file
    if platform.system() == "Windows":
#        if 'ipykernel_launcher.py' in sys.argv[0]:
 #           command = "MuseScore3 " + file + " -o " + file[:-4] + ".mp3"
 #           output = !{command}
 #           print(output)
 #       else:
        subprocess.run("MuseScore3 " + file + " -o " + file[:-4] + ".mp3", shell=True, capture_output=True)
    else:
#        if 'ipykernel_launcher.py' in sys.argv[0]:
#            command = "mscore " + file + " -o " + file[:-4] + ".mp3"
#            output = !{command}
#        else:
        subprocess.run("mscore " + file + " -o " + file[:-4] + ".mp3", shell=True, capture_output=True)

    return file[:-4] + ".mp3"


# recommended output_file format: .flv
def merge_video_audio(video_file, audio_file, output_file):
    # merges the supplied video_file and audio_file into a new output_file video
#   if 'ipykernel_launcher.py' in sys.argv[0]:
#        command = "ffmpeg -i " + video_file + " -i " + audio_file + " -c:v copy -c:a aac " + output_file
#        cmd_output = !{command}
#        print(cmd_output)
#    else:
    subprocess.run("ffmpeg -i " + video_file + " -i " + audio_file + " -c:v copy -c:a aac " + output_file,
                       shell=True, capture_output=True)

    return output_file


def plot_loss_axes(checkpoints,ax):
    A, epoch =get_first_checkpoint_idx_per_epoch(checkpoints)
    idx = get_small_dist_location(A)
    bool_indices = bool_idx_Array(epoch, idx)
    A = np.array(A)
    epoch = np.array(epoch)
    ax.set_xlabel("Checkpoints")
    ax.set_xlim(0,len(checkpoints)-1)

    ax2 = ax.twiny()
    ax2.set_xlabel("Epoch")
    ax2.set_xlim(0, len(checkpoints)-1)

    ax2.set_xticks(A[bool_indices ])
    ax2.set_xticklabels(map(str,epoch[bool_indices]), fontsize="x-small")


def save_heatmap_imgs(hists, dir_="heatmap_imgs/", file="heatmap.jpg"):
    # takes sample hists as input either from histograms_allepochs_samplegrads or from upper_triangles_to_histograms
    # needs to be called before build_video_fps_adjusted, output dir_ and file can be used as input for build_video_fps_adjusted
    img = np.zeros((hists.shape))
    for i in range(hists.shape[0]):
        max_ = np.max(hists)
        img[i] = hists[i]
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.imshow(img.T, cmap='hot', vmax=max_)
        ytick_labels = [0, 0.25, 0.5, 0.75, 1]
        ytick_labels = list(map(str, yticks))
        ax.set_yticks([0, 4, 9, 14, 19])
        ax.set_yticklabels(ytick_labels)
        cax = plt.axes([0.91, 0.2, 0.05, 0.6])
        plt.colorbar(cax=cax)
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file



def histograms_allepochs_samplegrads(Grads_Epoch, lower_bound, upper_bound, bin_count=20):
    # expects Grads_Epoch of shape epoch x Batchsize x num_parameters as input
    # calculates average Gradient of Batches and turns them into histograms
    # meant to be a helper function for save_grads_and_loss and does not need to be called directly

    bins = np.linspace(lower_bound, upper_bound, bin_count + 1)
    histograms = np.zeros((Grads_Epoch.shape[0], bin_count))
    for i in range(Grads_Epoch.shape[0]):
        histograms[i] = np.histogram(np.nanmean(Grads_Epoch[i], axis=0), bins)[0]
    return histograms




def save_grads_and_loss(SampleGrads, bins, train_loss, val_loss, checkpoints, dir_="hist_grad_loss/",
                        file="hist_grad_loss_plot.jpg"):
    # takes sample grads as input(SampleGrads after transform through logabs also eligible)
    # bins  best computed by find_hist_edges for untransformed gradients,
    # shape[0] of train_loss and val_loss need to be the sampe as SampleGrads.shape[0]

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 15})

    # initalize values for histogtrams
    max_bin = int(np.max(histograms_allepochs_samplegrads(SampleGrads, bins[0], bins[-1], bin_count=len(bins))))
    max_bin_digits = len(str(max_bin))

    # initialize values for Loss
    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)

    bins = np.linspace(bins[0], bins[-1], len(bins) + 1)
    for i in range(SampleGrads.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
        ax1.hist(SampleGrads[i].mean(axis=0), bins)
        # get a good upper bound for y axis
        ax1.set_ylim([0, math.ceil(max_bin / 10 ** (max_bin_digits - 1)) * 10 ** (max_bin_digits - 1)])
        ax1.set_xlim([bins[0], bins[-1]])
        ax1.set_title('Sample Gradient Histograms')
        # ax1.set_xlabel("Epoch",fontfamily = "serif", fontsize ='medium')
        ax1.text(0.8, 0.95, "Epoch: {}, Batch: {}".format(checkpoints[i][0], checkpoints[i][1]),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize='medium')
        ax1.set_ylabel("Number of Gradient Elements", fontfamily="serif", fontsize='medium')
        ax1.set_xlabel("Gradient Element Value", fontfamily="serif", fontsize='medium')

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file


def save_grads_and_loss_logscaled(SampleGrads, bins, train_loss, val_loss, checkpoints=[], dir_="hist_grad_loss/",
                                  file="hist_grad_loss_plot.jpg"):
    # takes sample grads as input(SampleGrads after transform through logabs also eligible)
    # bins  best computed by find_hist_edges for untransformed gradients,
    # shape[0] of train_loss and val_loss need to be the sampe as SampleGrads.shape[0]

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 20})

    # initalize values for histogtrams
    max_bin = int(np.nanmax(histograms_allepochs_samplegrads(SampleGrads, bins[0], bins[-1], bin_count=len(bins))))
    max_bin_digits = len(str(max_bin))
    print(max_bin)

    # initialize values for Loss
    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)
    bins = np.linspace(bins[0], bins[-1], len(bins) + 1)

    for i in range(SampleGrads.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
        ax1.hist(np.nanmean(SampleGrads[i], axis=0), bins)
        # get a good upper bound for y axis
        ax1.set_ylim([0, math.ceil(max_bin / 10 ** (max_bin_digits - 1)) * 10 ** (max_bin_digits - 1)])
        ax1.set_xlim([bins[0], bins[-1]])
        ax1.set_title('Sample Gradient Histograms')
        ax1.text(0.8, 0.95, "Epoch: {}, Batch: {}".format(checkpoints[i][0], checkpoints[i][1]),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize='medium')
        # ax1.set_xlabel("Epoch",fontfamily = "serif", fontsize ='medium')
        ax1.set_ylabel("Number of Gradient Elements", fontfamily="serif", fontsize='medium')
        ax1.set_xlabel("Gradient Element Value", fontfamily="serif", fontsize='medium')

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file


def build_video_fps_adjusted(dir_="histograms/", file="hist.jpg", out="out.flv", BPM=120, note_length=4):
    output = ""
    # does not get and data, only name of directory, and file name
    # stichtes images in folder into a video
    # fps of video can be changed through note_length, which can take values from 1-16, with 1 being the lowest fps and 16 the highest
    # ! same note length has to be used as in the to_Track function(default is note_length=4 in all to_Track functions)

    BPS = BPM / 60
    fps = int(BPS * note_length / 4)
    fps = str(fps)
    # support normal python files and jupyter notebooks
#    if 'ipykernel_launcher.py' in sys.argv[0]:
#        # print("1")
#        command = 'ffmpeg -r ' + fps + ' -i ' + dir_ + '%d_' + file + ' -c:v libx264 -vf "fps=' + fps + ' ,format=yuv420p" ' + dir_ + out
#        output = !{command}
#        print(output)
#    else:
        # print("2")
    subprocess.run(
            'ffmpeg -r ' + fps + ' -i ' + dir_ + '%d_' + file + ' -c:v libx264 -vf "fps=' + fps + ' ,format=yuv420p" ' + dir_ + out,
            shell=True, capture_output=True)

    return out


def Replace_0_with_NaN_hists(A):
    # array of shape checkpoints x data(e.g. gammas)
    A = copy.deepcopy(A)
    for i in range(A.shape[0]):
        A[i][np.argwhere(A[i] == 0.0)] = np.NaN

        # A[i,idx:] = np.NaN
    return A

def ceil_decimal(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def find_hist_bins(SampleGrads, bin_count=20):
    # input of format epoch x Batchsize x num_parameters. Only to be used for normal, untransformed gradients
    # output are the bins which caputre at least 95% of the values (basically removes extreme outliers which would mess up)
    # the x-axis scaling of the histograms later
    mean0, var0 = get_mean_and_variance(SampleGrads)
    SampleGrads_flat = np.nanmean(SampleGrads, axis=1).flatten()

    # get at least 95% of data into Histogram
    count = 2

    bins = np.linspace(-var0, var0, bin_count)
    print("mean: {}, var: {}".format(mean0, var0))
    while sum(np.histogram(SampleGrads_flat, bins)[0]) / SampleGrads_flat.shape[0] < 0.95:
        print(sum(np.histogram(SampleGrads_flat, bins)[0]) / SampleGrads_flat.shape[0])
        bins = np.linspace(-count * var0, count * var0, bin_count)
        print(bins[0], bins[-1])
        count += 1

    return bins


def max_hist_values(npHists, logy=False):
    max_bin_heights = 0
    print("npHists shape0: {}".format(npHists.shape[0]))
    if logy == True:
        for i in range(npHists.shape[0]):
            print("i max_hist_values: {}".format(i))
            if max_bin_heights < np.nansum(logabs_transform(npHists[i])):
                max_bin_heights = np.nansum(logabs_transform(npHists[i]))

    else:
        for i in range(npHists.shape[0]):
            if max_bin_heights < sum(npHists[i]):
                max_bin_heights = sum(npHists[i])
    return int(np.ceil(max_bin_heights))


def save_grads_and_loss_logscaled(SampleGrads, bins, train_loss, val_loss, checkpoints=[], dir_="hist_grad_loss/",
                                  file="hist_grad_loss_plot.jpg"):
    # takes sample grads as input(SampleGrads after transform through logabs also eligible)
    # bins  best computed by find_hist_edges for untransformed gradients,
    # shape[0] of train_loss and val_loss need to be the sampe as SampleGrads.shape[0]

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 20})

    # initalize values for histogtrams
    max_bin = int(np.nanmax(histograms_allepochs_samplegrads(SampleGrads, bins[0], bins[-1], bin_count=len(bins))))
    max_bin_digits = len(str(max_bin))
    print(max_bin)

    # initialize values for Loss
    max_train = np.nanmax(train_loss)
    max_val = np.nanmax(val_loss)
    max_total = max(max_train, max_val)
    bins = np.linspace(bins[0], bins[-1], len(bins) + 1)

    for i in range(SampleGrads.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
        ax1.hist(np.nanmean(SampleGrads[i], axis=0), bins)
        # get a good upper bound for y axis
        ax1.set_ylim([0, math.ceil(max_bin / 10 ** (max_bin_digits - 1)) * 10 ** (max_bin_digits - 1)])
        ax1.set_xlim([bins[0], bins[-1]])
        ax1.set_title('Sample Gradient Histograms')
        ax1.text(0.8, 0.95, "Epoch: {}, Batch: {}".format(checkpoints[i][0], checkpoints[i][1]),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize='medium')
        # ax1.set_xlabel("Epoch",fontfamily = "serif", fontsize ='medium')
        ax1.set_ylabel("Number of Gradient Elements", fontfamily="serif", fontsize='medium')
        ax1.set_xlabel("Gradient Element Value", fontfamily="serif", fontsize='medium')

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file


def get_first_checkpoint_idx_per_epoch(checkpoints):
    A = [0]
    epoch = [0]
    current_epoch = 0
    for i in range(len(checkpoints)):
        if checkpoints[i][0] != current_epoch:
            A.append(i)
            epoch.append(checkpoints[i][0])
            current_epoch = checkpoints[i][0]

    return A, epoch

def get_small_dist_location(A):
    for i in range(len(A)-1):
        if A[i+1] - A[i] <= 2:
            return i


def bool_idx_Array(A, idx):
    Bool_array = [True for i in A]

    for i in range(len(A)):
        if i <= idx:
            continue
        elif np.mod(A[i], 10) != 0:
            Bool_array[i] = False
    Bool_array[9] = False
    return Bool_array


def get_mean_and_variance(SampleGrads):
    # input of Form Batchsize x num_parameters
    means, variances = scipy.stats.describe(SampleGrads, nan_policy="omit")[2:4]
    mean_comb = np.nanmean(means)
    var_comb = np.nanvar(variances)
    return mean_comb, var_comb


def pan_stereo_sound(t1, t2, output="output.mp3", file1="t1.mid", file2="t2.mid"):
    # input are two Mingus Track objects t1 and t2
    # output is a mingus composition where track t1 is only played on the left ear and track t2 is only played on the right
    # ear (or vice versa if change_order is set to True)
    mingus.midi.midi_file_out.write_Track(file1, t1)
    mingus.midi.midi_file_out.write_Track(file2, t2)
    t1_mp3 = convert_midi_to_mp3(file1)
    t2_mp3 = convert_midi_to_mp3(file2)
    t1_AudioSeg = AudioSegment.from_mp3(t1_mp3)
    t2_AudioSeg = AudioSegment.from_mp3(t2_mp3)
    t1_AudioSeg = t1_AudioSeg.apply_gain_stereo(0, -120)
    t2_AudioSeg = t2_AudioSeg.apply_gain_stereo(-120, -8)
    t_overlaid = t1_AudioSeg.overlay(t2_AudioSeg)
    t_overlaid.export(output, format="mp3")
    return output

def get_mean_variance_wholeData(SampleGrads):
    mean,var = scipy.stats.describe(np.nanmean(SampleGrads,axis=1).flatten(), nan_policy ="omit")[2:4]
    return mean,var

def logabs_transform(A):
    #
    A_logabs = np.log(np.abs(A))
    A_logabs[A_logabs == -inf] = np.NAN
    return A_logabs


# change function do directly get bins as input?
def histograms_allepochs_samplegrads_heatmap(Grads_Epoch, lower_bound, upper_bound, bin_count=20):
    # expects Grads_Epoch of shape epoch x Batchsize x num_parameters as input
    # calculates average Gradient of Batches and turns them into histograms
    # meant to be a helper function for save_grads_and_loss and does not need to be called directly

    bins = np.linspace(lower_bound, upper_bound, bin_count + 1)
    histograms_bins = np.zeros((Grads_Epoch.shape[0], bin_count))
    histograms_binedges = np.zeros((Grads_Epoch.shape[0], bin_count + 1))
    for i in range(Grads_Epoch.shape[0]):
        histograms_bins[i], histograms_binedges[i] = np.histogram(np.nanmean(Grads_Epoch[i], axis=0), bins)

    return histograms_bins, histograms_binedges[0]


def save_grads_and_loss_heatmap(grads, bins, train_loss, val_loss, checkpoints, dir_="heatmap_gradslogabs_loss/",
                                file="heatmap_loss_plot.jpg"):
    # takes output of epoch_upper_triangles as input for upper_triangles
    # bins is np.linspace(0,1,Instrument_dict["bin_count"])) or np.linspace(-1,1,Instrument_dict["bin_count"])
    # shape[0] of train_loss and val_loss need to be the sampe as upper_triangles.shape[0]

    # saves images of heatmap and loss ins supplied directory
    # output are dir_ and file of images, which need to be used as input for build_video_fps_adjusted
    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 18})
    # initalize values for histograms

    grads_hists, grads_binedges = histograms_allepochs_samplegrads_heatmap(grads, bins[0], bins[-1],
                                                                           bin_count=len(bins))
    max_ = np.max(grads_hists)
    img = np.zeros((grads_hists.shape))
    # initialize values for Loss
    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)

    y_axis = np.linspace(0, len(bins) - 1, len(bins))

    # print(bins)
    for i in range(grads_hists.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 7), gridspec_kw={'width_ratios': [1.5, 1]})
        img[i] = grads_hists[i]
        # print(img[i].shape)
        im = ax1.imshow(img.T, cmap='hot', vmax=max_, aspect="auto")
        ax1.set_ylim([0, len(bins) - 1])
        ytick_labels = list(map(int, bins[::3]))
        ytick_labels = list(map(str, ytick_labels))
        ax1.set_yticks(y_axis[::3])
        # ax1.set_ylim(int(bins[0]))
        ax1.set_yticklabels(ytick_labels)
        ax1.set_title("Heatmap of Histograms")
        ax1.set_xlabel("Checkpoints", fontfamily="serif", fontsize='medium')
        # cax = ax1.axes([0.91, 0.2, 0.05, 0.6])
        # divider = make_axes_locatable(ax1)
        plt.colorbar(im, ax=ax1)

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file


def sonify_loss(loss_val, Instrument_dict, note_length=4, channel=3):
    inst = MidiInstrument()
    inst.instrument_nr = Instrument_dict["instrument1_nr"]
    t1 = Track(inst)

    # partition loss values in as many bins as there are hist_bins in the used histogram.
    loss_range = np.linspace(np.nanmin(loss_val), np.nanmax(loss_val), Instrument_dict["bin_count"])
    notes = np.array(list(Instrument_dict["instrument1"]))
    for i in range(loss_val.shape[0]):
        nc = NoteContainer()
        closest_idx = np.argmin(np.abs(loss_range - loss_val[i]))
        if np.isnan(loss_val[i]):
            break
        else:
            nc.add_note(Note(notes[closest_idx], velocity=50, channel=channel))
            t1.add_notes(nc, note_length)
    return t1


def Histograms_to_Track_TwoInstruments(hists, Instrument_dict1, Instrument_dict2, note_length=4,
                                       switch_instruments=False):
    # Function takes inputs:
    # np.ndarray of ints: hists: Calculated histograms of form epoch x histogram bins. best computed by difference_consecutive_histograms function
    # dictionary: Instrument_dict: One of the preselected Dictionaries with two Instruments e.g.
    # int: note_length, to change speed/duration of the output Track. default are quarter notes. range from 1-16

    # Function gives mingus Composition object as output

    if (hists.shape[1] != Instrument_dict1["bin_count"] + Instrument_dict2["bin_count"]):
        print("Fehler")
    # initalize piano
    if switch_instruments == False:

        inst1 = MidiInstrument()
        inst1.instrument_nr = Instrument_dict1["instrument1_nr"]
        inst1_notes = Instrument_dict1["instrument1"]
        t1 = Track(inst1)

        inst2 = MidiInstrument()
        inst2.instrument_nr = Instrument_dict2["instrument1_nr"]
        t2 = Track(inst2)
        inst2_notes = Instrument_dict2["instrument1"]

    elif switch_instruments:

        inst1 = MidiInstrument()
        inst1.instrument_nr = Instrument_dict2["instrument1_nr"]
        inst1_notes = Instruments_dict["instrument1"]
        t1 = Track(inst1)

        inst2 = MidiInstrument()
        inst2.instrument_nr = Instruments_dict["instrument1_nr"]
        t2 = Track(inst2)
        inst2_notes = Instruments_dict["instrument1"]

    keys = np.array(list(Instrument_dict1["instrument1"]))
    keys = np.concatenate((keys, np.array(list(Instrument_dict2["instrument1"]))))
    # print("len keys: {}".format(len(keys)))
    # print(keys)

    hists_max = np.max(hists)
    relative_velocities = interpolate.interp1d((0, hists_max), (0, 120))
    for i in range(hists.shape[0]):
        nc_inst1 = NoteContainer()
        nc_inst2 = NoteContainer()
        for j in range(0, hists.shape[1] - 1):
            if j <= (hists.shape[1] / 2 - 1):
                nc_inst1.add_note(Note(keys[j], velocity=int(relative_velocities(hists[i, j])), channel=1))
            else:
                nc_inst2.add_note(Note(keys[j], velocity=int(relative_velocities(hists[i, j])), channel=2))
        t1.add_notes(nc_inst1, note_length)
        t2.add_notes(nc_inst2, note_length)

    return t1, t2


def save_eig_and_loss_logscaled(eigenvals, bin_count, train_loss, val_loss, checkpoints=[], dir_="hist_eig_loss/",
                                file="hist_eig_loss_plot.jpg"):
    # takes sample grads as input(SampleGrads after transform through logabs also eligible)
    # bins  best computed by find_hist_edges for untransformed gradients,
    # shape[0] of train_loss and val_loss need to be the sampe as SampleGrads.shape[0]

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 20})

    bins = np.linspace(0, np.nanmax(eigenvals), 20)
    # initalize values for histogtrams
    max_bin = int(np.nanmax(histograms_epoch(eigenvals, bins[0], bins[-1], len(bins))))
    max_bin_digits = len(str(max_bin))
    print(max_bin)

    # initialize values for Loss
    max_train = np.nanmax(train_loss)
    max_val = np.nanmax(val_loss)
    max_total = max(max_train, max_val)
    bins = np.linspace(bins[0], bins[-1], len(bins) + 1)

    for i in range(eigenvals.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
        ax1.hist(eigenvals[i], bins)
        # get a good upper bound for y axis
        ax1.set_ylim([0, math.ceil(max_bin / 10 ** (max_bin_digits - 1)) * 10 ** (max_bin_digits - 1)])
        ax1.set_xlim([bins[0], bins[-1]])
        ax1.set_title('Eigenvalue Histograms')
        ax1.text(0.8, 0.95, "Epoch: {}, Batch: {}".format(checkpoints[i][0], checkpoints[i][1]),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize='medium')
        # ax1.set_xlabel("Epoch",fontfamily = "serif", fontsize ='medium')
        ax1.set_ylabel("Number of Eigenvalues", fontfamily="serif", fontsize='medium')
        ax1.set_xlabel("Eigenvalue", fontfamily="serif", fontsize='medium')

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file


def save_eig_and_loss_logscaled_instrument_separated(eigenvals, bins1, bins2, train_loss, val_loss, checkpoints=[],
                                 dir_="hist_eig_separated_loss/", file="hist_eig_separated_loss_plot.jpg"):
    # takes sample grads as input(SampleGrads after transform through logabs also eligible)
    # bins  best computed by find_hist_edges for untransformed gradients,
    # shape[0] of train_loss and val_loss need to be the sampe as SampleGrads.shape[0]

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 18})

    # initalize values for histogtrams
    max_bin1 = int(np.nanmax(histograms_epoch(eigenvals, bins1[0], bins1[-1], len(bins1))))
    max_bin_digits1 = len(str(max_bin1))
    print(max_bin1)

    max_bin2 = int(np.nanmax(histograms_epoch(eigenvals, bins2[0], bins2[-1], len(bins2))))
    max_bin_digits2 = len(str(max_bin2))
    print(max_bin2)

    # initialize values for Loss
    max_train = np.nanmax(train_loss)
    max_val = np.nanmax(val_loss)
    max_total = max(max_train, max_val)

    bins1 = np.linspace(bins1[0], bins1[-1], len(bins1) + 1)
    bins2 = np.linspace(bins2[0], bins2[-1], len(bins2) + 1)

    for i in range(eigenvals.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 10))

        ax1.hist(eigenvals[i], bins1)
        # get a good upper bound for y axis
        ax1.set_ylim([0, math.ceil(max_bin1 / 10 ** (max_bin_digits1 - 1)) * 10 ** (max_bin_digits1 - 1)])
        ax1.set_xlim([bins1[0], bins1[-1]])
        # ax1.set_title('Eigenvalue Histograms for eigenvalues between 0 and 1')
        ax1.text(0.7, 0.95, "Epoch: {}, Batch: {}".format(checkpoints[i][0], checkpoints[i][1]),
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes, fontsize='medium')
        # ax1.set_xlabel("Epoch",fontfamily = "serif", fontsize ='medium')
        ax1.set_ylabel("Number of Eigenvalues", fontfamily="serif", fontsize='medium')
        ax1.set_xlabel("Eigenvalue", fontfamily="serif", fontsize='medium')

        ax2.hist(eigenvals[i], bins2)
        ax2.set_ylim([1, math.ceil(max_bin2 / 10 ** (max_bin_digits2 - 1)) * 10 ** (max_bin_digits2 - 1)])
        ax2.set_yscale("log")
        # get a good upper bound for y axis
        # print(math.ceil(max_bin2/10**(max_bin_digits2-1))*10**(max_bin_digits2-1))

        ax1.set_title('separated Eigenvalue Histograms for ranges [0,1] and [1,12] ', pad=38, loc="left")
        ax2.set_xlim([bins2[0], bins2[-1]])
        # ax2.set_title('Eigenvalue Histograms for eigenvalues between 0 and 1')
        # ax1.set_xlabel("Epoch",fontfamily = "serif", fontsize ='medium')
        ax2.set_ylabel("Number of Eigenvalues", fontfamily="serif", fontsize='medium')
        ax2.set_xlabel("Eigenvalue", fontfamily="serif", fontsize='medium')

        # plot loss of current epoch
        bins3 = np.linspace(0, i, i + 1)
        ax3.plot(bins3, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax3.plot(bins3, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax3.set_xlim([-1, train_loss.shape[0] + 1])
        ax3.set_ylim([-0.3, max_total + 1])
        ax3.set_title('Training and Validation Loss')
        ax3.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax3.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax3)
        ax3.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        # plt.tight_layout()
        plt.close()

    return dir_, file


def Histogram_to_Track_OneInstrument_logvelocity3(hists, Instrument_dict, note_length=4, channel=1):
    # Function takes inputs:
    # np.ndarray of ints: hists: Calculated histograms of form epoch x histogram bins. best computed by histograms_allepochs_samplegrads function
    # or upper_triangles_to_histogram function or difference_consecutive_histograms function.
    # dictionary: Instrument_dict: One of the preselected Dictionaries with two Instruments e.g.
    # int: note_length, to change speed/duration of the output Track. default are quarter notes. range from 1-16

    # Function gives mingus Composition object as output

    inst = MidiInstrument()
    inst.instrument_nr = Instrument_dict["instrument1_nr"]
    t1 = Track(inst)
    hists_NaN = Replace_0_with_NaN_hists(hists)
    x_vals = np.geomspace(1, np.nanmax(hists), 70)
    y_vals = np.linspace(50, 120, 70)
    print(x_vals)
    print(y_vals)

    relative_velocities = interpolate.interp1d(x_vals, y_vals)
    for i in range(hists.shape[0]):
        # print(i)
        nc_inst = NoteContainer()
        if all_empty(hists_NaN[i]):
            nc_inst.add_note(Note("A-4", velocity=0, channel=channel))
        else:
            for j, key in enumerate(Instrument_dict["instrument1"]):
                if np.isnan(hists_NaN[i, j]):
                    # print("key: {}, velocity: {}".format(key,int(relative_velocities(hists[i,j]))))
                    continue
                else:
                    print("i,j: {},{}".format(i, j))
                    print("bar heights: {}".format(hists_NaN[i, j]))
                    print("rel_vel: {}".format(int(relative_velocities(hists_NaN[i, j]))))

                    nc_inst.add_note(Note(key, velocity=int(relative_velocities(hists_NaN[i, j])), channel=channel))
        t1.add_notes(nc_inst, note_length)

    return t1

def top_k_eigenvalues(eigen_vals,k, num_checkpoints=100):
    top_k = np.zeros((num_checkpoints,k))
    for i in range(eigen_vals.shape[0]):
        top_k[i] = np.sort(eigen_vals[i])[-k:]
    return top_k


def save_top_eigenvals_and_loss_heatmap(top_eigenvals, bins, train_loss, val_loss, checkpoints,
                                        dir_="heatmap_eigenvals_loss/", file="heatmap_loss_plot.jpg"):
    # takes output of epoch_upper_triangles as input for upper_triangles
    # bins is np.linspace(0,1,Instrument_dict["bin_count"])) or np.linspace(-1,1,Instrument_dict["bin_count"])
    # shape[0] of train_loss and val_loss need to be the sampe as upper_triangles.shape[0]

    # saves images of heatmap and loss ins supplied directory
    # output are dir_ and file of images, which need to be used as input for build_video_fps_adjusted
    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 18})
    # initalize values for histograms

    eigenvals_hists = histograms_epoch(top_eigenvals, bins[0], bins[-1], len(bins))
    max_ = np.max(eigenvals_hists)
    img = np.zeros((eigenvals_hists.shape))
    # initialize values for Loss
    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)

    y_axis = np.linspace(0, len(bins) - 1, len(bins))

    # print(bins)
    for i in range(eigenvals_hists.shape[0]):
        # plot histogram of current epoch
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 7), gridspec_kw={'width_ratios': [1.5, 1]})
        img[i] = eigenvals_hists[i]
        # print(img[i].shape)
        im = ax1.imshow(img.T, cmap='hot', vmax=max_, aspect="auto")
        ax1.set_ylim([0, len(bins) - 1])
        ytick_labels = list(np.round(bins[::3], decimals=2))
        ytick_labels = list(map(str, ytick_labels))
        ax1.set_yticks(y_axis[::3])
        # ax1.set_ylim(int(bins[0]))
        ax1.set_yticklabels(ytick_labels)
        ax1.set_title("Heatmap of Scalar Products Histograms")
        ax1.set_xlabel("Checkpoints", fontfamily="serif", fontsize='medium')
        # cax = ax1.axes([0.91, 0.2, 0.05, 0.6])
        # divider = make_axes_locatable(ax1)
        plt.colorbar(im, ax=ax1)

        # plot loss of current epoch
        bins2 = np.linspace(0, i, i + 1)
        ax2.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        ax2.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        ax2.set_xlim([-1, train_loss.shape[0] + 1])
        ax2.set_ylim([-0.3, max_total + 1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        ax2.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, ax2)
        ax2.legend()
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file

def Replace_0_with_NaN(A):
    # array of shape checkpoints x data(e.g. gammas)
    A = copy.deepcopy(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            idx = np.argwhere(A[i,j] == 0.0)[0][0]
            A[i,j,idx:] = np.NaN
        #print(idx)
        #A[i,idx:] = np.NaN
    return A


def get_hists_unweighted(Lambdas, bins, n_checkpoints):
    # input of shape n_checkpoints

    A = np.zeros((n_checkpoints, len(bins) - 1))
    for i in range(Lambdas.shape[0]):
        A[i] = np.histogram(Lambdas[i], bins)[0]
    return A

def scatter_hist(x, y, ax, ax_histx, ax_histy, binsx, binsy, checkpoint, LambdaSNR=True, log=False):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([binsx[0], binsx[-1]])
    ax.set_ylim([binsy[0], binsy[-1]])
    ax_histx.text(1.15, 1.08, "Epoch: {}\n  Batch: {}".format(checkpoint[0], checkpoint[1]),
                  horizontalalignment='center',
                  verticalalignment='center', transform=ax.transAxes, fontsize='medium')

    histx = ax_histx.hist(x, bins=binsx, log=log)[0]
    histy = ax_histy.hist(y, bins=binsy, orientation='horizontal', log=log)[0]

    if LambdaSNR == True:
        # print("lambdaSNR")
        ax_histx.set_title("SNR of \u03BB")
        ax.set_ylabel("SNR(\u03BB{})".format(get_sub("k")), fontfamily="serif", fontsize='medium')
    else:
        # print("gammaSNR")
        ax_histx.set_title("SNR of \u03B3")
        ax.set_ylabel("SNR(\u03B3{})".format(get_sub("k")), fontfamily="serif", fontsize='medium')

    ax.set_xlabel("Value of Curvature \u03BB{}".format(get_sub("k")), fontfamily="serif", fontsize='medium')
    # now determine nice limits by hand:
    return histx, histy


def save_SNR_and_loss_logscaled(Data, bin_count, train_loss, val_loss, LambdaSNR=True, checkpoints=[], log=False,
                                dir_="hist_grad_loss/", file="hist_grad_loss_plot.jpg"):
    # Data must eigther be the Gammas(scalarproduct of eigenvektors and gradient) or Lambdas(scalarproduct of eigenvectors in Gram space w.r.t GGN)
    # eigenvectors are conjungate gradients of GGN

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 20})

    Data_NaN = Replace_0_with_NaN(Data)
    Data_NaN_mean = np.nanmean(Data_NaN, axis=1)
    Data_NaN_var = np.nanvar(Data_NaN, axis=1, ddof=1)
    SNR = Data_NaN_mean ** 2 / Data_NaN_var
    # print(Data_NaN_mean.shape)
    # print(SNR.shape)

    xmin, xmax = np.nanmin(Data_NaN_mean), np.nanmax(Data_NaN_mean)
    ymin, ymax = np.nanmin(SNR), np.nanmax(SNR)

    print("xmin: {}\nxmax: {}".format(xmin, xmax))
    print("ymin: {}\nymax: {}".format(ymin, ymax))

    binsx = np.logspace(np.log10(xmin), np.log10(xmax), bin_count + 1)
    binsy = np.logspace(np.log10(ymin), np.log10(ymax), bin_count + 1)

    histsx_ = get_hists_unweighted(Data_NaN_mean, binsx, len(checkpoints))
    histsy_ = get_hists_unweighted(SNR, binsy, len(checkpoints))

    max_histsx = np.nanmax(histsx_)
    max_histsy = np.nanmax(histsy_)

    # initialize values for Loss
    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)

    histsx = np.zeros((Data.shape[0], bin_count))
    histsy = np.zeros((Data.shape[0], bin_count))

    for i in range(Data.shape[0]):
        # print(i)
        fig = plt.figure(figsize=(20, 10))
        # fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=(7, 5))
        # width_ratios=(7, 2, 7), height_ratios=(2, 7),
        #              left=0.1, right=0.9, bottom=0.1, top=0.9,
        #              wspace=0.05, hspace=0.05)
        gs0 = gs[0].subgridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                                wspace=0.05, hspace=0.05)
        gs1 = gs[1].subgridspec(2, 1, height_ratios=(2, 7), wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs0[1, 0])
        ax_histx = fig.add_subplot(gs0[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs0[1, 1], sharey=ax)
        ax_histx.set_ylim([0, max_histsx])
        ax_histy.set_xlim([0, max_histsy])
        # gs[1,2].subgridspec(gs[1,2].get_position()[0],gs[1,2].get_position()[1], wspace = 0.5)

        histsx[i], histsy[i] = scatter_hist(Data_NaN_mean[i], SNR[i], ax, ax_histx, ax_histy, binsx, binsy,
                                            checkpoints[i], LambdaSNR=LambdaSNR, log=log)

        loss_plot = fig.add_subplot(gs1[1])
        bins2 = np.linspace(0, i, i + 1)
        loss_plot.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        loss_plot.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        loss_plot.set_xlim([-1, train_loss.shape[0] + 1])
        loss_plot.set_ylim([-0.3, max_total + 1])
        loss_plot.set_title('Training and Validation Loss')
        loss_plot.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        loss_plot.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, loss_plot)
        loss_plot.legend()

        """
        # plot loss of current epoch
        bins2 =  np.linspace(0,i,i+1)
        ax2.plot(bins2, np.nanmean(train_batch_loss[0:i+1],axis=1),"-",label="train_loss",linewidth=2.5)
        ax2.plot(bins2, np.nanmean(val_batch_loss[0:i+1],axis=1),"-",label="val_loss",linewidth=2.5)
        ax2.set_xlim([-1,train_batch_loss.shape[0]+1])
        ax2.set_ylim([-0.3,max_total+1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch",fontfamily ="serif", fontsize ='medium')
        ax2.set_ylabel("Loss",fontfamily = "serif" , fontsize ='medium')
        ax2.legend()
        """
        plt.savefig(dir_ + str(i) + "_" + file)

        # plt.tight_layout()
        plt.close()

    return dir_, file, histsx, histsy


def save_SNRGamma_and_loss_logscaled(Lambdas, Gammas, bin_count, train_loss, val_loss, LambdaSNR=False, checkpoints=[],
                                     ycutoff=1e-10, log=False, dir_="hist_grad_loss/", file="hist_grad_loss_plot.jpg"):
    # Data must eigther be the Gammas(scalarproduct of eigenvektors and gradient) or Lambdas(scalarproduct of eigenvectors in Gram space w.r.t GGN)
    # eigenvectors are conjungate gradients of GGN

    Path(dir_).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 20})

    Lambdas_NaN = Replace_0_with_NaN(Lambdas)
    Lambdas_NaN_mean = np.nanmean(Lambdas_NaN, axis=1)

    Gammas_NaN = Replace_0_with_NaN(Gammas)
    Gammas_NaN_mean = np.nanmean(Gammas_NaN, axis=1)
    Gammas_NaN_var = np.nanvar(Gammas_NaN, axis=1, ddof=1)
    SNR = Gammas_NaN_mean ** 2 / Gammas_NaN_var
    # print(Data_NaN_mean.shape)
    # print(SNR.shape)

    xmin, xmax = np.nanmin(Lambdas_NaN_mean), np.nanmax(Lambdas_NaN_mean)
    ymin, ymax = np.nanmin(SNR), np.nanmax(SNR)

    if ymin < ycutoff:
        ymin = ycutoff

    print("xmin: {}\nxmax: {}".format(xmin, xmax))
    print("ymin: {}\nymax: {}".format(ymin, ymax))

    binsx = np.logspace(np.log10(xmin), np.log10(xmax), bin_count + 1)
    binsy = np.logspace(np.log10(ymin), np.log10(ymax), bin_count + 1)

    histsx_ = get_hists_unweighted(Lambdas_NaN_mean, binsx, len(checkpoints))
    histsy_ = get_hists_unweighted(SNR, binsy, len(checkpoints))

    max_histsx = np.nanmax(histsx_)
    max_histsy = np.nanmax(histsy_)
    # print(max_histsx)
    # print(max_histsy)

    max_train = np.max(train_loss)
    max_val = np.max(val_loss)
    max_total = max(max_train, max_val)

    histsx = np.zeros((Lambdas.shape[0], bin_count))
    histsy = np.zeros((Gammas.shape[0], bin_count))
    # print(histsx.shape)
    # print(histsy.shape)
    for i in range(Lambdas.shape[0]):
        # print(i)
        fig = plt.figure(figsize=(20, 10))
        # fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=(7, 5))
        # width_ratios=(7, 2, 7), height_ratios=(2, 7),
        #              left=0.1, right=0.9, bottom=0.1, top=0.9,
        #              wspace=0.05, hspace=0.05)
        gs0 = gs[0].subgridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                                wspace=0.05, hspace=0.05)
        gs1 = gs[1].subgridspec(2, 1, height_ratios=(2, 7), wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs0[1, 0])
        ax_histx = fig.add_subplot(gs0[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs0[1, 1], sharey=ax)
        ax_histx.set_ylim([0, max_histsx])
        ax_histy.set_xlim([0, max_histsy])
        # gs[1,2].subgridspec(gs[1,2].get_position()[0],gs[1,2].get_position()[1], wspace = 0.5)

        histsx[i], histsy[i] = scatter_hist(Lambdas_NaN_mean[i], SNR[i], ax, ax_histx, ax_histy, binsx, binsy,
                                            checkpoints[i], LambdaSNR=LambdaSNR, log=log)

        loss_plot = fig.add_subplot(gs1[1])
        bins2 = np.linspace(0, i, i + 1)
        loss_plot.plot(bins2, train_loss[0:i + 1], "-", label="train_loss", linewidth=2.5)
        loss_plot.plot(bins2, val_loss[0:i + 1], "-", label="val_loss", linewidth=2.5)
        loss_plot.set_xlim([-1, train_loss.shape[0] + 1])
        loss_plot.set_ylim([-0.3, max_total + 1])
        loss_plot.set_title('Training and Validation Loss')
        loss_plot.set_xlabel("Epoch", fontfamily="serif", fontsize='medium')
        loss_plot.set_ylabel("Loss", fontfamily="serif", fontsize='medium')
        plot_loss_axes(checkpoints, loss_plot)
        loss_plot.legend()
        """
        # plot loss of current epoch
        bins2 =  np.linspace(0,i,i+1)
        ax2.plot(bins2, np.nanmean(train_batch_loss[0:i+1],axis=1),"-",label="train_loss",linewidth=2.5)
        ax2.plot(bins2, np.nanmean(val_batch_loss[0:i+1],axis=1),"-",label="val_loss",linewidth=2.5)
        ax2.set_xlim([-1,train_batch_loss.shape[0]+1])
        ax2.set_ylim([-0.3,max_total+1])
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel("Epoch",fontfamily ="serif", fontsize ='medium')
        ax2.set_ylabel("Loss",fontfamily = "serif" , fontsize ='medium')
        ax2.legend()
        """
        plt.savefig(dir_ + str(i) + "_" + file)
        plt.close()

    return dir_, file, histsx, histsy


superscript_map = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ",
    "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ᶦ", "j": "ʲ", "k": "ᵏ",
    "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "q": "۹", "r": "ʳ",
    "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
    "z": "ᶻ", "A": "ᴬ", "B": "ᴮ", "C": "ᶜ", "D": "ᴰ", "E": "ᴱ", "F": "ᶠ",
    "G": "ᴳ", "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ",
    "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "Q": "Q", "R": "ᴿ", "S": "ˢ", "T": "ᵀ",
    "U": "ᵁ", "V": "ⱽ", "W": "ᵂ", "X": "ˣ", "Y": "ʸ", "Z": "ᶻ", "+": "⁺",
    "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾"}

trans = str.maketrans(
    ''.join(superscript_map.keys()),
    ''.join(superscript_map.values()))

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)





import pandas as pd
import random
from sklearn import preprocessing
import numpy as np
import os
from scipy.signal import welch
from scipy.integrate import simps
from pylsl import StreamInlet, resolve_byprop
import utils
import random
import warnings

def getaccuracy(predictions, y):
    from sklearn import metrics

    predbin = []
    for pred in predictions:
        if pred[0] > pred[1]:
            predbin.append([1, 0])
        else:
            predbin.append([0, 1])

    return metrics.accuracy_score(y, np.array(predbin))*100


def modelfname(trainsongs,uid,folder="models/"):
    return folder + '_'.join(str(id) for id in trainsongs) + "_" + str(uid) + ".h5"


def train(X,Y,x,y,modelfile, dnnonly=True):

    dnnacc, dnnmodel = train_dnn(X,Y,x,y,modelfile)
    if dnnonly:
        dnnmodel.save(modelfile)
        return dnnacc

    akacc, akmodel = train_autokeras(X,Y,x,y,modelfile)

    print("DNN accuracy: {}\nAutoKeras accuracy: {}".format(dnnacc, akacc))

    if dnnacc > akacc:
        dnnmodel.save(modelfile)
        return dnnacc
    else:
        akmodel.save(modelfile)
        return akacc


def train_autokeras(X,Y,x,y,modelfile,max_trials=10,epochs=600):
    from sklearn import metrics
    import autokeras as ak
    from sklearn.preprocessing import MinMaxScaler

    clf = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=max_trials)
    clf.fit(X, Y, validation_data=(x, y), epochs=epochs)
    model = clf.export_model()
    model.save(modelfile)

    akpred = clf.predict(x)

    acc = getaccuracy(akpred, y)

    return acc, model


def train_dnn(X,Y,x,y,modelfile,epochs=600):
    from sklearn import metrics
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    n_features = X.shape[1]
    opt = tf.keras.optimizers.SGD(momentum=0.0, nesterov=False)
    model = Sequential()
    model.add(Dense(128, input_shape = (n_features,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(len(np.unique(Y)), activation='sigmoid'))
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X, Y, epochs=500, batch_size=32)

    preds = model.predict(x)
    acc = getaccuracy(preds, y)

    return acc, model



def train_rf(X,Y,x,y,modelfile,epochs=600):
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X, Y)
    pred = np.array(clf.predict(x))
    acc = metrics.accuracy_score(y, pred)*100

    return acc

def padlabels(arr):
    labels = np.zeros((len(arr), len(np.unique(arr))))
    i = 0
    for pred in arr:
        labels[i][int(pred)] = 1
        i+=1

    return labels

def find_beat(array, value):
    array = np.asarray(array)
    idx = [(np.abs(array - value)).argmin()]
    #idx.append(np.abs(array - value/2).argmin())
    return idx



def bandpower(data, sf, band, window_sec=0.5, relative=True):
    #https://raphaelvallat.com/bandpower.html
    band = np.asarray(band)
    low, high = band
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp



def bandpower_beat(data, sf, bpm, window_sec=None, relative=True):
    #https://raphaelvallat.com/bandpower.html

    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / bpm) * sf
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]

    idx_band = find_beat(freqs, bpm)

    bp = psd[idx_band]
    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp



def recordeeg(duration):
    warnings.filterwarnings('ignore')

    BUFFER_LENGTH = 5
    EPOCH_LENGTH = 1
    OVERLAP_LENGTH = 0.8
    SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    band_buffer = np.zeros((n_win_test, 4))
    musedata = []

    while True:
        eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * fs))
        musedata += eeg_data
        if len(musedata) > duration*fs:
            return musedata
            break

def getFeatures(duration=5, songid=None):
    return geteegsamples(recordeeg(duration), songid=songid)

def geteegsamples(musedata, iterations=100, songid=None):
    fs = 256
    n = 256
    k = 0
    channels = ["TP9", "AF7", "AF8", "TP10"]
    samples = []
    for m in range(iterations):
        raw_data = pd.DataFrame(data=musedata, columns=[
                                "TP9", "AF7", "AF8", "TP10", "Right AUX"])
        values = dict()
        if songid != None:
            values["songid"] = songid
        l = 0

        start = round((random.randint(10, 65) / 100) * len(raw_data))
        end = start + random.randint(start, len(raw_data))

        if end > len(raw_data) or end - start < 4*n:
            end = len(raw_data)
        raw_data = pd.DataFrame(data=musedata, columns=[
                                "TP9", "AF7", "AF8", "TP10", "Right AUX"]).reset_index().loc[start:end].reset_index()
        del raw_data["Right AUX"]
        data = raw_data.loc[:, channels[0]]
        eeg_bands = {'Delta': (0.001, 4),
                     'Theta': (4, 8),
                     'Alpha': (8, 12),
                     'Beta': (12, 30),
                     'Gamma': (30, 45)}
        beats = {
            # '100BPM': 100/60, We don't analyse the beats in this implementation (yet).....
            # '86BPM': 86/60
        }

        eeg_band_fft = dict()
        avgs = dict()
        channelavgs = []
        for band in eeg_bands:
            eeg_band_fft[band] = dict()
            for channel in channels:
                eeg_band_fft[band][channel] = []

        for beat in beats:
            eeg_band_fft[beat] = dict()
            for channel in channels:
                eeg_band_fft[beat][channel] = []

        for channel in channels:
            avgs[channel] = []

        for i in range(int((len(data)-2)/3), len(data), int((len(data)-2)/3)):
            for channel in channels:
                locdata = raw_data.loc[i - int((len(data)-2)/3):i, channel]
                for band in eeg_bands:
                    eeg_band_fft[band][channel].append(bandpower(locdata, fs, eeg_bands[band]))
                for beat in beats:
                    eeg_band_fft[beat][channel].append(bandpower_beat(locdata, fs, beats[beat]))

        for channel in channels:
            avgs[channel].append(np.average(
                np.array(eeg_band_fft['Alpha'][channel]))/np.average(
                    np.array(eeg_band_fft['Delta'][channel])))
            avgs[channel].append(np.average(
                np.array(eeg_band_fft['Beta'][channel]))/np.average(
                    np.array(eeg_band_fft['Theta'][channel])))
            avgs[channel].append(np.average(
                np.array(eeg_band_fft['Theta'][channel]))/np.average(
                    np.array(eeg_band_fft['Alpha'][channel])))

            for band in eeg_bands:
                avgs[channel].append(np.average(eeg_band_fft[band][channel]))

            for beat in beats:
                avgs[channel].append(np.average(
                    np.array(eeg_band_fft[beat][channel])))
        l = 0
        for channel in channels:
            avg = np.round(np.array(avgs[channel]).reshape(-1, 1), decimals=8)
            channelavgs.append(avg)

            for average in avg:
                values[str(l)] = average
                l += 1
        if not np.isnan(np.sum(np.array(channelavgs))):
            k += 1
            samples.append(pd.DataFrame(values, index=[0]))
    df = pd.concat(samples)
    return df

import numpy as np
import scipy
import librosa
import os



def HPSS_Fitzgerald(x, fs=44100):
	
	if fs != 44100:
		x = librosa.resample(x,fs,44100)
		fs = 44100

	# Parameter settings follow Fitzgerald's paper
	X = librosa.core.stft(x,n_fft=4096,hop_length=1024,window='hamming')
	X_abs = np.abs(X)+0.0000000001
	phase = X/X_abs

	M_harm = np.zeros((X_abs.shape))
	M_perc = np.zeros((X_abs.shape))

	for k in range(X_abs.shape[0]):
		M_harm[k,:] = scipy.signal.medfilt(X_abs[k,:],kernel_size=17)

	for l in range(X_abs.shape[1]):
		M_perc[:,l] = scipy.signal.medfilt(X_abs[:,l],kernel_size=17)

	X_harmonic = (np.square(M_harm)/( np.square(M_harm)+np.square(M_perc)+0.00000000001 ))*X_abs
	X_percussive = (np.square(M_perc)/( np.square(M_harm)+np.square(M_perc)+0.00000000001 ))*X_abs

	harmonic = librosa.core.istft(X_harmonic*phase,hop_length=1024,window='hamming')
	percussive = librosa.core.istft(X_percussive*phase,hop_length=1024,window='hamming')
	harmonic = harmonic/np.max(harmonic)
	percussive = percussive/np.max(percussive)

	return (harmonic,percussive)



def main():
	### Example
	filename = './example.wav'
	x, fs = librosa.core.load(filename,sr=44100,mono=True)

	harmonic,percussive = HPSS_Fitzgerald(x,fs=44100)

	# Save
	librosa.output.write_wav(os.path.splitext(filename)[0]+'_H.wav',harmonic,fs)
	librosa.output.write_wav(os.path.splitext(filename)[0]+'_P.wav',percussive,fs)


if __name__ == "__main__":
	main()

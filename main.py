import streamlit as st

# Session-states for streamlit that are shared between sessions.
# Touch this if you want to break something fast.
try:
    # check if the key exists in session state
    _=st.session_state.record_flag
    _=st.session_state.upload_flag
    _=st.session_state.got_audio
    _=st.session_state.get_mfcc
    _=st.session_state.audio_path
    _=st.session_state.mfcc_path
    _=st.session_state.import_flag
except AttributeError:
    # otherwise set it to false
    st.session_state.record_flag=False
    st.session_state.upload_flag=False
    st.session_state.got_audio=False
    st.session_state.get_mfcc=False
    st.session_state.audio_path=""
    st.session_state.mfcc_path=""
    st.session_state.import_flag=False


st.title("STT-Engine")
st.session_state.import_flag=False
if not st.session_state.import_flag:
    bar = st.progress(0)
    with st.spinner(text="Loading Resources"):
        from librosa.display import specshow
        bar.progress(9.09/100)
        import matplotlib.pyplot as plt
        bar.progress(2*9.09/100)
        import scipy.io.wavfile
        bar.progress(3*9.09/100)
        import numpy as np
        bar.progress(4*9.09/100)
        from math import pi,cos,log,exp,floor
        bar.progress(5*9.09/100)
        from scipy.fftpack import dct
        bar.progress(6*9.09/100)
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
        bar.progress(7*9.09/100)
        import os
        bar.progress(8*9.09/100)
        from tensorflow.keras.models import load_model
        bar.progress(9*9.09/100)
        from tensorflow.keras.utils import image_dataset_from_directory
        bar.progress(10*9.09/100)
        from torchaudio import load as torchload
        bar.progress(11*9.09/100)
        st.session_state.import_flag=True
        st.success("Imported Resources")

else:
    bar.progress(100/100)
def remake_audio(path):
    audio,sampleRate = torchload(path)
    audio=audio.numpy()
    audio=np.mean(audio,axis=0)
    audioSize=audio.size
    rem=audioSize%16000
    audio=np.append(audio,[np.int16(0)]*(16000-rem))
    path+="_temp.wav"
    scipy.io.wavfile.write(path,sampleRate,audio)
    return path


# All the audio samples are single-channel(mono)
# Loads audio file from path and returns tuple of (sample rate, audio array)
def load_audio(path):
    path=remake_audio(path)
    sampleRate, audio = scipy.io.wavfile.read(path)
    return sampleRate,audio,path

# Time vs. Amplitude plot o original audio, loaded initially.
def plot_audio_init(audio):
    fig=plt.figure(figsize=(12,5))
    plt.plot(audio)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal")
    st.pyplot(fig)
    plt.close()



# Voice-Activity Detection using Silero-VAD.
# Detects first instance of speech in an audio.
# If an error occurs or speech is not detected, function returns tuple of (-1,audio). 
# A check on returned sampleRate can be applied after function implementation to check if speech is detected without error, or not.

def vad(sampleRate,audio,path):
    if sampleRate%16000!=0:
        sampleRate=16000
    wav = read_audio(path,sampling_rate=sampleRate)
    os.remove(path)
    model = load_silero_vad()
    try:
        speech_timestamps = get_speech_timestamps(wav,model,sampling_rate=sampleRate)
    except:
        print(path,sampleRate)
    try:
        x=speech_timestamps[0]['start']
        y=speech_timestamps[0]['end']
        wav=wav[x:y]
        audio = wav.detach().cpu().numpy()
        sampleRate=len(audio)
        return sampleRate,audio
    except:
        print(path)
        return -1,audio

#Time vs Amplitude plot of audio after voice-activity isolation
def plot_audio_vad(audio):
    fig=plt.figure(figsize=(12,5))
    plt.plot(audio)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Audio Signal after VAD")
    st.pyplot(fig)
    plt.close()



# To generate MFCC, we follow the following steps:
# audioInput -> pre-emphasis -> framing -> windowing -> fourier transform -> Inverse Mel Scale Filter Bank -> Log() -> DCT
# Pre-emphasis layer
# Amplifies higher frequencies in order to balance the spectrum (higher frequencies have lower energies)
def pre_emphasize(sampleRate,audio):
    pre_emphasis = 0.97
    audio_preemphasized=[]
    for i in range(1,sampleRate):
        audio_preemphasized=np.append(audio_preemphasized,audio[i]-(audio[i-1]*pre_emphasis))

    return audio_preemphasized

# Time vs. Amplitude plot of audio after pre-emphasis
def plot_audio_pre_emphasis(audio_preemphasized):
    # Plot the pre-emphasized signal
    plt.figure(figsize=(14, 5))
    plt.plot(audio_preemphasized)
    plt.title('Pre-emphasized Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    plt.close()



# Framing Layer
# Since the audio wave is more than a second, windowing is necesarry in order to fully capture the features and allow for correct
# calculations to be performed. Thus, for ease of calculations, we slice the wave.
# The signal/wave is separated into sections or frames of 25-30 milliseconds.
# Since some parts of the signal are always at the ends of the frames, and we have to perform hamming window, this may result in data loss.
# To tackle this, we frame-shift with a stride of 15ms. This ensures that parts of signals get to be in the center of the signal.

def frame_audio(sampleRate,audio_preemphasized):
    shift_stride=220  # ~10 millisecond of stride
    frame_size=650 # ~30 millisecond frame
    audio_frames=[]

    # Produces 65 audio frames
    for i in range(0,sampleRate-frame_size,shift_stride):
        audio_frames.append(audio_preemphasized[i:i+frame_size])

    return frame_size,audio_frames

# Time vs. Amplitude plot of audio after framing
def plot_audio_frame(audio_frames):
    plt.figure(figsize=(12,4))
    plt.plot(audio_frames[2])
    plt.title('Framed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    plt.close()



# Windowing Layer
# Since sudden increase/decrease of amplitude at the edges of the frames create noisy outcomes, we have to smoothen it.
# Thus, we apply hamming window

def hamm_audio(sampleRate,frame_size,audio_frames):
    hammed_audio=[]
    for frame in audio_frames:
        temp_hammed_audio=[]
        for i in range(0,frame_size):
            temp_hammed_audio.append(frame[i]*(0.54-0.46*cos(2*pi*i/(frame_size-1))))
        
        hammed_audio.append(temp_hammed_audio)

    return hammed_audio

# Time vs. Amplitude plot of audio after application of hamming window
def plot_audio_hammed(hammed_audio):
    plt.figure(figsize=(12,4))
    plt.plot(hammed_audio[2])
    plt.title('Windowed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    plt.close()



# FFT(Fast Fourier Transform) Layer
# Used to convert time-domain signal to frequency-domain to analyze frequency components of speech.
# Output of FFT gives complex frequency spectrum (both magnitude and phase)
# Since we only need magnitude, we evaluate the power spectrum from the output of FFT
# NFFT specifies number of points for the FFT. The output is NFFT/2 points

def pow_spec(hammed_audio):
    NFFT=2048
    complex_power_spectrums=np.fft.rfft(hammed_audio,NFFT)
    power_spectrum=(1/NFFT)*pow(np.abs(complex_power_spectrums),2)
    return NFFT,power_spectrum

# Frequency vs. Power/Frequency plot of audio after FFT
def plot_power_spectrum(power_spectrum):
    plt.figure(figsize=(12,4))
    plt.plot(power_spectrum[2])
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.show()
    plt.close()



# Mel-filter banks
# Mel-scale related to human-percieved frequency to its actual frequency. Since humans do not hear sound linearly,
# i.e, linear gaps in frequency does not amount to linear change in pitch, we use mel-scale.
# Mel-scale is a logarithm scale, which imitates hearing of humans. Thus, it enables us to capture features as if heard by human.

# Computing the Mel-Filter bank
# 1. Decide upper and lower frequencies in Hertz(SampleRate/2 and 300Hz repectively) 
# 2. Convert them to mels.
# 3. Compute 12 linearly-spaced frequencies inclusive of lower and upper mels.
# 4. Convert these points back to Hertz.
# 5. Round the frequencies to their nearest FFT Bins.
# 6. Create Filterbanks


# Creates frequency bins and returns tuple of (number of filters, frequency bins)
def mels(sampleRate,NFFT):
    freq_to_mel=lambda freq:1125*log(1+freq/700)
    lower_hz=300
    upper_hz=sampleRate/2

    lower_mel=freq_to_mel(lower_hz)
    upper_mel=freq_to_mel(upper_hz)

    n_filters=40
    mel_arr=np.linspace(lower_mel,upper_mel,n_filters+2)
    hz_arr=[700*(exp((i/1125))-1) for i in mel_arr]

    freq_bin=[floor((NFFT+1)*hz_arr_i/sampleRate) for hz_arr_i in hz_arr]
    return n_filters,freq_bin

# Computing the filterbanks.
# Returns filter_banks
def mel_filterbanks(NFFT,n_filters,freq_bin,power_spectrum):
    temp_filter_bank=np.zeros((n_filters,int((NFFT/2))+1))
    for i in range(1,n_filters+1):
        for k in range(0,int((NFFT/2))):  #frame length
            if k<freq_bin[i]:
                temp_filter_bank[i-1][k]=0
            elif freq_bin[i-1]<=k and k<=freq_bin[i]:
                temp_filter_bank[i-1][k]=(k-freq_bin[i-1])/(freq_bin[i]-freq_bin[i-1])
            elif freq_bin[i]<=k and k<=freq_bin[i+1]:
                temp_filter_bank[i-1][k]=(freq_bin[i+1]-k)/(freq_bin[i+1]-freq_bin[i])
            else:
                temp_filter_bank[i-1][k]=0


    filter_banks=np.dot(power_spectrum, temp_filter_bank.T)
    filter_banks=np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks=np.log(filter_banks+1e-8)

    return filter_banks

# Mel-Spectogram plot of filter banks
def plot_mel_spectogram(sampleRate,filter_banks):
    plt.figure(figsize=(12, 4))
    specshow(filter_banks.T, sr=sampleRate, x_axis='time', y_axis='mel',cmap='turbo')
    plt.colorbar()
    plt.title("Mel-Spectogram")
    plt.tight_layout()
    plt.show()
    plt.close()



# Generate MFCCs
# We apply DCT on the filterbanks to obtain a set of 26 Mel-Frequency Cepstral Coefficients.
# We only require first 13 coefficients for ASR purposes. Rest are to be discarded.
# Returns mfcc
def gen_mfcc(filter_banks):
    mfcc = dct(filter_banks, type=2, axis=1)[:, 1:13] # Keep 2-13
    return mfcc

# Plot of first 13 Mel-Frequency Cepstral Coefficients
def plot_mfcc(sampleRate,mfcc):
    plt.figure(figsize=(12, 4))
    specshow(mfcc.T, sr=sampleRate, x_axis='time', y_axis='mel',cmap='turbo',vmin=-100,vmax=100)
    plt.colorbar()
    plt.title("Mel-Cepstral Frequency Coefficients")
    plt.tight_layout()
    plt.show()
    plt.close()



# To simplify the process of getting MFCC:
#   get_mfcc(path), returns tuple (flag,mfcc).
#       input(String: path)
#       gets audio file from the path and applies necessary functions to obtain Mel-Frequency Cepstral Coefficients.
#       If, during loading of audio or application of VAD, some error occurs, sampleRate

def get_mfcc(path):
    flag=0
    sampleRate,audio,path=load_audio(path)
    sampleRate,audio=vad(sampleRate,audio,path)
    if sampleRate==-1:
        flag=1
        return flag,flag
    audio_preemphasized=pre_emphasize(sampleRate,audio)
    frame_size,audio_frames=frame_audio(sampleRate,audio_preemphasized)
    hammed_audio=hamm_audio(sampleRate,frame_size,audio_frames)
    NFFT,power_spectrum=pow_spec(hammed_audio)
    n_filters,freq_bin=mels(sampleRate,NFFT)
    filter_banks=mel_filterbanks(NFFT,n_filters,freq_bin,power_spectrum)
    mfcc=gen_mfcc(filter_banks)
    return flag,mfcc

def get_mel_filterbanks(path):
    sampleRate,audio,path=load_audio(path)
    sampleRate,audio=vad(sampleRate,audio,path)
    audio_preemphasized=pre_emphasize(sampleRate,audio)
    frame_size,audio_frames=frame_audio(sampleRate,audio_preemphasized)
    hammed_audio=hamm_audio(sampleRate,frame_size,audio_frames)
    NFFT,power_spectrum=pow_spec(hammed_audio)
    n_filters,freq_bin=mels(sampleRate,NFFT)
    filter_banks=mel_filterbanks(NFFT,n_filters,freq_bin,power_spectrum)
    return filter_banks



# Extracts MFCCs from the audio files provided in the input_path and stores the MFCC images at output_path

def prepare_data(input_path,output_path):
    os.makedirs(output_path, exist_ok=True)
    file_names = os.listdir(input_path)
    for file in file_names:
        flag,mfcc=get_mfcc(input_path+file)
        if flag==1:
            continue
        fig=plt.figure(figsize=(1, 1),frameon=False)
        specshow(mfcc.T, x_axis='time', y_axis='mel',cmap='turbo',vmin=-100,vmax=100)   
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path+file+".png")
        plt.close()




# Check the first 3 of data.
# image tensor(tf.tensor) needs to be converted to numpy array and normalized to range 0-1, for appropriate plot.
# label tensor's 0th index contains the label.
# 0 = Bird
# 1 = Cat
# 2 = Dog
# num is used to iterate over num images and labels in dataset
def check_dataset(dataset,num):
    import cv2
    for image_tensor,label_tensor in train_dataset:
        if num<=0:
            break
        image=image_tensor.numpy()
        image=image[0]/255
        label=label_tensor[0].numpy()
        cv2.imshow(f"{label}",image)
        print(label)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        num-=1





# Now, we predict audio that is not present in the training or validation data.
# These audio files were collected from sources on the internet.

# Extracting MFCCs from the audio files.
# prepare_data('test_words/audioData/bird/','test_words/mfccs/bird/')
# prepare_data('test_words/audioData/cat/','test_words/mfccs/cat/')
# prepare_data('test_words/audioData/dog/','test_words/mfccs/dog/')



# # Make labeled dataset from the MFCCs obtained
# test_data=image_dataset_from_directory('test_words/mfccs/',
#   seed=123,labels='inferred',image_size=(100,100),batch_size=1)



# # Evaluation
# loss,accuracy=model.evaluate(test_data)



# # Generate Text from prediction
# words={
#     0:"Bird",
#     1:"Cat",
#     2:"Dog"
# }
# for im,lb in test_data:
#     pred=model.predict(im,verbose=3)
#     print(f"Prediction: {words[np.argmax(pred)]} \t True Value: {words[lb[0].numpy()]}")

        
# If record button hasn't been pressed yet, display button and mark as pressed if so. 
if not st.session_state.record_flag and st.button("Record Audio"):
    if st.session_state.got_audio==True:
        os.remove(st.session_state.audio_path+"rec_audio.wav")
        st.session_state.got_audio=False
    st.session_state.record_flag=True
    st.session_state.upload_flag=False
    st.rerun()
        
# If upload button hasn't been pressed yet, display button and mark as pressed if so.
if not st.session_state.upload_flag and st.button("Upload Audio"):
    if st.session_state.got_audio==True:
        os.remove(st.session_state.audio_path+"rec_audio.wav")
        st.session_state.got_audio=False
    st.session_state.record_flag=False
    st.session_state.upload_flag=True
    st.rerun()


# Record button is pressed
if st.session_state.record_flag==True and st.session_state.upload_flag==False:
    audio=st.audio_input("Record a voice message")
    if audio:
        os.makedirs("test_words/rt_test/audio",exist_ok=True)
        with open("test_words/rt_test/audio/rec_audio.wav", "wb") as f:
            f.write(audio.getbuffer())
            st.write("Audio recorded and saved successfully!")

        del audio
        st.session_state.audio_path="test_words/rt_test/audio/"
        st.session_state.mfcc_path="test_words/rt_test/mfcc/"
        st.session_state.record_flag=False
        st.session_state.got_audio=True


# Upload button is pressed
if st.session_state.record_flag==False and st.session_state.upload_flag==True:
    audio=st.file_uploader("Upload Audio")
    st.audio(audio,format="audio/wav")
    if audio:
        os.makedirs("test_words/rt_test/audio",exist_ok=True)
        with open("test_words/rt_test/audio/rec_audio.wav", "wb") as f:
            f.write(audio.getbuffer())
            st.write("Audio uploaded and saved successfully!")

        del audio
        st.session_state.audio_path="test_words/rt_test/audio/"
        st.session_state.mfcc_path="test_words/rt_test/mfcc/"
        st.session_state.upload_flag=False
        st.session_state.got_audio=True


# If audio is stored, display get mfcc button
# If clicked, run get_mfcc()
if st.session_state.got_audio==True and st.button("Generate Text"):
    st.session_state.got_audio=False
    st.session_state.get_mfcc=True
    st.rerun()


# If GET MFCC button is clicked, extract MFCC and predict word
# remove the files stored and reset flags to initial state.
if st.session_state.get_mfcc==True:
    st.session_state.get_mfcc=False
    prepare_data(st.session_state.audio_path,st.session_state.mfcc_path+"img/")
    model=load_model('models/MNIST/model3.keras')
    test_data=image_dataset_from_directory(st.session_state.mfcc_path,image_size=(100,100),batch_size=1)
    pred=model.predict(test_data)
    st.session_state.record_flag=False
    st.session_state.upload_flag=False
    # Generate Text from prediction
    words={
        0:"0",
        1:"1",
        2:"2",
        3:"3",
        4:"4",
        5:"5",
        6:"6",
        7:"7",
        8:"8",
        9:"9"
    }
    st.write(f"Prediction: {words[np.argmax(pred)]}")



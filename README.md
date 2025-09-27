# Speech-To-Text Engine
Experimental STT-Engine based on Mel-Cepstral Frequency Coefficients and Convolutional Neural Network for text generation from speech.

# User guide
This notebook serves as an experimental speech-to-text engine that is based on Mel-Frequency Cepstral Coefficients of the given audio for speech recognition and its respective spell generation.
For the time being, we evaluate speech recognition on three words: 'Bird', 'Cat', 'Dog'.

610 audio files of words are used to train the model:
193 for birds,
207 for cats,
210 for dogs.

This is a small part of a public dataset for single-word speech recognition, 2017 by author Warden,Pete.

# Program Workflow
1. Load Audio  
   Audio numpy array and sample rate are extracted from the audio files provided in the path.
   Since, some audio files can have differing samples and channels, we turn the number of samples to multiples of 16000 for better voice detection.

3. Voice Activity Detection    
   Audio samples can have differing lengths and speech can vary inside the audio file. There may be some portions where speech is empty. Thus, we need to extract the point where the speech starts and where the speech ends. This is done with the help of silero-vad library, which is a time-based model trained on ____. It supports audio with sample rate of 8000Hz or 16000Hz (or its multiples). We isolate the speech and return its start point and end point, for better classification.

4. MFCC Generation  
  i. Pre-emphasis layer  
  ii. Framing Layer  
  iii. Windowing Layer  
  iv. Fast Fourier Transform Layer  
  v. Mel-Filterbanks Layer  
  vi. MFCC-Generation Layer  

The MFCCs are stored as RGB (100x100) images.

5. Process all the audio files using the above flow and make a dataset.
   The training dataset consists of 80% of the images and validation dataset consists of 20% of the images.

6. Convolutional Neural Network to process the images.
7. Train the model
8. Validate the model (~97% accuracy).
9. Input test audio.
10. Make prediction
11. Generate text

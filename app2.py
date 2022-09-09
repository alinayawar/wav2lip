
# import librosa.display
# import IPython
# from IPython.display import Audio
import numpy as np
import torch
from re import A
from os import listdir, path
# import numpy as np
import scipy, cv2, os, sys, argparse, audio
# import audio
import json, subprocess, random, string
# from models import Wav2Lip
import platform

import warnings
warnings.filterwarnings("ignore")

from inference import face_detect,load_model,main
# from skimage import img_as_ubyte
from flask import Flask, request,jsonify,send_file,render_template
from flask_cors import CORS, cross_origin

from PIL import Image
import io
import requests
import urllib.request


app = Flask(__name__)
CORS(app, resources = {
        r"/*":{
            "origins": "*"
        }
    }, headers='Content-Type')  ##added for origin access
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def homepage():
    return render_template("index.html", title="JUST WORK")


@app.route('/post', methods=['GET', 'POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def post():

    if request.method == 'POST':

         if not os.path.isfile(face):
		      raise ValueError('--face argument must be a valid path to video/image file')

	     elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		     full_frames = [cv2.imread(face)]
		     fps = fps

	     else:
		     video_stream = cv2.VideoCapture(face)
		     fps = video_stream.get(cv2.CAP_PROP_FPS)

		     print('Reading video frames...')

		     full_frames = []
		     while 1:
			      still_reading, frame = video_stream.read()
			      if not still_reading:
				    video_stream.release()
				    break
			      if resize_factor > 1:
				  frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

			      if rotate:
				  frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			      y1, y2, x1, x2 = crop
			      if x2 == -1: x2 = frame.shape[1]
			      if y2 == -1: y2 = frame.shape[0]

			      frame = frame[y1:y2, x1:x2]

			      full_frames.append(frame)

	     print ("Number of frames available for inference: "+str(len(full_frames)))

         if not audio.endswith('.wav'):
		     print('Extracting raw audio...')
	     	 command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio, 'temp/temp.wav')

	         subprocess.call(command, shell=True)
		     audio = 'temp/temp.wav'

	     wav = audio.load_wav(audio, 16000)
         mel = audio.melspectrogram(wav)
         print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
		    raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

		mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
	    while 1:
		    start_idx = int(i * mel_idx_multiplier)
		    if start_idx + mel_step_size > len(mel[0]):
			     mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			     break
	     	mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
	    	i += 1

	    print("Length of mel chunks: {}".format(len(mel_chunks)))

    	full_frames = full_frames[:len(mel_chunks)]

    	batch_size = wav2lip_batch_size
    	gen = datagen(full_frames.copy(), mel_chunks)

	    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		   if i == 0:
		     	model = load_model(checkpoint_path)
		    	print ("Model loaded")

			    frame_h, frame_w = full_frames[0].shape[:-1]
			    out = cv2.VideoWriter('temp/result.avi',
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()



if __name__ == '__main__':
        # app.run(host='0.0.0.0', port=5000, debug=True)
        app.run(debug=True, use_reloader=False,host='0.0.0.0', port=5000)





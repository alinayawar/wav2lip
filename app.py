
# import librosa.display
# import IPython
# from IPython.display import Audio
import warnings

import cv2
import numpy as np
import torch

# import audio
# from models import Wav2Lip
warnings.filterwarnings("ignore")

from inference import face_detect,load_model
# from skimage import img_as_ubyte
from flask import Flask, request, send_file,render_template
from flask_cors import CORS, cross_origin

from PIL import Image
import io

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

        image = request.files['image']
        print('image recieved')
        print(" ")

        print(type(image))
        print(" ")

        image = image.read()
        print('after image read')
        print(" ")

        print(type(image))
        print(" ")

        # image = Image.open(requests.get(image, stream=True).raw)
        image = Image.open(io.BytesIO(image))
        print('after image open')
        print(" ")

        print(type(image))
        print(" ")

        image =np.array(image)
        # image = resize(image, (256, 256))[..., :3]
        img_size = 96
        image = cv2.resize(image, (img_size,img_size))

        print("image resized")
        print(" ")

        audio = request.files['audio']
        print('audio recieved')
        print(" ")

        print(type(audio))
        print(" ")

        audio = audio.read()
        print('audio read')
        print(" ")

        print(type(audio))
        print(" ")

        audio = 'temp/temp.wav'
        audio = audio.load_wav(audio, 16000)
        # audio = imageio.get_reader(audio, '.wav')
        # video = imageio.get_reader(requests.get(audio, allow_redirects=True, stream=True).raw, 'mp4')
        # print('video url open')
        # print(" ")

        print(type(audio))
        print(" ")
        mel = audio.melspectrogram(audio)
        print(mel.shape)
        detector,batch_size = face_detect(images=image
                                    )
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

        full_frames = full_frames[:len(mel_chunks)]
        for i, (img_batch, mel_batch, image, coords) in enumerate(tqdm(gen,
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                model = load_model(path='wav2lip_gen.path')
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

        # out.release()
        # fps = audio.get_meta_data()['fps']
        # print(fps)
        # driving_audio = []
        # try:
        #     for im in audio:
        #         driving_audio.append(im)
        # except RuntimeError:
        #     pass
        # audio.close()

        # driving_audio = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        # print('video resized')

        # source_image, driving_video,fps = control(image, video)

        # results, kp_detector = _load(checkpoint_path='wav2lip_gen.path',
        #                                         cpu=True
        #                                         )
        # print("generator done")

        # detector,	batch_size = face_detect(images=image
        #                             ) #cpu

        model = load_model(path='wav2lip_gen.path')

        cv2.VideoWriter('result.avi',
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        return send_file('generatedVideo.mp4',
                        as_attachment=True)



if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
        # app.run(debug=True, use_reloader=False,host='0.0.0.0', port=5000)





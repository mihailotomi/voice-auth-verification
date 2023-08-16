from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
from speechbrain.pretrained import SpeakerRecognition
import os


verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

def save_audio_file(audio_file):
    file_path = os.path.join(os.getcwd(), 'audio.wav')
    with open(file_path, 'wb') as f:
        f.write(audio_file)
    return file_path

class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get('Content-Type')
        if content_type.startswith('multipart/form-data'):
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                         'CONTENT_TYPE': content_type,
                         })
            audio_file = form.getfirst('audio')
            if audio_file:
                audio_file_path = save_audio_file(audio_file)

                output = myFunc(audio_file_path)
                if isinstance(output, bool):
                    output_str = str(output).lower()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(output_str.encode('utf-8'))
                else:
                    self.send_error(500, 'Invalid output type')
            else:
                self.send_error(400, 'Missing audio file')
        else:
            self.send_error(415, 'Unsupported content type')

        

def myFunc(audio_file_path):
    score, prediction = verification.verify_files("Recording1.wav", audio_file_path) # Same Speaker
    prediction = prediction.item()
    return prediction


if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), MyHTTPRequestHandler)
    print('Listening on http://localhost:8000')
    server.serve_forever()


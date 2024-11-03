# app.py
from flask import Flask, render_template, request
import os, uuid
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import get_window

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLOT_FOLDER'] = 'static/plots'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)

def save_file(file):
    id = uuid.uuid4().hex
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{id}.wav")
    file.save(path)
    return id, path

def plot_signal(time, signal, plot_path):
    plt.figure(figsize=(8, 3))
    plt.plot(time, signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.ylim(signal.min() * 1.1, signal.max() * 1.1)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def extract_segment(data, fs, start, duration):
    start_idx = int(start * fs)
    end_idx = int((start + duration) * fs)
    return data[start_idx:end_idx]

def compute_spectrum(segment, fs, window_type):
    window = get_window(window_type.lower(), len(segment))
    windowed = segment * window
    N = len(windowed)
    W = fft(windowed)
    mag = np.abs(W)
    if N %2 ==0:
        mag = mag[:N//2+1]
    else:
        mag = mag[:(N+1)//2]
    mag[1:-1] *=2 if N%2==0 else 1
    freq = np.linspace(0, fs/2, len(mag))
    return freq, mag

def plot_spectrum(freq, mag, plot_path):
    plt.figure(figsize=(8, 3))
    plt.plot(freq, mag)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.wav'):
            # Generate a new unique id for the new file upload
            id, path = save_file(file)
            fs, data = wavfile.read(path)
            if data.ndim > 1:
                data = data[:, 0]
            time = np.arange(len(data)) / fs
            plot_path = os.path.join(app.config['PLOT_FOLDER'], f"{id}_time.png")
            plot_signal(time, data, plot_path)
            # Render the template with the new file's id and plot
            return render_template('index.html', id=id, plot=f"plots/{id}_time.png")
    # If no file is uploaded, render the page without an id
    return render_template('index.html', id=None)


@app.route('/analyze', methods=['POST'])
def analyze():
    id = request.form.get('id')
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{id}.wav")
    fs, data = wavfile.read(path)
    if data.ndim >1:
        data = data[:,0]
    start = float(request.form.get('start'))
    duration = float(request.form.get('duration'))
    segment = extract_segment(data, fs, start, duration)
    time = np.arange(len(segment))/fs + start
    seg_plot = os.path.join(app.config['PLOT_FOLDER'], f"{id}_segment.png")
    plot_signal(time, segment, seg_plot)
    window = request.form.get('window', 'hann')
    freq, mag = compute_spectrum(segment, fs, window)
    spec_plot = os.path.join(app.config['PLOT_FOLDER'], f"{id}_spectrum.png")
    plot_spectrum(freq, mag, spec_plot)
    return render_template('index.html', id=id, plot=f"plots/{id}_time.png",
                           segment_plot=f"plots/{id}_segment.png",
                           spectrum_plot=f"plots/{id}_spectrum.png")

if __name__ == '__main__':
    app.run(debug=True)

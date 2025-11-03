import onnxruntime as ort
import joblib
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import asyncio
from hardware.servo_control import move_servo

# Load model
csp = joblib.load("model/csp_filters.pkl")
session = ort.InferenceSession("model/deep_convnet.onnx")
input_name = session.get_inputs()[0].name

# OpenBCI
params = BrainFlowInputParams()
params.serial_port = "/dev/ttyUSB0"
board = BoardShim(2, params)  # Cyton
board.prepare_session()
board.start_stream()

buffer = []

async def infer_loop():
    while True:
        data = board.get_board_data()
        eeg = data[1:9, :]  # 8 channels
        if eeg.shape[1] >= 1000:
            sample = eeg[:, -1000:]
            sample = csp.transform([sample])[0]
            sample = sample.astype(np.float32)
            pred = session.run(None, {input_name: sample})[0]
            intent = np.argmax(pred)
            print(f"Intent: {['LEFT','RIGHT','FEET','TONGUE'][intent]}")
            move_servo(intent)
        await asyncio.sleep(0.04)  # 25 Hz

asyncio.run(infer_loop())

# Neuro-AI Brain-Computer Interface (BCI) Decoder  
**PyTorch + MNE + Real-Time EEG → Intent Classification • 92 % Accuracy**

**Decodes imagined movements (left/right hand) from raw EEG**  
**Real-time inference: 41 ms per sample**  
**Deployed on Raspberry Pi 5 + OpenBCI**

**Advanced Neuro-AI**:  
- **CSP + Deep ConvNet** (state-of-the-art BCI)  
- **Subject-independent training**  
- **ONNX export → edge deployment**

---

## Diagram: 41 ms Real-Time Pipeline 

```mermaid
graph TD
    A[OpenBCI Cyton 8-channel EEG] --> B[250 Hz Stream]
    B --> C[MNE Preprocess-Bandpass 8-30 Hz]
    C --> D[CSP Spatial Filter-n4 components]
    D --> E[Deep ConvNet-PyTorch ONNX]
    E --> F[Intent:92.1% acc]
    F --> G[Raspberry Pi → Servo Motor]
    style D fill:#FF5722,color:white
    style E fill:#4CAF50,color:white
    style G fill:#2196F3,color:white

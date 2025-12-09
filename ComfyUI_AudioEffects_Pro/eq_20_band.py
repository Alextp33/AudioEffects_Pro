import torch
import torchaudio
import numpy as np
from scipy import signal

class AudioEqualizer20Band:
    """
    Equalizzatore parametrico a 20 bande per ComfyUI
    Frequenze standard da 20Hz a 20kHz
    """
    
    def __init__(self):
        # Frequenze centrali delle 20 bande (in Hz)
        self.frequencies = [
            20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
            2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
        ][:20]  # Primi 20
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # 20 bande di equalizzazione (-12dB a +12dB)
                "band_20hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_25hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_31hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_40hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_50hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_63hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_80hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_100hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_125hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_160hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_200hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_250hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_315hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_400hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_500hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_630hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_800hz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_1khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_1_25khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_1_6khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_2khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_2_5khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_3_15khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_4khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_5khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_6_3khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_8khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_10khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_12_5khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_16khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "band_20khz": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                # Parametri aggiuntivi
                "q_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "overall_gain": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_eq"
    CATEGORY = "audio/effects"
    
    def apply_eq(self, audio, q_factor, overall_gain, **band_gains):
        """
        Applica equalizzazione a 20 bande all'audio
        """
        # Estrai i dati audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Converti in numpy se necessario
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().numpy()
        else:
            audio_np = waveform
        
        # Lista dei guadagni per ogni banda
        gains = [
            band_gains.get("band_20hz", 0.0),
            band_gains.get("band_25hz", 0.0),
            band_gains.get("band_31hz", 0.0),
            band_gains.get("band_40hz", 0.0),
            band_gains.get("band_50hz", 0.0),
            band_gains.get("band_63hz", 0.0),
            band_gains.get("band_80hz", 0.0),
            band_gains.get("band_100hz", 0.0),
            band_gains.get("band_125hz", 0.0),
            band_gains.get("band_160hz", 0.0),
            band_gains.get("band_200hz", 0.0),
            band_gains.get("band_250hz", 0.0),
            band_gains.get("band_315hz", 0.0),
            band_gains.get("band_400hz", 0.0),
            band_gains.get("band_500hz", 0.0),
            band_gains.get("band_630hz", 0.0),
            band_gains.get("band_800hz", 0.0),
            band_gains.get("band_1khz", 0.0),
            band_gains.get("band_1_25khz", 0.0),
            band_gains.get("band_1_6khz", 0.0)
        ]
        
        # Applica equalizzazione per ogni banda
        processed_audio = audio_np.copy()
        
        for i, (freq, gain_db) in enumerate(zip(self.frequencies, gains)):
            if abs(gain_db) > 0.01:  # Solo se c'è un guadagno significativo
                # Calcola coefficienti del filtro parametrico
                gain_linear = 10 ** (gain_db / 20)
                w0 = 2 * np.pi * freq / sample_rate
                alpha = np.sin(w0) / (2 * q_factor)
                
                if gain_db > 0:  # Boost
                    A = gain_linear
                    b0 = 1 + alpha * A
                    b1 = -2 * np.cos(w0)
                    b2 = 1 - alpha * A
                    a0 = 1 + alpha / A
                    a1 = -2 * np.cos(w0)
                    a2 = 1 - alpha / A
                else:  # Cut
                    A = gain_linear
                    b0 = 1 + alpha / A
                    b1 = -2 * np.cos(w0)
                    b2 = 1 - alpha / A
                    a0 = 1 + alpha * A
                    a1 = -2 * np.cos(w0)
                    a2 = 1 - alpha * A
                
                # Normalizza coefficienti
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                # Applica filtro
                if len(processed_audio.shape) == 1:  # Mono
                    processed_audio = signal.lfilter(b, a, processed_audio)
                else:  # Stereo
                    for ch in range(processed_audio.shape[0]):
                        processed_audio[ch] = signal.lfilter(b, a, processed_audio[ch])
        
        # Applica guadagno generale
        if abs(overall_gain) > 0.01:
            gain_linear = 10 ** (overall_gain / 20)
            processed_audio *= gain_linear
        
        # Limita per evitare clipping
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        # Converti di nuovo in tensor
        if isinstance(waveform, torch.Tensor):
            processed_tensor = torch.from_numpy(processed_audio).to(waveform.device).to(waveform.dtype)
        else:
            processed_tensor = processed_audio
        
        return ({"waveform": processed_tensor, "sample_rate": sample_rate},)

# Registra il nodo
NODE_CLASS_MAPPINGS = {
    "AudioEqualizer20Band": AudioEqualizer20Band
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioEqualizer20Band": "Audio 20-Band Equalizer"
}
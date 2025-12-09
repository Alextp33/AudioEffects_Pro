import torch
import torchaudio
import numpy as np
from scipy import signal
import json

class AudioEqualizer20BandSlider:
    """
    Equalizzatore con interfaccia slider migliorata
    """
    
    def __init__(self):
        self.frequencies = [
            20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600
        ]
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # Sliders con nomi più chiari e range ottimizzato
                "low_bass_20hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Sub Bass - 20Hz"
                }),
                "low_bass_31hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Sub Bass - 31Hz"
                }),
                "bass_50hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Bass - 50Hz"
                }),
                "bass_80hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Bass - 80Hz"
                }),
                "bass_125hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Bass - 125Hz"
                }),
                "low_mid_200hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Low Mid - 200Hz"
                }),
                "low_mid_315hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Low Mid - 315Hz"
                }),
                "mid_500hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Mid - 500Hz"
                }),
                "mid_800hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Mid - 800Hz"
                }),
                "upper_mid_1250hz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Upper Mid - 1.25kHz"
                }),
                "presence_2khz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Presence - 2kHz (VOCAL CLARITY)"
                }),
                "presence_3_15khz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Presence - 3.15kHz (ANTI-METALLIC)"
                }),
                "high_mid_5khz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "High Mid - 5kHz"
                }),
                "high_8khz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "High - 8kHz"
                }),
                "high_12_5khz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "High - 12.5kHz"
                }),
                "air_16khz": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Air - 16kHz"
                }),
            },
            "optional": {
                # Preset comuni
                "vocal_preset": (["None", "Warm Vocal", "Clear Vocal", "Anti-Metallic", "Radio Voice"], {
                    "default": "None"
                }),
                "music_preset": (["None", "Bass Boost", "Presence", "Sparkle", "Vintage", "Modern"], {
                    "default": "None"
                }),
                # Controlli avanzati
                "q_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "display": "slider", "tooltip": "Bandwidth (lower = wider)"
                }),
                "master_gain": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider", "tooltip": "Overall volume"
                }),
                "dry_wet_mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider", "tooltip": "Dry/Wet balance"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_eq_slider"
    CATEGORY = "audio/effects"
    
    def get_preset_values(self, preset_name):
        """Preset EQ per situazioni comuni"""
        presets = {
            "Warm Vocal": {
                "low_mid_200hz": 2.0,
                "mid_500hz": 1.0,
                "presence_3_15khz": -2.0,
                "high_8khz": 1.0
            },
            "Clear Vocal": {
                "low_mid_315hz": -1.0,
                "mid_800hz": 1.5,
                "upper_mid_1250hz": 2.0,
                "presence_3_15khz": -1.5,
                "high_12_5khz": 2.0
            },
            "Anti-Metallic": {
                "presence_2khz": -3.0,
                "presence_3_15khz": -4.0,
                "high_mid_5khz": -2.0,
                "low_mid_200hz": 2.0,
                "mid_500hz": 1.0
            },
            "Bass Boost": {
                "bass_50hz": 4.0,
                "bass_80hz": 3.0,
                "bass_125hz": 2.0
            },
            "Presence": {
                "upper_mid_1250hz": 2.0,
                "presence_2khz": 3.0,
                "high_8khz": 2.0
            },
            "Sparkle": {
                "high_8khz": 2.0,
                "high_12_5khz": 3.0,
                "air_16khz": 2.0
            }
        }
        return presets.get(preset_name, {})
    
    def apply_eq_slider(self, audio, vocal_preset="None", music_preset="None", 
                       q_factor=1.0, master_gain=0.0, dry_wet_mix=1.0, **slider_values):
        """
        Applica EQ con interfaccia slider migliorata
        """
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Converti in numpy
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().numpy()
            original_device = waveform.device
            original_dtype = waveform.dtype
        else:
            audio_np = waveform
            original_device = None
            original_dtype = None
        
        # Applica preset se selezionato
        eq_values = slider_values.copy()
        if vocal_preset != "None":
            preset_vals = self.get_preset_values(vocal_preset)
            eq_values.update(preset_vals)
        if music_preset != "None":
            preset_vals = self.get_preset_values(music_preset)
            eq_values.update(preset_vals)
        
        # Mappa slider ai valori di frequenza
        freq_mapping = {
            "low_bass_20hz": 20,
            "low_bass_31hz": 31.5,
            "bass_50hz": 50,
            "bass_80hz": 80,
            "bass_125hz": 125,
            "low_mid_200hz": 200,
            "low_mid_315hz": 315,
            "mid_500hz": 500,
            "mid_800hz": 800,
            "upper_mid_1250hz": 1250,
            "presence_2khz": 2000,
            "presence_3_15khz": 3150,
            "high_mid_5khz": 5000,
            "high_8khz": 8000,
            "high_12_5khz": 12500,
            "air_16khz": 16000
        }
        
        # Processa audio
        processed_audio = audio_np.copy()
        
        for slider_name, freq in freq_mapping.items():
            gain_db = eq_values.get(slider_name, 0.0)
            
            if abs(gain_db) > 0.01:
                # Calcola filtro parametrico
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
                
                # Normalizza e applica
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                if len(processed_audio.shape) == 1:
                    processed_audio = signal.lfilter(b, a, processed_audio)
                else:
                    for ch in range(processed_audio.shape[0]):
                        processed_audio[ch] = signal.lfilter(b, a, processed_audio[ch])
        
        # Master gain
        if abs(master_gain) > 0.01:
            gain_linear = 10 ** (master_gain / 20)
            processed_audio *= gain_linear
        
        # Dry/Wet mix
        if dry_wet_mix < 1.0:
            processed_audio = audio_np * (1 - dry_wet_mix) + processed_audio * dry_wet_mix
        
        # Limita
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        
        # Riconverti
        if original_device is not None:
            processed_tensor = torch.from_numpy(processed_audio).to(original_device).to(original_dtype)
        else:
            processed_tensor = processed_audio
        
        return ({"waveform": processed_tensor, "sample_rate": sample_rate},)

# Registrazione nodo
NODE_CLASS_MAPPINGS = {
    "AudioEqualizer20BandSlider": AudioEqualizer20BandSlider
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioEqualizer20BandSlider": "🎛️ Audio EQ 20-Band Slider Pro"
}
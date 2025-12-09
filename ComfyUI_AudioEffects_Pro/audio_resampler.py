import torch
import numpy as np
import random

class AudioResampler:
    """
    Audio Resampler - Genera varianti dello stesso brano
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_audio": ("AUDIO",),
                "resampling_method": (["Noise Injection", "Latent Variation", "Rhythm Variation", "Harmonic Variation"], {
                    "default": "Noise Injection"
                }),
                "variation_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.1, "max": 0.9, "step": 0.1,
                    "tooltip": "How different from original (0.1=subtle, 0.9=very different)"
                }),
                "preserve_structure": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Maintain original song structure"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 999999999,
                    "tooltip": "Random seed for reproducible variations"
                }),
                "tempo_variation": ("FLOAT", {
                    "default": 0.0, "min": -0.2, "max": 0.2, "step": 0.05,
                    "tooltip": "Change tempo (-0.2 = slower, +0.2 = faster)"
                }),
                "pitch_variation": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Change pitch in semitones"
                }),
                "harmonic_shift": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Shift harmonic content"
                }),
                "rhythm_complexity": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.1,
                    "tooltip": "Add/remove rhythmic elements"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "LATENT", "STRING")
    RETURN_NAMES = ("resampled_audio", "variation_latent", "variation_info")
    FUNCTION = "resample_audio"
    CATEGORY = "audio/generation"
    
    def add_controlled_noise(self, audio_np, strength):
        """Aggiungi rumore controllato per variazioni"""
        noise_level = strength * 0.05  # Max 5% di rumore
        noise = np.random.normal(0, noise_level, audio_np.shape)
        
        # Applica rumore solo nelle parti con energia
        energy = np.abs(audio_np)
        noise_mask = energy > np.percentile(energy, 20)  # Solo dove c'è suono
        
        noisy_audio = audio_np.copy()
        noisy_audio[noise_mask] += noise[noise_mask]
        
        return noisy_audio
    
    def time_stretch(self, audio_np, factor):
        """Cambia tempo senza cambiare pitch"""
        try:
            import librosa
            return librosa.effects.time_stretch(audio_np, rate=factor)
        except:
            # Fallback semplice
            if factor > 1:
                # Faster - skip samples
                indices = np.arange(0, len(audio_np), factor).astype(int)
                return audio_np[indices]
            else:
                # Slower - interpolate
                from scipy import interpolate
                x_old = np.arange(len(audio_np))
                x_new = np.arange(0, len(audio_np)-1, factor)
                f = interpolate.interp1d(x_old, audio_np, kind='cubic')
                return f(x_new)
    
    def pitch_shift(self, audio_np, sample_rate, semitones):
        """Cambia pitch senza cambiare tempo"""
        try:
            import librosa
            return librosa.effects.pitch_shift(audio_np, sr=sample_rate, n_steps=semitones)
        except:
            # Fallback: time stretch + resample
            factor = 2 ** (semitones / 12)
            stretched = self.time_stretch(audio_np, factor)
            # Resample back to original length
            from scipy import signal
            return signal.resample(stretched, len(audio_np))
    
    def harmonic_variation(self, audio_np, sample_rate, shift_amount):
        """Varia il contenuto armonico"""
        # FFT per analisi
        fft = np.fft.rfft(audio_np, axis=-1)
        freqs = np.fft.rfftfreq(audio_np.shape[-1], 1/sample_rate)
        
        # Shift armonico
        shift_factor = 1 + shift_amount * 0.1  # Max 10% shift
        
        # Applica shift alle frequenze medie (non toccare troppo i bassi)
        mid_freq_mask = (freqs > 200) & (freqs < 4000)
        fft[..., mid_freq_mask] *= shift_factor
        
        # Back to time domain
        varied_audio = np.fft.irfft(fft, axis=-1)
        return varied_audio
    
    def rhythm_variation(self, audio_np, sample_rate, complexity_change):
        """Varia elementi ritmici"""
        # Analizza transients/beats
        energy = np.abs(audio_np)
        
        if complexity_change > 0:
            # Aggiungi micro-variazioni ritmiche
            variation_freq = 16  # Variazioni ogni 16 samples
            variation = np.sin(np.arange(len(audio_np)) * 2 * np.pi / variation_freq)
            variation *= complexity_change * 0.05
            varied_audio = audio_np * (1 + variation)
        else:
            # Semplifica (smooth out)
            from scipy import ndimage
            smooth_factor = int(abs(complexity_change) * 10)
            varied_audio = ndimage.uniform_filter1d(audio_np, smooth_factor)
        
        return varied_audio
    
    def generate_variation_latent(self, audio_shape, variation_strength, seed):
        """Genera un latent per variation injection"""
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Simula un latent space variation
        latent_size = (1, 4, audio_shape[-1] // 8)  # Approximation
        variation_latent = torch.randn(latent_size) * variation_strength
        
        return variation_latent
    
    def resample_audio(self, original_audio, resampling_method="Noise Injection", 
                      variation_strength=0.3, preserve_structure=0.8, seed=-1,
                      tempo_variation=0.0, pitch_variation=0.0, harmonic_shift=0.0, 
                      rhythm_complexity=0.0):
        
        waveform = original_audio["waveform"]
        sample_rate = original_audio["sample_rate"]
        
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().numpy()
            original_device = waveform.device
            original_dtype = waveform.dtype
        else:
            audio_np = waveform
            original_device = None
            original_dtype = None
        
        # Set seed for reproducibility
        if seed != -1:
            np.random.seed(seed)
            random.seed(seed)
        
        print(f"🔄 Resampling audio with method: {resampling_method}")
        print(f"   Variation strength: {variation_strength}")
        print(f"   Preserve structure: {preserve_structure}")
        
        # Apply variations based on method
        varied_audio = audio_np.copy()
        variation_info = f"Method: {resampling_method}\n"
        
        if resampling_method == "Noise Injection":
            # Add controlled noise for variation
            noise_strength = variation_strength * (1 - preserve_structure)
            varied_audio = self.add_controlled_noise(varied_audio, noise_strength)
            variation_info += f"Noise level: {noise_strength:.2f}\n"
            
        elif resampling_method == "Latent Variation":
            # Simulate latent space variation (simplified)
            variation_noise = np.random.normal(0, variation_strength * 0.02, varied_audio.shape)
            energy_mask = np.abs(varied_audio) > np.percentile(np.abs(varied_audio), 30)
            varied_audio[energy_mask] += variation_noise[energy_mask]
            variation_info += f"Latent noise applied\n"
            
        elif resampling_method == "Rhythm Variation":
            varied_audio = self.rhythm_variation(varied_audio, sample_rate, rhythm_complexity)
            variation_info += f"Rhythm complexity: {rhythm_complexity:+.2f}\n"
            
        elif resampling_method == "Harmonic Variation":
            varied_audio = self.harmonic_variation(varied_audio, sample_rate, harmonic_shift)
            variation_info += f"Harmonic shift: {harmonic_shift:+.2f}\n"
        
        # Apply additional variations
        if abs(tempo_variation) > 0.01:
            tempo_factor = 1 + tempo_variation
            varied_audio = self.time_stretch(varied_audio, tempo_factor)
            variation_info += f"Tempo: {tempo_variation:+.1%}\n"
        
        if abs(pitch_variation) > 0.01:
            varied_audio = self.pitch_shift(varied_audio, sample_rate, pitch_variation)
            variation_info += f"Pitch: {pitch_variation:+.1f} semitones\n"
        
        # Preserve original structure if requested
        if preserve_structure > 0.5:
            # Blend with original to maintain structure
            blend_factor = preserve_structure
            varied_audio = varied_audio * (1 - blend_factor) + audio_np * blend_factor
            variation_info += f"Structure preserved: {preserve_structure:.1%}\n"
        
        # Normalize and limit
        varied_audio = np.clip(varied_audio, -1.0, 1.0)
        
        # Generate variation latent for potential use in other nodes
        variation_latent = self.generate_variation_latent(audio_np.shape, variation_strength, seed)
        
        # Convert back to tensor
        if original_device is not None:
            resampled_tensor = torch.from_numpy(varied_audio).to(original_device).to(original_dtype)
        else:
            resampled_tensor = varied_audio
        
        variation_info += f"Seed used: {seed if seed != -1 else 'random'}"
        
        print(f"✅ Resampling complete!")
        
        return (
            {"waveform": resampled_tensor, "sample_rate": sample_rate},
            variation_latent,
            variation_info
        )

# Registrazione nodo
NODE_CLASS_MAPPINGS = {
    "AudioResampler": AudioResampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioResampler": "🔄 Audio Resampler Pro"
}
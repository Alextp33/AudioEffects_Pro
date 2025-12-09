import torch
import torchaudio
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import librosa

class AudioUpscaler:
    """
    Audio Upscaler - Aumenta qualità, sample rate e bit depth
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "upscale_method": (["Linear", "Cubic", "Lanczos", "AI_Enhanced"], {
                    "default": "AI_Enhanced"
                }),
                "target_sample_rate": ([22050, 44100, 48000, 96000, 192000], {
                    "default": 48000
                }),
                "target_bit_depth": ([16, 24, 32], {
                    "default": 24
                }),
                "enhancement_level": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Level of AI enhancement"
                }),
            },
            "optional": {
                "noise_reduction": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Reduce background noise"
                }),
                "high_freq_restore": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Restore high frequencies"
                }),
                "dynamics_enhance": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Enhance dynamic range"
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Stereo field width"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "upscale_audio"
    CATEGORY = "audio/enhancement"
    
    def spectral_enhance(self, audio_np, sample_rate, enhancement_level):
        """
        AI-like spectral enhancement
        """
        # FFT per analisi frequenze
        fft = np.fft.rfft(audio_np, axis=-1)
        freqs = np.fft.rfftfreq(audio_np.shape[-1], 1/sample_rate)
        
        # Enhancement curve (simula AI)
        enhancement_curve = np.ones_like(freqs)
        
        # Boost high frequencies (simulate detail recovery)
        high_freq_mask = freqs > 8000
        enhancement_curve[high_freq_mask] *= (1 + enhancement_level * 0.5)
        
        # Subtle mid-range enhancement
        mid_freq_mask = (freqs > 1000) & (freqs < 8000)
        enhancement_curve[mid_freq_mask] *= (1 + enhancement_level * 0.2)
        
        # Apply enhancement
        fft_enhanced = fft * enhancement_curve
        
        # Back to time domain
        enhanced_audio = np.fft.irfft(fft_enhanced, axis=-1)
        
        return enhanced_audio
    
    def high_freq_restoration(self, audio_np, sample_rate, restore_level):
        """
        Restore high frequencies using harmonic prediction
        """
        if restore_level <= 0:
            return audio_np
            
        # Analyze lower frequencies to predict higher ones
        fft = np.fft.rfft(audio_np, axis=-1)
        freqs = np.fft.rfftfreq(audio_np.shape[-1], 1/sample_rate)
        
        # Find fundamental frequencies
        low_freq_mask = freqs < 4000
        high_freq_mask = freqs >= 8000
        
        # Generate harmonics
        low_freq_content = fft[..., low_freq_mask]
        
        # Simple harmonic restoration (basic AI simulation)
        restoration_factor = restore_level * 0.3
        fft[..., high_freq_mask] += low_freq_content[..., :len(fft[..., high_freq_mask])] * restoration_factor
        
        restored_audio = np.fft.irfft(fft, axis=-1)
        return restored_audio
    
    def noise_reduction_spectral(self, audio_np, sample_rate, reduction_level):
        """
        Spectral noise reduction
        """
        if reduction_level <= 0:
            return audio_np
            
        # Spectral subtraction method
        fft = np.fft.rfft(audio_np, axis=-1)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Estimate noise floor (assume first 0.1 seconds is noise)
        noise_samples = int(0.1 * sample_rate)
        if audio_np.shape[-1] > noise_samples:
            noise_fft = np.fft.rfft(audio_np[..., :noise_samples], axis=-1)
            noise_magnitude = np.mean(np.abs(noise_fft), axis=-1, keepdims=True)
            
            # Spectral subtraction
            clean_magnitude = magnitude - reduction_level * noise_magnitude
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)  # Floor
            
            # Reconstruct
            clean_fft = clean_magnitude * np.exp(1j * phase)
            clean_audio = np.fft.irfft(clean_fft, axis=-1)
            return clean_audio
        
        return audio_np
    
    def enhance_dynamics(self, audio_np, enhance_level):
        """
        Enhance dynamic range
        """
        if enhance_level <= 0:
            return audio_np
            
        # Multi-band compression/expansion
        enhanced = audio_np.copy()
        
        # Subtle expansion for more dynamics
        expansion_ratio = 1 + enhance_level * 0.2
        enhanced = np.sign(enhanced) * np.power(np.abs(enhanced), 1/expansion_ratio)
        
        return enhanced
    
    def stereo_widening(self, audio_np, width):
        """
        Enhance stereo width
        """
        if len(audio_np.shape) < 2 or audio_np.shape[0] < 2 or width == 1.0:
            return audio_np
            
        left = audio_np[0]
        right = audio_np[1]
        
        # Mid/Side processing
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        
        # Widen stereo field
        side_enhanced = side * width
        
        # Back to Left/Right
        left_enhanced = mid + side_enhanced
        right_enhanced = mid - side_enhanced
        
        return np.array([left_enhanced, right_enhanced])
    
    def upscale_audio(self, audio, upscale_method="AI_Enhanced", target_sample_rate=48000, 
                     target_bit_depth=24, enhancement_level=0.5, noise_reduction=0.3,
                     high_freq_restore=0.7, dynamics_enhance=0.4, stereo_width=1.0):
        
        waveform = audio["waveform"]
        current_sample_rate = audio["sample_rate"]
        
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().numpy()
            original_device = waveform.device
            original_dtype = waveform.dtype
        else:
            audio_np = waveform
            original_device = None
            original_dtype = None
        
        print(f"🎵 Upscaling audio from {current_sample_rate}Hz to {target_sample_rate}Hz")
        
        # 1. Sample Rate Conversion
        if current_sample_rate != target_sample_rate:
            if upscale_method == "Linear":
                audio_np = librosa.resample(audio_np, orig_sr=current_sample_rate, 
                                          target_sr=target_sample_rate, res_type='linear')
            elif upscale_method == "Cubic":
                audio_np = librosa.resample(audio_np, orig_sr=current_sample_rate, 
                                          target_sr=target_sample_rate, res_type='scipy')
            elif upscale_method == "Lanczos":
                audio_np = librosa.resample(audio_np, orig_sr=current_sample_rate, 
                                          target_sr=target_sample_rate, res_type='sinc_best')
            else:  # AI_Enhanced
                # Use best quality resampling + enhancements
                audio_np = librosa.resample(audio_np, orig_sr=current_sample_rate, 
                                          target_sr=target_sample_rate, res_type='sinc_best')
        
        # 2. AI-like Enhancements (only for AI_Enhanced method)
        if upscale_method == "AI_Enhanced":
            # Spectral enhancement
            audio_np = self.spectral_enhance(audio_np, target_sample_rate, enhancement_level)
            
            # High frequency restoration
            audio_np = self.high_freq_restoration(audio_np, target_sample_rate, high_freq_restore)
            
            # Noise reduction
            audio_np = self.noise_reduction_spectral(audio_np, target_sample_rate, noise_reduction)
            
            # Dynamic range enhancement
            audio_np = self.enhance_dynamics(audio_np, dynamics_enhance)
        
        # 3. Stereo enhancement
        if len(audio_np.shape) > 1:
            audio_np = self.stereo_widening(audio_np, stereo_width)
        
        # 4. Bit depth conversion (simulate)
        if target_bit_depth == 24:
            # Add subtle dithering for 24-bit simulation
            dither_amount = 1.0 / (2**23)  # 24-bit quantization
            dither = np.random.normal(0, dither_amount, audio_np.shape) * 0.1
            audio_np = audio_np + dither
        elif target_bit_depth == 32:
            # 32-bit float - no quantization needed
            pass
        
        # 5. Normalize and limit
        audio_np = np.clip(audio_np, -1.0, 1.0)
        
        # Convert back to tensor
        if original_device is not None:
            upscaled_tensor = torch.from_numpy(audio_np).to(original_device).to(original_dtype)
        else:
            upscaled_tensor = audio_np
        
        print(f"✅ Audio upscaled successfully!")
        print(f"   Enhancement Level: {enhancement_level}")
        print(f"   Noise Reduction: {noise_reduction}")
        print(f"   High Freq Restore: {high_freq_restore}")
        
        return ({"waveform": upscaled_tensor, "sample_rate": target_sample_rate},)

# Registrazione nodo
NODE_CLASS_MAPPINGS = {
    "AudioUpscaler": AudioUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioUpscaler": "🔊 Audio Upscaler Pro"
}
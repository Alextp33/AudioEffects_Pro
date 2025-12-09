import torch
import numpy as np
from scipy import signal

class SimpleAudioDenoiser:
    """
    Simple Audio Denoiser - Stereo compatible, reliable, no complex processing
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "noise_reduction": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Amount of noise reduction (0=none, 1=maximum)"
                }),
                "high_freq_reduction": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Reduce high frequency noise/hiss"
                }),
                "low_freq_reduction": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Reduce low frequency hum/rumble"
                }),
            },
            "optional": {
                "preserve_quality": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "How much to preserve original vs cleaned (higher = more original)"
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Smooth out harsh artifacts"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "simple_denoise"
    CATEGORY = "audio/enhancement"
    
    def ensure_2d_audio(self, audio_np):
        """
        Garantisce che l'audio sia sempre in formato 2D (channels, samples)
        """
        if len(audio_np.shape) == 1:
            # Mono: (samples,) → (1, samples)
            return audio_np.reshape(1, -1)
        elif len(audio_np.shape) == 2:
            # Già 2D, verifica orientamento
            if audio_np.shape[0] > audio_np.shape[1]:
                # Probabilmente (samples, channels) → (channels, samples)
                return audio_np.T
            else:
                # Già (channels, samples)
                return audio_np
        elif len(audio_np.shape) == 3:
            # 3D: (batch, channels, samples) → (channels, samples)
            return audio_np.squeeze(0)
        else:
            # Shape complessa: flatten e rendi mono
            return audio_np.flatten().reshape(1, -1)
    
    def simple_highpass_filter(self, audio_channel, sample_rate, cutoff_hz=80):
        """
        Filtro passa-alto semplice per rimuovere rumble
        """
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff_hz / nyquist
            
            # Butterworth high-pass filter di ordine 2
            b, a = signal.butter(2, normalized_cutoff, btype='high')
            filtered = signal.filtfilt(b, a, audio_channel)
            return filtered
        except:
            # Fallback: ritorna originale se il filtro fallisce
            return audio_channel
    
    def simple_lowpass_filter(self, audio_channel, sample_rate, cutoff_hz=12000):
        """
        Filtro passa-basso semplice per rimuovere hiss
        """
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = min(cutoff_hz / nyquist, 0.99)  # Evita valori >= 1
            
            # Butterworth low-pass filter di ordine 2
            b, a = signal.butter(2, normalized_cutoff, btype='low')
            filtered = signal.filtfilt(b, a, audio_channel)
            return filtered
        except:
            # Fallback: ritorna originale
            return audio_channel
    
    def simple_noise_gate(self, audio_channel, threshold_db=-40):
        """
        Noise gate semplice - riduce parti sotto la soglia
        """
        # Converti threshold da dB a linear
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Calcola envelope del segnale
        envelope = np.abs(audio_channel)
        
        # Smooth envelope per evitare clicking
        window_size = min(1024, len(envelope) // 100)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            envelope = np.convolve(envelope, kernel, mode='same')
        
        # Crea mask per noise gate
        gate_mask = envelope > threshold_linear
        
        # Smooth le transizioni del gate
        gate_smooth = gate_mask.astype(float)
        if window_size > 1:
            smooth_kernel = np.ones(window_size // 4) / (window_size // 4)
            gate_smooth = np.convolve(gate_smooth, smooth_kernel, mode='same')
        
        # Applica gate
        gated_audio = audio_channel * gate_smooth
        
        return gated_audio
    
    def simple_smoothing(self, audio_channel, amount):
        """
        Smoothing semplice per ridurre artifacts
        """
        if amount <= 0:
            return audio_channel
        
        # Kernel size basato su amount
        kernel_size = int(amount * 10) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if kernel_size > 1:
            # Convoluzione con kernel gaussiano semplice
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(audio_channel, kernel, mode='same')
            
            # Blend con originale
            blend_factor = amount * 0.3  # Max 30% smoothing
            return audio_channel * (1 - blend_factor) + smoothed * blend_factor
        else:
            return audio_channel
    
    def simple_spectral_cleanup(self, audio_channel, sample_rate, reduction_amount):
        """
        Pulizia spettrale molto semplice e sicura
        """
        if reduction_amount <= 0:
            return audio_channel
        
        try:
            # FFT
            fft = np.fft.rfft(audio_channel)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Frequenze
            freqs = np.fft.rfftfreq(len(audio_channel), 1/sample_rate)
            
            # Riduzione selettiva per frequenze problematiche
            reduction_curve = np.ones_like(magnitude)
            
            # Riduci molto alte frequenze (hiss)
            high_freq_mask = freqs > 8000
            reduction_curve[high_freq_mask] *= (1 - reduction_amount * 0.3)
            
            # Riduci molto basse frequenze (rumble)
            low_freq_mask = freqs < 60
            reduction_curve[low_freq_mask] *= (1 - reduction_amount * 0.4)
            
            # Riduci frequenze "problematiche" (dove spesso c'è rumore)
            problem_freq_mask = (freqs > 12000) | (freqs < 40)
            reduction_curve[problem_freq_mask] *= (1 - reduction_amount * 0.2)
            
            # Applica riduzione
            clean_magnitude = magnitude * reduction_curve
            
            # Ricostruisci
            clean_fft = clean_magnitude * np.exp(1j * phase)
            clean_audio = np.fft.irfft(clean_fft, n=len(audio_channel))
            
            return clean_audio
            
        except Exception as e:
            print(f"⚠️ Spectral cleanup failed: {e}, returning original")
            return audio_channel
    
    def simple_denoise(self, audio, noise_reduction=0.3, high_freq_reduction=0.2, 
                      low_freq_reduction=0.1, preserve_quality=0.8, smoothing=0.2):
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Convert to numpy safely
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().numpy()
            original_device = waveform.device
            original_dtype = waveform.dtype
        else:
            audio_np = np.array(waveform)
            original_device = None
            original_dtype = None
        
        print(f"🔇 Simple denoising - Input shape: {audio_np.shape}")
        
        # Ensure 2D format (channels, samples)
        try:
            audio_2d = self.ensure_2d_audio(audio_np)
            num_channels, num_samples = audio_2d.shape
            print(f"   Processed as: {num_channels} channels, {num_samples} samples")
        except Exception as e:
            print(f"⚠️ Shape conversion failed: {e}, skipping denoising")
            return (audio,)  # Return original
        
        # Process each channel separately
        cleaned_channels = []
        
        for ch in range(num_channels):
            try:
                channel_audio = audio_2d[ch].copy()
                original_channel = channel_audio.copy()
                
                print(f"   Processing channel {ch+1}/{num_channels}")
                
                # 1. Low frequency cleanup (rumble removal)
                if low_freq_reduction > 0:
                    channel_audio = self.simple_highpass_filter(channel_audio, sample_rate, cutoff_hz=80)
                
                # 2. High frequency cleanup (hiss removal)
                if high_freq_reduction > 0:
                    channel_audio = self.simple_lowpass_filter(channel_audio, sample_rate, cutoff_hz=15000)
                
                # 3. Spectral cleanup
                if noise_reduction > 0:
                    channel_audio = self.simple_spectral_cleanup(channel_audio, sample_rate, noise_reduction)
                
                # 4. Noise gate (very gentle)
                if noise_reduction > 0.5:
                    channel_audio = self.simple_noise_gate(channel_audio, threshold_db=-45)
                
                # 5. Smoothing
                if smoothing > 0:
                    channel_audio = self.simple_smoothing(channel_audio, smoothing)
                
                # 6. Preserve original quality (blend)
                if preserve_quality > 0:
                    blend_factor = preserve_quality
                    channel_audio = channel_audio * (1 - blend_factor) + original_channel * blend_factor
                
                # 7. Ensure same length as original
                if len(channel_audio) != len(original_channel):
                    min_length = min(len(channel_audio), len(original_channel))
                    channel_audio = channel_audio[:min_length]
                
                cleaned_channels.append(channel_audio)
                
            except Exception as e:
                print(f"⚠️ Channel {ch} processing failed: {e}, using original")
                cleaned_channels.append(audio_2d[ch])
        
        # Combine channels
        try:
            if len(cleaned_channels) == 1:
                cleaned_audio = np.array(cleaned_channels[0])
                if len(cleaned_audio.shape) == 1:
                    cleaned_audio = cleaned_audio.reshape(1, -1)
            else:
                cleaned_audio = np.array(cleaned_channels)
            
            # Ensure output matches input format
            if len(audio_np.shape) == 1 and cleaned_audio.shape[0] == 1:
                # Original was mono, return mono
                cleaned_audio = cleaned_audio.flatten()
            elif len(audio_np.shape) == 3:
                # Original was 3D, return 3D
                cleaned_audio = cleaned_audio.reshape(1, *cleaned_audio.shape)
            
            print(f"   Output shape: {cleaned_audio.shape}")
            
        except Exception as e:
            print(f"⚠️ Channel combination failed: {e}, returning original")
            return (audio,)
        
        # Convert back to tensor
        try:
            if original_device is not None:
                cleaned_tensor = torch.from_numpy(cleaned_audio).to(original_device).to(original_dtype)
            else:
                cleaned_tensor = cleaned_audio
            
            # Final safety check
            if not isinstance(cleaned_tensor, torch.Tensor):
                cleaned_tensor = torch.tensor(cleaned_audio)
            
            if len(cleaned_tensor.shape) == 0:
                print("⚠️ Result is 0D tensor, returning original")
                return (audio,)
            
            print(f"✅ Simple denoising complete! Final tensor shape: {cleaned_tensor.shape}")
            
            return ({"waveform": cleaned_tensor, "sample_rate": sample_rate},)
            
        except Exception as e:
            print(f"⚠️ Tensor conversion failed: {e}, returning original")
            return (audio,)

# Registrazione nodo
NODE_CLASS_MAPPINGS = {
    "SimpleAudioDenoiser": SimpleAudioDenoiser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleAudioDenoiser": "🔇 Simple Audio Denoiser"
}
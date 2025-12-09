import torch
import torchaudio
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

class AudioDenoiser:
    """
    Audio Denoiser - Rimuove fruscio, rumore di fondo e artifacts
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "noise_reduction_method": (["Spectral Subtraction", "Wiener Filter", "Median Filter", "AI Enhanced", "Gentle Clean"], {
                    "default": "AI Enhanced"
                }),
                "noise_reduction_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1,
                    "tooltip": "Intensity of noise removal"
                }),
                "preserve_quality": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "How much to preserve original quality vs noise removal"
                }),
            },
            "optional": {
                "noise_floor_db": ("FLOAT", {
                    "default": -40.0, "min": -60.0, "max": -20.0, "step": 2.0,
                    "tooltip": "Noise floor threshold in dB"
                }),
                "high_freq_preserve": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Preserve high frequencies (avoid muffled sound)"
                }),
                "adaptive_threshold": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically adapt to audio content"
                }),
                "hiss_removal": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Specific removal of tape hiss and high-freq noise"
                }),
                "hum_removal": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Remove 50/60Hz electrical hum"
                }),
                "click_removal": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Remove clicks and pops"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("cleaned_audio", "noise_analysis")
    FUNCTION = "denoise_audio"
    CATEGORY = "audio/enhancement"
    
    def estimate_noise_profile(self, audio_np, sample_rate):
        """
        Stima il profilo del rumore dall'audio (FIXED per stereo)
        """
        # Gestione audio stereo/mono
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            # Stereo: usa il primo canale per analisi rumore
            analysis_audio = audio_np[0]
        else:
            # Mono o shape diversa
            analysis_audio = audio_np.flatten()
        
        # Usa i primi 0.5 secondi per stimare il rumore
        noise_samples = min(int(0.5 * sample_rate), len(analysis_audio) // 4)
        
        if noise_samples > 100:
            noise_segment = analysis_audio[:noise_samples]
        else:
            # Fallback: usa le parti più quiet
            rms_window = 1024
            if len(analysis_audio) > rms_window:
                # Calcola RMS per finestre
                num_windows = len(analysis_audio) // rms_window
                rms_values = []
                for i in range(num_windows):
                    start = i * rms_window
                    end = start + rms_window
                    window_rms = np.sqrt(np.mean(analysis_audio[start:end]**2))
                    rms_values.append(window_rms)
                
                # Prendi le finestre più quiet (percentile più basso)
                quiet_threshold = np.percentile(rms_values, 20)
                quiet_indices = [i for i, rms in enumerate(rms_values) if rms <= quiet_threshold]
                
                if quiet_indices:
                    # Combina le finestre quiet
                    noise_segments = []
                    for idx in quiet_indices[:3]:  # Max 3 finestre
                        start = idx * rms_window
                        end = start + rms_window
                        noise_segments.append(analysis_audio[start:end])
                    noise_segment = np.concatenate(noise_segments)
                else:
                    noise_segment = analysis_audio[:noise_samples] if noise_samples > 0 else analysis_audio[:1024]
            else:
                noise_segment = analysis_audio
        
        # Analisi spettrale del rumore
        if len(noise_segment) > 0:
            noise_fft = np.fft.rfft(noise_segment)
            noise_power = np.abs(noise_fft)**2
        else:
            # Fallback per casi estremi
            noise_power = np.ones(512) * 1e-10
        
        return noise_power
    
    def spectral_subtraction(self, audio_np, sample_rate, strength, noise_floor_db):
        """
        Rimozione rumore tramite sottrazione spettrale (FIXED)
        """
        # Stima profilo rumore
        noise_power = self.estimate_noise_profile(audio_np, sample_rate)
        
        # Processa ogni canale separatamente
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            # Stereo
            cleaned_channels = []
            for ch in range(audio_np.shape[0]):
                cleaned_ch = self._process_channel_spectral(audio_np[ch], noise_power, strength, noise_floor_db)
                cleaned_channels.append(cleaned_ch)
            cleaned_audio = np.array(cleaned_channels)
        else:
            # Mono
            audio_flat = audio_np.flatten()
            cleaned_audio = self._process_channel_spectral(audio_flat, noise_power, strength, noise_floor_db)
        
        return cleaned_audio
    
    def _process_channel_spectral(self, channel_audio, noise_power, strength, noise_floor_db):
        """
        Processa un singolo canale per sottrazione spettrale
        """
        # FFT del canale
        fft = np.fft.rfft(channel_audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Adatta noise_power alla lunghezza dell'FFT
        if len(noise_power) != len(magnitude):
            # Interpola o tronca
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, len(noise_power))
            new_indices = np.linspace(0, 1, len(magnitude))
            interpolator = interp1d(old_indices, noise_power, kind='linear', fill_value='extrapolate')
            noise_power_adjusted = interpolator(new_indices)
        else:
            noise_power_adjusted = noise_power
        
        # Soglia dinamica
        noise_threshold = noise_power_adjusted * (10 ** (noise_floor_db / 20))
        
        # Sottrazione spettrale con over-subtraction
        alpha = strength * 1.5  # Over-subtraction factor
        clean_magnitude = magnitude - alpha * np.sqrt(noise_threshold)
        
        # Floor per evitare artifacts
        clean_magnitude = np.maximum(clean_magnitude, 0.15 * magnitude)
        
        # Ricostruisci segnale
        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_audio = np.fft.irfft(clean_fft, n=len(channel_audio))
        
        return clean_audio
    
    def wiener_filter(self, audio_np, sample_rate, strength):
        """
        Filtro di Wiener per denoising (FIXED)
        """
        # Stima SNR
        noise_power = self.estimate_noise_profile(audio_np, sample_rate)
        
        # Processa ogni canale
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            # Stereo
            filtered_channels = []
            for ch in range(audio_np.shape[0]):
                filtered_ch = self._process_channel_wiener(audio_np[ch], noise_power, strength)
                filtered_channels.append(filtered_ch)
            filtered_audio = np.array(filtered_channels)
        else:
            # Mono
            audio_flat = audio_np.flatten()
            filtered_audio = self._process_channel_wiener(audio_flat, noise_power, strength)
        
        return filtered_audio
    
    def _process_channel_wiener(self, channel_audio, noise_power, strength):
        """
        Filtro Wiener per un singolo canale
        """
        fft = np.fft.rfft(channel_audio)
        signal_power = np.abs(fft)**2
        
        # Adatta noise_power
        if len(noise_power) != len(signal_power):
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, len(noise_power))
            new_indices = np.linspace(0, 1, len(signal_power))
            interpolator = interp1d(old_indices, noise_power, kind='linear', fill_value='extrapolate')
            noise_power_adjusted = interpolator(new_indices)
        else:
            noise_power_adjusted = noise_power
        
        # Wiener filter
        snr = signal_power / (noise_power_adjusted + 1e-10)
        wiener_gain = snr / (snr + strength)
        
        # Applica filtro
        filtered_fft = fft * wiener_gain
        filtered_audio = np.fft.irfft(filtered_fft, n=len(channel_audio))
        
        return filtered_audio
    
    def median_filter_denoise(self, audio_np, strength):
        """
        Filtro mediano per rimuovere clicks e pops (FIXED)
        """
        kernel_size = int(strength * 5) + 1  # 1-6 samples
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            # Stereo
            filtered_channels = []
            for ch in range(audio_np.shape[0]):
                filtered_ch = median_filter(audio_np[ch], size=kernel_size)
                filtered_channels.append(filtered_ch)
            filtered = np.array(filtered_channels)
        else:
            # Mono
            audio_flat = audio_np.flatten()
            filtered = median_filter(audio_flat, size=kernel_size)
        
        return filtered
    
    def remove_hum(self, audio_np, sample_rate, strength):
        """
        Rimuove ronzio elettrico (50/60Hz e armoniche) - FIXED
        """
        if strength <= 0:
            return audio_np
        
        # Frequenze da rimuovere (50Hz, 60Hz e armoniche)
        hum_freqs = [50, 60, 100, 120, 150, 180]
        
        # Processa ogni canale
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            # Stereo
            filtered_channels = []
            for ch in range(audio_np.shape[0]):
                filtered_ch = self._remove_hum_channel(audio_np[ch], sample_rate, hum_freqs, strength)
                filtered_channels.append(filtered_ch)
            filtered_audio = np.array(filtered_channels)
        else:
            # Mono
            audio_flat = audio_np.flatten()
            filtered_audio = self._remove_hum_channel(audio_flat, sample_rate, hum_freqs, strength)
        
        return filtered_audio
    
    def _remove_hum_channel(self, channel_audio, sample_rate, hum_freqs, strength):
        """
        Rimuove hum da un singolo canale
        """
        filtered_audio = channel_audio.copy()
        
        for freq in hum_freqs:
            if freq < sample_rate / 2:  # Nyquist check
                # Notch filter
                Q = 30  # Narrow filter
                w0 = 2 * np.pi * freq / sample_rate
                alpha = np.sin(w0) / (2 * Q)
                
                # Notch filter coefficients
                b0 = 1
                b1 = -2 * np.cos(w0)
                b2 = 1
                a0 = 1 + alpha
                a1 = -2 * np.cos(w0)
                a2 = 1 - alpha
                
                # Normalize
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                # Apply with strength control
                try:
                    filtered_freq = signal.lfilter(b, a, filtered_audio)
                    filtered_audio = filtered_audio * (1 - strength) + filtered_freq * strength
                except:
                    # Fallback se il filtro fallisce
                    pass
        
        return filtered_audio
    
    def remove_hiss(self, audio_np, sample_rate, strength):
        """
        Rimuove fruscio ad alta frequenza (FIXED)
        """
        if strength <= 0:
            return audio_np
        
        # Processa ogni canale
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            # Stereo
            dehissed_channels = []
            for ch in range(audio_np.shape[0]):
                dehissed_ch = self._remove_hiss_channel(audio_np[ch], sample_rate, strength)
                dehissed_channels.append(dehissed_ch)
            dehissed_audio = np.array(dehissed_channels)
        else:
            # Mono
            audio_flat = audio_np.flatten()
            dehissed_audio = self._remove_hiss_channel(audio_flat, sample_rate, strength)
        
        return dehissed_audio
    
    def _remove_hiss_channel(self, channel_audio, sample_rate, strength):
        """
        Rimuove hiss da un singolo canale
        """
        # High-frequency noise reduction
        fft = np.fft.rfft(channel_audio)
        freqs = np.fft.rfftfreq(len(channel_audio), 1/sample_rate)
        
        # Target high frequencies (above 8kHz)
        hiss_mask = freqs > 8000
        
        # Gentle high-frequency reduction
        reduction_factor = 1 - strength * 0.3  # Max 30% reduction
        fft[hiss_mask] *= reduction_factor
        
        # Back to time domain
        dehissed_audio = np.fft.irfft(fft, n=len(channel_audio))
        
        return dehissed_audio
    
    def ai_enhanced_denoise(self, audio_np, sample_rate, strength, preserve_quality):
        """
        Denoising avanzato che simula AI (FIXED)
        """
        # Multi-stage processing
        cleaned = audio_np.copy()
        
        # Stage 1: Spectral subtraction gentle
        cleaned = self.spectral_subtraction(cleaned, sample_rate, strength * 0.5, -35)
        
        # Stage 2: Frequency-dependent processing
        if len(cleaned.shape) == 2 and cleaned.shape[0] > 1:
            # Stereo
            enhanced_channels = []
            for ch in range(cleaned.shape[0]):
                enhanced_ch = self._ai_enhance_channel(cleaned[ch], sample_rate, strength, preserve_quality)
                enhanced_channels.append(enhanced_ch)
            cleaned = np.array(enhanced_channels)
        else:
            # Mono
            audio_flat = cleaned.flatten()
            cleaned = self._ai_enhance_channel(audio_flat, sample_rate, strength, preserve_quality)
        
        return cleaned
    
    def _ai_enhance_channel(self, channel_audio, sample_rate, strength, preserve_quality):
        """
        AI enhancement per un singolo canale
        """
        fft = np.fft.rfft(channel_audio)
        freqs = np.fft.rfftfreq(len(channel_audio), 1/sample_rate)
        
        # Frequency-dependent processing
        for i, freq in enumerate(freqs):
            if freq < 100:  # Low frequencies - gentle
                noise_reduction = strength * 0.3
            elif freq < 1000:  # Mid frequencies - moderate
                noise_reduction = strength * 0.5
            elif freq < 8000:  # High-mid - preserve speech
                noise_reduction = strength * 0.4
            else:  # High frequencies - aggressive on noise
                noise_reduction = strength * 0.7
            
            # Apply frequency-specific noise reduction
            signal_power = np.abs(fft[i])**2
            # Simple noise estimation
            noise_est = signal_power * 0.1  # Assume 10% is noise
            snr = signal_power / (noise_est + 1e-10)
            gain = snr / (snr + noise_reduction)
            fft[i] *= gain
        
        # Back to time domain
        enhanced_audio = np.fft.irfft(fft, n=len(channel_audio))
        
        return enhanced_audio
    
    def gentle_clean(self, audio_np, sample_rate, strength):
        """
        Pulizia molto leggera per audio già buoni (FIXED)
        """
        # Processa ogni canale
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            # Stereo
            cleaned_channels = []
            for ch in range(audio_np.shape[0]):
                cleaned_ch = self._gentle_clean_channel(audio_np[ch], strength)
                cleaned_channels.append(cleaned_ch)
            clean_audio = np.array(cleaned_channels)
        else:
            # Mono
            audio_flat = audio_np.flatten()
            clean_audio = self._gentle_clean_channel(audio_flat, strength)
        
        return clean_audio
    
    def _gentle_clean_channel(self, channel_audio, strength):
        """
        Gentle clean per un singolo canale
        """
        # Solo rimozione di noise floor molto basso
        fft = np.fft.rfft(channel_audio)
        magnitude = np.abs(fft)
        
        # Soglia molto bassa
        threshold = np.percentile(magnitude, 5) * strength
        
        # Soft thresholding
        clean_magnitude = np.where(magnitude < threshold, 
                                 magnitude * 0.7, 
                                 magnitude)
        
        # Ricostruisci
        phase = np.angle(fft)
        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_audio = np.fft.irfft(clean_fft, n=len(channel_audio))
        
        return clean_audio
    
    def analyze_noise(self, audio_np, sample_rate):
        """
        Analizza il contenuto di rumore (FIXED)
        """
        # Usa il primo canale per analisi se stereo
        if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
            analysis_audio = audio_np[0]
        else:
            analysis_audio = audio_np.flatten()
        
        # RMS analysis
        rms = np.sqrt(np.mean(analysis_audio**2))
        
        # Frequency analysis
        fft = np.fft.rfft(analysis_audio)
        freqs = np.fft.rfftfreq(len(analysis_audio), 1/sample_rate)
        magnitude = np.abs(fft)
        
        # Noise characteristics
        low_freq_noise = np.mean(magnitude[freqs < 100])
        mid_freq_noise = np.mean(magnitude[(freqs >= 100) & (freqs < 1000)])
        high_freq_noise = np.mean(magnitude[freqs >= 8000])
        
        analysis = f"Audio Analysis:\n"
        analysis += f"RMS Level: {20*np.log10(rms + 1e-10):.1f} dB\n"
        analysis += f"Low Freq Noise: {20*np.log10(low_freq_noise + 1e-10):.1f} dB\n"
        analysis += f"Mid Freq Noise: {20*np.log10(mid_freq_noise + 1e-10):.1f} dB\n"
        analysis += f"High Freq Noise: {20*np.log10(high_freq_noise + 1e-10):.1f} dB\n"
        analysis += f"Channels: {'Stereo' if len(audio_np.shape) == 2 and audio_np.shape[0] > 1 else 'Mono'}"
        
        return analysis
    
    def denoise_audio(self, audio, noise_reduction_method="AI Enhanced", noise_reduction_strength=0.5,
                     preserve_quality=0.8, noise_floor_db=-40.0, high_freq_preserve=0.7,
                     adaptive_threshold=True, hiss_removal=0.6, hum_removal=0.4, click_removal=0.3):
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().numpy()
            original_device = waveform.device
            original_dtype = waveform.dtype
        else:
            audio_np = waveform
            original_device = None
            original_dtype = None
        
        print(f"🔇 Denoising audio with method: {noise_reduction_method}")
        print(f"   Input shape: {audio_np.shape}")
        print(f"   Strength: {noise_reduction_strength}")
        print(f"   Preserve Quality: {preserve_quality}")
        
        # Analisi iniziale
        noise_analysis = self.analyze_noise(audio_np, sample_rate)
        
        # Applica metodo selezionato
        try:
            if noise_reduction_method == "Spectral Subtraction":
                cleaned_audio = self.spectral_subtraction(audio_np, sample_rate, noise_reduction_strength, noise_floor_db)
                
            elif noise_reduction_method == "Wiener Filter":
                cleaned_audio = self.wiener_filter(audio_np, sample_rate, noise_reduction_strength)
                
            elif noise_reduction_method == "Median Filter":
                cleaned_audio = self.median_filter_denoise(audio_np, noise_reduction_strength)
                
            elif noise_reduction_method == "AI Enhanced":
                cleaned_audio = self.ai_enhanced_denoise(audio_np, sample_rate, noise_reduction_strength, preserve_quality)
                
            elif noise_reduction_method == "Gentle Clean":
                cleaned_audio = self.gentle_clean(audio_np, sample_rate, noise_reduction_strength)
            
            # Applicazioni specifiche
            if hum_removal > 0:
                cleaned_audio = self.remove_hum(cleaned_audio, sample_rate, hum_removal)
                
            if hiss_removal > 0:
                cleaned_audio = self.remove_hiss(cleaned_audio, sample_rate, hiss_removal)
                
            if click_removal > 0:
                cleaned_audio = self.median_filter_denoise(cleaned_audio, click_removal * 0.5)
            
            # Preserva qualità originale se richiesto
            if preserve_quality > 0.8 and noise_reduction_method != "Gentle Clean":
                blend_factor = (preserve_quality - 0.8) * 5  # Scale 0.8-1.0 to 0-1
                if cleaned_audio.shape == audio_np.shape:
                    cleaned_audio = cleaned_audio * (1 - blend_factor) + audio_np * blend_factor
            
        except Exception as e:
            print(f"⚠️ Error in denoising: {e}")
            # Fallback: gentle processing
            cleaned_audio = self.gentle_clean(audio_np, sample_rate, noise_reduction_strength * 0.5)
        
        # Normalize
        cleaned_audio = np.clip(cleaned_audio, -1.0, 1.0)
        
        # Convert back to tensor
        if original_device is not None:
            cleaned_tensor = torch.from_numpy(cleaned_audio).to(original_device).to(original_dtype)
        else:
            cleaned_tensor = cleaned_audio
        
        # Update analysis
        noise_analysis += f"\n\nDenoising Applied:\n"
        noise_analysis += f"Method: {noise_reduction_method}\n"
        noise_analysis += f"Strength: {noise_reduction_strength}\n"
        if hum_removal > 0:
            noise_analysis += f"Hum Removal: {hum_removal}\n"
        if hiss_removal > 0:
            noise_analysis += f"Hiss Removal: {hiss_removal}\n"
        
        print(f"✅ Denoising complete!")
        print(f"   Output shape: {cleaned_audio.shape}")
        
        return (
            {"waveform": cleaned_tensor, "sample_rate": sample_rate},
            noise_analysis
        )

# Registrazione nodo
NODE_CLASS_MAPPINGS = {
    "AudioDenoiser": AudioDenoiser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioDenoiser": "🔇 Audio Denoiser Pro"
}
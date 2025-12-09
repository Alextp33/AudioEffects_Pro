import torch
import numpy as np

class AudioChannelSplitter:
    """
    Audio Channel Splitter - Separa canali stereo in Left/Right/Mono
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "output_mode": (["Left Only", "Right Only", "Mono Mix", "Separate Channels"], {
                    "default": "Separate Channels"
                }),
            },
            "optional": {
                "mono_mix_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Left/Right balance for mono mix (0.0=full left, 1.0=full right)"
                }),
                "channel_swap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap Left and Right channels"
                }),
                "phase_invert_right": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert phase of right channel"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("left_channel", "right_channel", "mono_mix", "channel_info")
    FUNCTION = "split_channels"
    CATEGORY = "audio/utility"
    
    def split_channels(self, audio, output_mode="Separate Channels", mono_mix_ratio=0.5, 
                      channel_swap=False, phase_invert_right=False):
        
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
        
        print(f"🎧 Splitting audio channels - Mode: {output_mode}")
        print(f"   Original shape: {audio_np.shape}")
        
        # Analyze input
        if len(audio_np.shape) == 1:
            # Mono input
            left_audio = audio_np
            right_audio = audio_np
            is_stereo = False
            channel_info = f"Input: Mono\nSamples: {len(audio_np)}\nDuration: {len(audio_np)/sample_rate:.2f}s"
        elif len(audio_np.shape) == 2:
            # Stereo input
            if audio_np.shape[0] == 2:
                is_stereo = True
                left_audio = audio_np[0]
                right_audio = audio_np[1]
                
                # Channel swap if requested
                if channel_swap:
                    left_audio, right_audio = right_audio, left_audio
                
                # Phase invert right if requested
                if phase_invert_right:
                    right_audio = -right_audio
                
                channel_info = f"Input: Stereo\nChannels: 2\nSamples per channel: {audio_np.shape[1]}\nDuration: {audio_np.shape[1]/sample_rate:.2f}s"
            else:
                # Unusual format
                left_audio = audio_np[0] if audio_np.shape[0] > 0 else np.zeros(audio_np.shape[1])
                right_audio = audio_np[1] if audio_np.shape[0] > 1 else left_audio
                is_stereo = audio_np.shape[0] > 1
                channel_info = f"Input: Multi-channel ({audio_np.shape[0]} channels)\nUsing first 2 channels"
        else:
            # Complex shape - flatten to stereo
            audio_flat = audio_np.reshape(-1)
            left_audio = right_audio = audio_flat
            is_stereo = False
            channel_info = f"Input: Complex shape {audio_np.shape}\nFlattened to mono"
        
        # Create mono mix
        if is_stereo:
            # Weighted mix based on ratio
            mono_audio = left_audio * (1 - mono_mix_ratio) + right_audio * mono_mix_ratio
        else:
            mono_audio = left_audio
        
        # Analysis
        if is_stereo:
            left_rms = np.sqrt(np.mean(left_audio**2))
            right_rms = np.sqrt(np.mean(right_audio**2))
            correlation = np.corrcoef(left_audio, right_audio)[0, 1]
            
            channel_info += f"\n\nChannel Analysis:"
            channel_info += f"\nLeft RMS: {20*np.log10(left_rms + 1e-10):.1f} dB"
            channel_info += f"\nRight RMS: {20*np.log10(right_rms + 1e-10):.1f} dB"
            channel_info += f"\nStereo Correlation: {correlation:.3f}"
            channel_info += f"\nStereo Width: {'Wide' if abs(correlation) < 0.8 else 'Narrow'}"
        
        # Convert back to tensor format
        def to_audio_format(audio_data):
            if original_device is not None:
                tensor = torch.from_numpy(audio_data).to(original_device).to(original_dtype)
            else:
                tensor = audio_data
            
            # Ensure proper shape for mono output
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)  # Add channel dimension
            
            return {"waveform": tensor, "sample_rate": sample_rate}
        
        # Prepare outputs based on mode
        if output_mode == "Left Only":
            main_output = to_audio_format(left_audio)
            left_out = main_output
            right_out = to_audio_format(np.zeros_like(left_audio))
            mono_out = main_output
            channel_info += f"\nOutput: Left channel only"
            
        elif output_mode == "Right Only":
            main_output = to_audio_format(right_audio)
            left_out = to_audio_format(np.zeros_like(right_audio))
            right_out = main_output
            mono_out = main_output
            channel_info += f"\nOutput: Right channel only"
            
        elif output_mode == "Mono Mix":
            main_output = to_audio_format(mono_audio)
            left_out = main_output
            right_out = main_output
            mono_out = main_output
            channel_info += f"\nOutput: Mono mix (L/R ratio: {1-mono_mix_ratio:.1f}/{mono_mix_ratio:.1f})"
            
        else:  # Separate Channels
            left_out = to_audio_format(left_audio)
            right_out = to_audio_format(right_audio)
            mono_out = to_audio_format(mono_audio)
            channel_info += f"\nOutput: Separate L/R channels + mono mix"
        
        channel_info += f"\nProcessing: {output_mode}"
        if channel_swap:
            channel_info += "\nChannels swapped"
        if phase_invert_right:
            channel_info += "\nRight phase inverted"
        
        print(f"✅ Channel splitting complete!")
        print(f"   Stereo input: {is_stereo}")
        print(f"   Output mode: {output_mode}")
        
        return (left_out, right_out, mono_out, channel_info)

class AudioChannelMixer:
    """
    Audio Channel Mixer - Combina canali separati in stereo
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_audio": ("AUDIO",),
                "right_audio": ("AUDIO",),
                "mix_mode": (["Stereo", "Mono Sum", "Mono Left", "Mono Right", "Side/Mid"], {
                    "default": "Stereo"
                }),
            },
            "optional": {
                "left_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Left channel gain"
                }),
                "right_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Right channel gain"
                }),
                "stereo_width": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Stereo field width (0=mono, 1=normal, 2=wide)"
                }),
                "phase_correlation": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Phase relationship between channels"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("mixed_audio", "mix_info")
    FUNCTION = "mix_channels"
    CATEGORY = "audio/utility"
    
    def mix_channels(self, left_audio, right_audio, mix_mode="Stereo", 
                    left_gain=1.0, right_gain=1.0, stereo_width=1.0, phase_correlation=0.0):
        
        # Get audio data
        left_waveform = left_audio["waveform"]
        right_waveform = right_audio["waveform"]
        sample_rate = left_audio["sample_rate"]
        
        # Convert to numpy
        if isinstance(left_waveform, torch.Tensor):
            left_np = left_waveform.cpu().numpy()
            right_np = right_waveform.cpu().numpy()
            original_device = left_waveform.device
            original_dtype = left_waveform.dtype
        else:
            left_np = left_waveform
            right_np = right_waveform
            original_device = None
            original_dtype = None
        
        # Ensure both channels are same length
        min_length = min(left_np.shape[-1], right_np.shape[-1])
        left_np = left_np[..., :min_length]
        right_np = right_np[..., :min_length]
        
        # Flatten to 1D if needed
        if len(left_np.shape) > 1:
            left_np = left_np.flatten()
        if len(right_np.shape) > 1:
            right_np = right_np.flatten()
        
        # Apply gains
        left_np *= left_gain
        right_np *= right_gain
        
        # Apply phase correlation
        if abs(phase_correlation) > 0.01:
            # Simple phase adjustment
            right_np = right_np * (1 - abs(phase_correlation)) + left_np * phase_correlation
        
        print(f"🎛️ Mixing channels - Mode: {mix_mode}")
        
        if mix_mode == "Stereo":
            # Standard stereo with width control
            if stereo_width != 1.0:
                # Mid/Side processing for width
                mid = (left_np + right_np) * 0.5
                side = (left_np - right_np) * 0.5 * stereo_width
                left_mixed = mid + side
                right_mixed = mid - side
            else:
                left_mixed = left_np
                right_mixed = right_np
            
            # Combine to stereo
            mixed_audio = np.array([left_mixed, right_mixed])
            mix_info = f"Stereo mix\nWidth: {stereo_width:.1f}\nL/R Gains: {left_gain:.1f}/{right_gain:.1f}"
            
        elif mix_mode == "Mono Sum":
            # Sum to mono
            mixed_audio = (left_np + right_np) * 0.5
            mix_info = f"Mono sum\nL/R Gains: {left_gain:.1f}/{right_gain:.1f}"
            
        elif mix_mode == "Mono Left":
            mixed_audio = left_np
            mix_info = f"Mono - Left channel only\nGain: {left_gain:.1f}"
            
        elif mix_mode == "Mono Right":
            mixed_audio = right_np
            mix_info = f"Mono - Right channel only\nGain: {right_gain:.1f}"
            
        elif mix_mode == "Side/Mid":
            # Mid/Side encoding
            mid = (left_np + right_np) * 0.5
            side = (left_np - right_np) * 0.5
            mixed_audio = np.array([mid, side])
            mix_info = f"Mid/Side encoding\nMid = (L+R)/2, Side = (L-R)/2"
        
        # Convert back to tensor
        if original_device is not None:
            mixed_tensor = torch.from_numpy(mixed_audio).to(original_device).to(original_dtype)
        else:
            mixed_tensor = mixed_audio
        
        # Ensure proper shape
        if len(mixed_tensor.shape) == 1:
            mixed_tensor = mixed_tensor.unsqueeze(0)
        
        mix_info += f"\nSample Rate: {sample_rate}Hz\nLength: {min_length} samples"
        
        print(f"✅ Channel mixing complete!")
        
        return (
            {"waveform": mixed_tensor, "sample_rate": sample_rate},
            mix_info
        )

# Registrazione nodi
NODE_CLASS_MAPPINGS = {
    "AudioChannelSplitter": AudioChannelSplitter,
    "AudioChannelMixer": AudioChannelMixer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioChannelSplitter": "🎧 Audio Channel Splitter",
    "AudioChannelMixer": "🎛️ Audio Channel Mixer"
}
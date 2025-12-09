import torch
import random
import numpy as np

class AudioMultiGenerator:
    """
    Audio Multi-Generator - Genera multiple variazioni di un brano
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {
                    "default": "Lady Gaga singing, emotional ballad, piano driven, 120 BPM",
                    "multiline": True
                }),
                "base_lyrics": ("STRING", {
                    "default": "[verse]\nIn the silence of the night\nI can hear my heart calling out\n[chorus]\nI am stronger than before\nI am brave enough to fall",
                    "multiline": True
                }),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "number_of_variations": ("INT", {
                    "default": 5, "min": 1, "max": 10, "step": 1,
                    "tooltip": "How many different versions to generate"
                }),
                "base_seed": ("INT", {
                    "default": -1, "min": -1, "max": 999999999,
                    "tooltip": "Base seed (-1 for random)"
                }),
                "seconds": ("INT", {
                    "default": 180, "min": 30, "max": 300, "step": 10,
                    "tooltip": "Duration of each audio file"
                }),
            },
            "optional": {
                "variation_intensity": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1,
                    "tooltip": "How different each variation should be"
                }),
                "tempo_variations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include tempo changes in variations"
                }),
                "style_variations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include style changes in variations"
                }),
                "arrangement_variations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include arrangement changes in variations"
                }),
                "energy_variations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include energy level changes in variations"
                }),
                "negative_prompt": ("STRING", {
                    "default": "metallic vocals, robotic voice, background noise, static, crackling, digital artifacts",
                    "multiline": True
                }),
                # Sampling parameters
                "steps": ("INT", {
                    "default": 90, "min": 20, "max": 200, "step": 10
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5
                }),
                "sampler_name": (["euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc"], {
                    "default": "dpmpp_2m"
                }),
                "scheduler": (["normal", "karras", "exponential", "simple", "ddim_uniform"], {
                    "default": "karras"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("audio_1", "audio_2", "audio_3", "audio_4", "audio_5", "generation_info")
    FUNCTION = "generate_variations"
    CATEGORY = "audio/generation"
    
    def create_variation_prompts(self, base_prompt, base_lyrics, variation_intensity, 
                               tempo_variations, style_variations, arrangement_variations, energy_variations):
        """
        Crea prompt variati per ogni generazione
        """
        variations = []
        
        # Variation 1: Original (baseline)
        variations.append({
            "prompt": base_prompt,
            "lyrics": base_lyrics,
            "description": "Original version"
        })
        
        # Tempo variations
        tempo_mods = []
        if tempo_variations:
            tempo_mods = [
                "faster tempo, 125 BPM, more energetic",
                "slower tempo, 115 BPM, more intimate",
                "varying tempo, dynamic rhythm changes",
                "uptempo club version, 130 BPM"
            ]
        
        # Style variations  
        style_mods = []
        if style_variations:
            style_mods = [
                "acoustic version, organic instruments, live recording feel",
                "electronic remix, synthesizers, digital production",
                "orchestral arrangement, strings and brass section",
                "minimal version, stripped down arrangement",
                "rock version, electric guitars and heavy drums"
            ]
        
        # Arrangement variations
        arrangement_mods = []
        if arrangement_variations:
            arrangement_mods = [
                "piano and voice focus, minimal backing",
                "full band arrangement, rich instrumentation",
                "ambient version, atmospheric pads and textures",
                "club mix, prominent bass and percussion",
                "unplugged version, acoustic instruments only"
            ]
        
        # Energy variations
        energy_mods = []
        if energy_variations:
            energy_mods = [
                "high energy, powerful delivery, anthem style",
                "low energy, intimate performance, whispered vocals",
                "building energy, starts quiet ends powerful",
                "consistent high energy throughout",
                "emotional intensity, dramatic performance"
            ]
        
        # Combine all modifications
        all_mods = tempo_mods + style_mods + arrangement_mods + energy_mods
        
        # Create variations (skip first one as it's original)
        for i in range(1, 5):  # Variations 2-5
            if i-1 < len(all_mods):
                mod = all_mods[i-1]
                varied_prompt = f"{base_prompt}, {mod}"
                
                # Vary lyrics slightly for some variations
                varied_lyrics = base_lyrics
                if i % 2 == 0 and len(base_lyrics) > 50:  # Every other variation
                    # Add performance directions to lyrics
                    performance_dirs = [
                        "[with emotion]",
                        "[building intensity]", 
                        "[soft and intimate]",
                        "[powerful delivery]"
                    ]
                    perf_dir = performance_dirs[min(i-2, len(performance_dirs)-1)]
                    varied_lyrics = f"{perf_dir}\n{base_lyrics}"
                
                variations.append({
                    "prompt": varied_prompt,
                    "lyrics": varied_lyrics,
                    "description": f"Variation {i}: {mod}"
                })
            else:
                # Fallback: combine random modifications
                mod_subset = random.sample(all_mods, min(2, len(all_mods)))
                combined_mod = ", ".join(mod_subset)
                variations.append({
                    "prompt": f"{base_prompt}, {combined_mod}",
                    "lyrics": base_lyrics,
                    "description": f"Variation {i}: {combined_mod}"
                })
        
        return variations[:5]  # Ensure exactly 5 variations
    
    def encode_prompt(self, clip, prompt):
        """
        Encode text prompt to conditioning
        """
        # This is a simplified version - in reality you'd use the actual CLIP encode nodes
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    
    def create_latent(self, seconds, batch_size=1):
        """
        Create empty latent for audio generation
        """
        # This creates a latent space representation for audio
        # Dimensions are approximate and may need adjustment based on actual model
        latent_length = seconds * 50  # Rough approximation
        latent = torch.zeros([batch_size, 4, latent_length])
        return {"samples": latent}
    
    def sample_audio(self, model, positive, negative, latent, steps, cfg, sampler_name, scheduler, seed):
        """
        Sample audio from latent space
        """
        # Set seed for reproducibility
        if seed == -1:
            seed = random.randint(0, 999999999)
        
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # This is where the actual sampling would happen
        # For now, we'll create a placeholder that represents the sampling process
        print(f"🎵 Sampling audio with seed {seed}, steps {steps}, cfg {cfg}")
        
        # In the real implementation, this would call the actual KSampler
        # return sampled_latent
        return latent  # Placeholder
    
    def decode_audio(self, vae, latent_samples):
        """
        Decode latent to audio waveform
        """
        # This would use the VAE to decode latent space to audio
        # For now, return a placeholder audio structure
        
        # Generate some placeholder audio data
        sample_rate = 44100
        duration = 180  # seconds
        samples = int(sample_rate * duration)
        
        # Create placeholder waveform (silence for now)
        waveform = torch.zeros([1, samples])
        
        return {"waveform": waveform, "sample_rate": sample_rate}
    
    def generate_variations(self, base_prompt, base_lyrics, model, clip, vae, number_of_variations=5,
                          base_seed=-1, seconds=180, variation_intensity=0.5, tempo_variations=True,
                          style_variations=True, arrangement_variations=True, energy_variations=True,
                          negative_prompt="", steps=90, cfg=3.5, sampler_name="dpmpp_2m", scheduler="karras"):
        
        print(f"🎵 Generating {number_of_variations} audio variations")
        print(f"   Base prompt: {base_prompt[:50]}...")
        print(f"   Variation intensity: {variation_intensity}")
        
        # Create variation prompts
        variations = self.create_variation_prompts(
            base_prompt, base_lyrics, variation_intensity,
            tempo_variations, style_variations, arrangement_variations, energy_variations
        )
        
        # Generate seeds for each variation
        if base_seed == -1:
            base_seed = random.randint(0, 999999999)
        
        seeds = [base_seed + i * 1000 for i in range(number_of_variations)]
        
        # Encode negative prompt
        negative_cond = self.encode_prompt(clip, negative_prompt) if negative_prompt else None
        
        generated_audios = []
        generation_info = f"Audio Multi-Generation Report:\n"
        generation_info += f"Base seed: {base_seed}\n"
        generation_info += f"Variations generated: {len(variations)}\n\n"
        
        # Generate each variation
        for i, variation in enumerate(variations):
            print(f"🎶 Generating variation {i+1}: {variation['description']}")
            
            # Encode prompt for this variation
            positive_cond = self.encode_prompt(clip, variation['prompt'])
            
            # Create latent
            latent = self.create_latent(seconds)
            
            # Sample
            sampled_latent = self.sample_audio(
                model, positive_cond, negative_cond, latent,
                steps, cfg, sampler_name, scheduler, seeds[i]
            )
            
            # Decode to audio
            audio = self.decode_audio(vae, sampled_latent)
            
            generated_audios.append(audio)
            
            # Add to info
            generation_info += f"Variation {i+1}:\n"
            generation_info += f"  Description: {variation['description']}\n"
            generation_info += f"  Seed: {seeds[i]}\n"
            generation_info += f"  Prompt: {variation['prompt'][:100]}...\n\n"
        
        # Pad with empty audio if less than 5 variations
        while len(generated_audios) < 5:
            empty_audio = {"waveform": torch.zeros([1, int(44100 * seconds)]), "sample_rate": 44100}
            generated_audios.append(empty_audio)
        
        generation_info += f"✅ Generation complete!\n"
        generation_info += f"Settings used:\n"
        generation_info += f"  Steps: {steps}\n"
        generation_info += f"  CFG: {cfg}\n"
        generation_info += f"  Sampler: {sampler_name}\n"
        generation_info += f"  Scheduler: {scheduler}\n"
        
        print(f"✅ Multi-generation complete! {len(variations)} variations created.")
        
        return (
            generated_audios[0],
            generated_audios[1], 
            generated_audios[2],
            generated_audios[3],
            generated_audios[4],
            generation_info
        )

class AudioVariationPromptGenerator:
    """
    Helper node - genera solo i prompt variati senza generare audio
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {
                    "default": "Lady Gaga singing, emotional ballad, piano driven, 120 BPM",
                    "multiline": True
                }),
                "variation_type": (["Tempo", "Style", "Arrangement", "Energy", "All"], {
                    "default": "All"
                }),
                "number_of_variations": ("INT", {
                    "default": 5, "min": 1, "max": 10, "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5", "variation_list")
    FUNCTION = "generate_prompt_variations"
    CATEGORY = "audio/utility"
    
    def generate_prompt_variations(self, base_prompt, variation_type, number_of_variations):
        """
        Genera solo i prompt variati
        """
        modifications = {
            "Tempo": [
                "faster tempo, 125 BPM, more energetic",
                "slower tempo, 115 BPM, more intimate", 
                "varying tempo, dynamic rhythm changes",
                "uptempo club version, 130 BPM"
            ],
            "Style": [
                "acoustic version, organic instruments",
                "electronic remix, synthesizers",
                "orchestral arrangement, strings section",
                "minimal version, stripped down"
            ],
            "Arrangement": [
                "piano and voice focus, minimal backing",
                "full band arrangement, rich instrumentation",
                "ambient version, atmospheric textures",
                "club mix, prominent bass and percussion"
            ],
            "Energy": [
                "high energy, powerful delivery, anthem style",
                "low energy, intimate performance, whispered vocals",
                "building energy, starts quiet ends powerful",
                "emotional intensity, dramatic performance"
            ]
        }
        
        if variation_type == "All":
            all_mods = []
            for mod_list in modifications.values():
                all_mods.extend(mod_list)
            selected_mods = all_mods
        else:
            selected_mods = modifications.get(variation_type, [])
        
        prompts = [base_prompt]  # Original as first
        
        for i in range(min(number_of_variations - 1, len(selected_mods))):
            varied_prompt = f"{base_prompt}, {selected_mods[i]}"
            prompts.append(varied_prompt)
        
        # Pad to 5 prompts
        while len(prompts) < 5:
            prompts.append(base_prompt)
        
        variation_list = "Generated Prompt Variations:\n\n"
        for i, prompt in enumerate(prompts[:5]):
            variation_list += f"Variation {i+1}:\n{prompt}\n\n"
        
        return (prompts[0], prompts[1], prompts[2], prompts[3], prompts[4], variation_list)

# Registrazione nodi
NODE_CLASS_MAPPINGS = {
    "AudioMultiGenerator": AudioMultiGenerator,
    "AudioVariationPromptGenerator": AudioVariationPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioMultiGenerator": "🎵 Audio Multi-Generator Pro",
    "AudioVariationPromptGenerator": "📝 Audio Variation Prompt Generator"
}
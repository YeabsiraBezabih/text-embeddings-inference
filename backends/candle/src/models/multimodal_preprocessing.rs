use candle::{Device, Result, Tensor};

pub const DEFAULT_PATCH_SIZE: usize = 16;
pub const DEFAULT_MAX_SOFT_TOKENS: usize = 280;
pub const DEFAULT_POOLING_KERNEL_SIZE: usize = 3;
pub const DEFAULT_MAX_PATCHES: usize = DEFAULT_MAX_SOFT_TOKENS * DEFAULT_POOLING_KERNEL_SIZE * DEFAULT_POOLING_KERNEL_SIZE; // 2520

pub const IMAGE_START_TOKEN_ID: u32 = 255999; // <|image>
pub const IMAGE_TOKEN_ID: u32 = 258880;       // <|image|>
pub const IMAGE_END_TOKEN_ID: u32 = 258882;   // <image|>

pub const AUDIO_START_TOKEN_ID: u32 = 256000; // <|audio>
pub const AUDIO_TOKEN_ID: u32 = 258881;       // <|audio|>
pub const AUDIO_END_TOKEN_ID: u32 = 258883;   // <audio|>

pub const VIDEO_TOKEN_ID: u32 = 258884;       // <|video|>

/// Computes the aspect-ratio-preserving resized dimensions matching Gemma 4 image processor.
pub fn get_aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> (usize, usize) {
    let side_mult = pooling_kernel_size * patch_size; // 3 * 16 = 48
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();

    let mut target_h = (((factor * height as f64) / side_mult as f64).floor() as usize) * side_mult;
    let mut target_w = (((factor * width as f64) / side_mult as f64).floor() as usize) * side_mult;

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;

    if target_h == 0 && target_w == 0 {
        target_h = side_mult;
        target_w = side_mult;
    } else if target_h == 0 {
        target_h = side_mult;
        target_w = std::cmp::min((width / height) * side_mult, max_side_length);
    } else if target_w == 0 {
        target_w = side_mult;
        target_h = std::cmp::min((height / width) * side_mult, max_side_length);
    }

    (target_h, target_w)
}

/// Patchifies HWC image data into flattened patches and 2D meshgrid position IDs.
/// Output patches: [max_patches, patch_size * patch_size * 3]
/// Output positions: [max_patches, 2] with (-1, -1) for padding.
pub fn patchify(
    hwc_data: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> (Vec<f32>, Vec<i64>, usize) {
    let num_patches_h = height / patch_size;
    let num_patches_w = width / patch_size;
    let patch_dim = patch_size * patch_size * channels;

    let mut patch_data = vec![0.0f32; max_patches * patch_dim];
    let mut out_idx = 0;

    for ph in 0..num_patches_h {
        for pw in 0..num_patches_w {
            for dy in 0..patch_size {
                let row_offset = (ph * patch_size + dy) * width * channels + pw * patch_size * channels;
                for dx in 0..patch_size {
                    let src = row_offset + dx * channels;
                    for c in 0..channels {
                        if src + c < hwc_data.len() {
                            patch_data[out_idx] = hwc_data[src + c];
                        }
                        out_idx += 1;
                    }
                }
            }
        }
    }

    let mut pos_data = vec![-1i64; max_patches * 2];
    let mut p_idx = 0;
    for row in 0..num_patches_h {
        for col in 0..num_patches_w {
            pos_data[p_idx] = col as i64;
            pos_data[p_idx + 1] = row as i64;
            p_idx += 2;
        }
    }

    let num_soft_tokens = (num_patches_h * num_patches_w) / (pooling_kernel_size * pooling_kernel_size);
    (patch_data, pos_data, num_soft_tokens)
}

/// Helper to convert patch data into Candle tensors ready for Vision Tower.
pub fn create_vision_tensors(
    patch_data: Vec<f32>,
    pos_data: Vec<i64>,
    max_patches: usize,
    patch_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let pixel_values = Tensor::from_vec(patch_data, (1, max_patches, patch_dim), device)?;
    let pixel_positions = Tensor::from_vec(pos_data, (1, max_patches, 2), device)?;
    Ok((pixel_values, pixel_positions))
}

/// Computes log-mel filterbank features from raw mono 16kHz audio samples.
/// Produces a (num_frames, 128) feature tensor.
pub fn extract_fbank_features(
    samples: &[f32],
    sample_rate: usize,
    num_mel_bins: usize,
    frame_length_ms: f32,
    hop_length_ms: f32,
) -> (Vec<f32>, usize) {
    let frame_size = ((sample_rate as f32 * frame_length_ms) / 1000.0) as usize;
    let hop_size = ((sample_rate as f32 * hop_length_ms) / 1000.0) as usize;
    let n_fft = 512;

    if samples.is_empty() || frame_size == 0 || hop_size == 0 {
        return (vec![0.0f32; num_mel_bins], 1);
    }

    let num_frames = if samples.len() >= frame_size {
        (samples.len() - frame_size) / hop_size + 1
    } else {
        1
    };

    // Precompute Mel filterbank weights
    let f_min = 0.0f32;
    let f_max = (sample_rate as f32) / 2.0;
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity(num_mel_bins + 2);
    for i in 0..=(num_mel_bins + 1) {
        let m = mel_min + (i as f32 / (num_mel_bins + 1) as f32) * (mel_max - mel_min);
        mel_points.push(mel_to_hz(m));
    }

    let mut fbank = vec![vec![0.0f32; n_fft / 2 + 1]; num_mel_bins];
    for m in 0..num_mel_bins {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for k in 0..=(n_fft / 2) {
            let freq = (k as f32 * sample_rate as f32) / n_fft as f32;
            if freq >= f_left && freq <= f_center {
                fbank[m][k] = (freq - f_left) / (f_center - f_left);
            } else if freq > f_center && freq <= f_right {
                fbank[m][k] = (f_right - freq) / (f_right - f_center);
            }
        }
    }

    // Hann window
    let mut window = Vec::with_capacity(frame_size);
    for i in 0..frame_size {
        let val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos());
        window.push(val);
    }

    let mut output = vec![0.0f32; num_frames * num_mel_bins];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let mut frame_fft_mag = vec![0.0f32; n_fft / 2 + 1];

        // Compute power spectrum
        for k in 0..=(n_fft / 2) {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for n in 0..frame_size {
                let s = if start + n < samples.len() {
                    samples[start + n] * window[n]
                } else {
                    0.0
                };
                let angle = -2.0 * std::f32::consts::PI * (k * n) as f32 / n_fft as f32;
                real += s * angle.cos();
                imag += s * angle.sin();
            }
            frame_fft_mag[k] = real * real + imag * imag;
        }

        // Filterbank multiplication and log compression
        for m in 0..num_mel_bins {
            let mut energy = 1e-6f32;
            for k in 0..=(n_fft / 2) {
                energy += frame_fft_mag[k] * fbank[m][k];
            }
            output[frame_idx * num_mel_bins + m] = energy.ln();
        }
    }

    (output, num_frames)
}

/// Helper to create prompt token sequence for multimodal inputs.
pub fn build_multimodal_tokens(
    text_prefix: &[u32],
    num_images: usize,
    num_audio_tokens: usize,
    text_suffix: &[u32],
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(
        text_prefix.len()
            + num_images * (DEFAULT_MAX_SOFT_TOKENS + 2)
            + (if num_audio_tokens > 0 { num_audio_tokens + 2 } else { 0 })
            + text_suffix.len(),
    );

    tokens.extend_from_slice(text_prefix);

    for _ in 0..num_images {
        tokens.push(IMAGE_START_TOKEN_ID);
        for _ in 0..DEFAULT_MAX_SOFT_TOKENS {
            tokens.push(IMAGE_TOKEN_ID);
        }
        tokens.push(IMAGE_END_TOKEN_ID);
    }

    if num_audio_tokens > 0 {
        tokens.push(AUDIO_START_TOKEN_ID);
        for _ in 0..num_audio_tokens {
            tokens.push(AUDIO_TOKEN_ID);
        }
        tokens.push(AUDIO_END_TOKEN_ID);
    }

    tokens.extend_from_slice(text_suffix);
    tokens
}

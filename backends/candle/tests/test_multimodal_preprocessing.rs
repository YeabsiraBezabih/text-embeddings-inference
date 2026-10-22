use candle::Device;
use text_embeddings_backend_candle::models::multimodal_preprocessing::*;

#[test]
fn test_aspect_ratio_preserving_size_square() {
    let (h, w) = get_aspect_ratio_preserving_size(512, 512, 16, 2520, 3);
    assert_eq!(h % 48, 0, "Height must be a multiple of side_mult (48)");
    assert_eq!(w % 48, 0, "Width must be a multiple of side_mult (48)");
    assert_eq!(h, w, "Square image should remain square");
    let patches = (h / 16) * (w / 16);
    assert!(patches <= 2520, "Total patches must not exceed max_patches");
}

#[test]
fn test_aspect_ratio_preserving_size_wide_aspect() {
    let (h, w) = get_aspect_ratio_preserving_size(1080, 1920, 16, 2520, 3);
    assert_eq!(h % 48, 0);
    assert_eq!(w % 48, 0);
    assert!(w > h, "Wide image must have width > height");
    let patches = (h / 16) * (w / 16);
    assert!(patches <= 2520);
}

#[test]
fn test_aspect_ratio_preserving_size_tall_aspect() {
    let (h, w) = get_aspect_ratio_preserving_size(1920, 1080, 16, 2520, 3);
    assert_eq!(h % 48, 0);
    assert_eq!(w % 48, 0);
    assert!(h > w, "Tall image must have height > width");
    let patches = (h / 16) * (w / 16);
    assert!(patches <= 2520);
}

#[test]
fn test_aspect_ratio_preserving_size_tiny_image() {
    let (h, w) = get_aspect_ratio_preserving_size(10, 10, 16, 2520, 3);
    assert_eq!(h, 768);
    assert_eq!(w, 768);
    let patches = (h / 16) * (w / 16);
    assert!(patches <= 2520);
}

#[test]
fn test_patchify_dimensions_and_counts() {
    let h = 48;
    let w = 48;
    let c = 3;
    let total_pixels = h * w * c;
    let dummy_pixels: Vec<f32> = (0..total_pixels).map(|i| (i as f32) / (total_pixels as f32)).collect();

    let (patches, positions, num_soft_tokens) = patchify(
        &dummy_pixels,
        h,
        w,
        c,
        16,
        DEFAULT_MAX_PATCHES,
        3,
    );

    let patch_dim = 16 * 16 * 3; // 768
    assert_eq!(patches.len(), DEFAULT_MAX_PATCHES * patch_dim);
    assert_eq!(positions.len(), DEFAULT_MAX_PATCHES * 2);

    // 48/16 = 3, 3*3 = 9 patches. 9 / (3*3) = 1 soft token.
    assert_eq!(num_soft_tokens, 1);

    // Verify first 9 patch positions are valid (col, row)
    for i in 0..9 {
        let col = positions[i * 2];
        let row = positions[i * 2 + 1];
        assert!(col >= 0 && col < 3);
        assert!(row >= 0 && row < 3);
    }

    // Verify padding patch positions are -1
    for i in 9..DEFAULT_MAX_PATCHES {
        assert_eq!(positions[i * 2], -1);
        assert_eq!(positions[i * 2 + 1], -1);
    }
}

#[test]
fn test_create_vision_tensors() {
    let device = Device::Cpu;
    let patches = vec![0.5f32; 10 * 768];
    let positions = vec![0i64; 10 * 2];

    let (val_t, pos_t) = create_vision_tensors(patches, positions, 10, 768, &device).unwrap();
    assert_eq!(val_t.dims3().unwrap(), (1, 10, 768));
    assert_eq!(pos_t.dims3().unwrap(), (1, 10, 2));
}

#[test]
fn test_extract_fbank_features_dimensions() {
    // 1 second of 16kHz audio = 16,000 samples
    let sample_rate = 16000;
    let duration_sec = 1.0f32;
    let total_samples = (sample_rate as f32 * duration_sec) as usize;
    let mut samples = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        // 440 Hz sine wave
        let t = i as f32 / sample_rate as f32;
        samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin());
    }

    let (fbank, num_frames) = extract_fbank_features(
        &samples,
        sample_rate,
        128,
        25.0, // 25ms frame
        10.0, // 10ms hop
    );

    assert!(num_frames >= 90 && num_frames <= 105, "Expected ~100 frames for 1s audio at 10ms hop, got {num_frames}");
    assert_eq!(fbank.len(), num_frames * 128);

    // Check no NaNs or Infs
    for &val in &fbank {
        assert!(!val.is_nan(), "Fbank feature contains NaN");
        assert!(!val.is_infinite(), "Fbank feature contains Inf");
    }
}

#[test]
fn test_extract_fbank_features_silence() {
    let samples = vec![0.0f32; 3200]; // 200ms of silence
    let (fbank, num_frames) = extract_fbank_features(
        &samples,
        16000,
        128,
        25.0,
        10.0,
    );
    assert!(num_frames > 0);
    assert_eq!(fbank.len(), num_frames * 128);
    for &val in &fbank {
        assert!(!val.is_nan());
        assert!(!val.is_infinite());
    }
}

#[test]
fn test_build_multimodal_tokens_structure() {
    let prefix = vec![2, 101]; // BOS, query token
    let suffix = vec![1];      // EOS

    // Single image prompt
    let tokens = build_multimodal_tokens(&prefix, 1, 0, &suffix);
    // prefix (2) + BOI (1) + 280 image tokens + EOI (1) + suffix (1) = 285
    assert_eq!(tokens.len(), 2 + 1 + 280 + 1 + 1);
    assert_eq!(tokens[0], 2);
    assert_eq!(tokens[1], 101);
    assert_eq!(tokens[2], IMAGE_START_TOKEN_ID);
    assert_eq!(tokens[3], IMAGE_TOKEN_ID);
    assert_eq!(tokens[282], IMAGE_TOKEN_ID);
    assert_eq!(tokens[283], IMAGE_END_TOKEN_ID);
    assert_eq!(tokens[284], 1);

    // Audio prompt with 50 audio soft tokens
    let audio_tokens = build_multimodal_tokens(&prefix, 0, 50, &suffix);
    // prefix (2) + BOA (1) + 50 audio tokens + EOA (1) + suffix (1) = 55
    assert_eq!(audio_tokens.len(), 2 + 1 + 50 + 1 + 1);
    assert_eq!(audio_tokens[2], AUDIO_START_TOKEN_ID);
    assert_eq!(audio_tokens[3], AUDIO_TOKEN_ID);
    assert_eq!(audio_tokens[52], AUDIO_TOKEN_ID);
    assert_eq!(audio_tokens[53], AUDIO_END_TOKEN_ID);
}

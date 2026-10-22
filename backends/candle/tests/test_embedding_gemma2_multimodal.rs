use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::fs::File;
use std::path::PathBuf;
use text_embeddings_backend_candle::models::embedding_gemma2::{
    EmbeddingGemma2ConfigWrapper, EmbeddingGemma2Model,
};
use text_embeddings_backend_candle::models::multimodal_preprocessing::*;
use text_embeddings_backend_core::{ModelType, Pool};

fn find_snapshot_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots_path = PathBuf::from(format!(
        "{home}/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots"
    ));
    let mut entries = std::fs::read_dir(snapshots_path).ok()?;
    while let Some(Ok(entry)) = entries.next() {
        if entry.file_type().ok()?.is_dir() {
            return Some(entry.path());
        }
    }
    None
}

#[test]
fn test_real_weights_multimodal_embeddinggemma2() -> Result<()> {
    let snapshot_dir = match find_snapshot_dir() {
        Some(dir) => dir,
        None => {
            eprintln!("Skipping test_real_weights_multimodal_embeddinggemma2: snapshot not found");
            return Ok(());
        }
    };

    let config_path = snapshot_dir.join("config.json");
    let weights_path = snapshot_dir.join("model.safetensors");

    if !config_path.exists() || !weights_path.exists() {
        eprintln!("Skipping test: config or weights file missing");
        return Ok(());
    }

    let device = Device::Cpu;
    let config_file = File::open(&config_path)?;
    let wrapper: EmbeddingGemma2ConfigWrapper = serde_json::from_reader(config_file)?;
    let text_config = wrapper.text_config();

    // Memory map the real safetensors weights via file path
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
    };

    println!("Loading EmbeddingGemma2Model with multimodal towers from real weights...");
    let model = EmbeddingGemma2Model::load(vb, text_config, ModelType::Embedding(Pool::Mean))?;

    assert!(model.vision_tower.is_some(), "vision_tower must be loaded from real weights");
    assert!(model.embed_vision.is_some(), "embed_vision must be loaded from real weights");
    assert!(model.audio_tower.is_some(), "audio_tower must be loaded from real weights");
    assert!(model.embed_audio.is_some(), "embed_audio must be loaded from real weights");

    // 1. Text-Only Forward Pass
    let input_ids = Tensor::from_vec(vec![2u32, 101, 102, 103, 1], (1, 5), &device)?;
    let position_ids = Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1, 5), &device)?;

    let text_emb = model.forward_multimodal(
        &input_ids,
        &position_ids,
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    assert_eq!(text_emb.dims2()?, (1, 768), "Text embedding must have shape (1, 768)");
    let norm = text_emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    assert!((norm - 1.0).abs() < 1e-4, "Embedding must be L2-normalized: got norm {norm}");

    // 2. Multimodal Text + Image Forward Pass
    let image_tokens = build_multimodal_tokens(&[2], 1, 0, &[101, 1]); // 284 tokens
    let num_tokens = image_tokens.len();
    let mm_input_ids = Tensor::from_vec(image_tokens, (1, num_tokens), &device)?;
    let mm_position_ids = Tensor::from_vec((0..num_tokens as u32).collect::<Vec<_>>(), (1, num_tokens), &device)?;

    // Dummy 48x48 image patchified
    let dummy_pixels = vec![0.5f32; 48 * 48 * 3];
    let (patches, positions, _) = patchify(&dummy_pixels, 48, 48, 3, 16, DEFAULT_MAX_PATCHES, 3);
    let (pixel_values, pixel_position_ids) = create_vision_tensors(
        patches,
        positions,
        DEFAULT_MAX_PATCHES,
        768,
        &device,
    )?;

    println!("Running text + image multimodal forward pass...");
    let image_emb = model.forward_multimodal(
        &mm_input_ids,
        &mm_position_ids,
        Some(&pixel_values),
        Some(&pixel_position_ids),
        None,
        None,
        None,
        None,
    )?;

    assert_eq!(image_emb.dims2()?, (1, 768), "Image embedding must have shape (1, 768)");
    let img_norm = image_emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    assert!((img_norm - 1.0).abs() < 1e-4, "Image embedding must be L2-normalized: got norm {img_norm}");

    // Verify embeddings are finite and non-trivial
    let img_vec = image_emb.to_vec2::<f32>()?;
    for &val in &img_vec[0] {
        assert!(!val.is_nan(), "Output vector contains NaN");
        assert!(!val.is_infinite(), "Output vector contains Inf");
    }

    // 3. Multimodal Text + Video (2 frames) Forward Pass
    let video_tokens = vec![2u32, 258884, 258884, 101, 1]; // Video placeholders
    let vid_len = video_tokens.len();
    let vid_input_ids = Tensor::from_vec(video_tokens, (1, vid_len), &device)?;
    let vid_position_ids = Tensor::from_vec((0..vid_len as u32).collect::<Vec<_>>(), (1, vid_len), &device)?;

    println!("Running text + video multimodal forward pass...");
    let video_emb = model.forward_multimodal(
        &vid_input_ids,
        &vid_position_ids,
        None,
        None,
        Some(&pixel_values),
        Some(&pixel_position_ids),
        None,
        None,
    )?;

    assert_eq!(video_emb.dims2()?, (1, 768));
    let vid_norm = video_emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    assert!((vid_norm - 1.0).abs() < 1e-4);

    // 4. Multimodal Text + Audio Forward Pass
    let dummy_audio = vec![0.1f32; 16000]; // 1s audio
    let (fbank, num_frames) = extract_fbank_features(&dummy_audio, 16000, 128, 25.0, 10.0);
    let input_features = Tensor::from_vec(fbank, (1, num_frames, 128), &device)?;

    let audio_tokens = build_multimodal_tokens(&[2], 0, 10, &[101, 1]);
    let aud_len = audio_tokens.len();
    let aud_input_ids = Tensor::from_vec(audio_tokens, (1, aud_len), &device)?;
    let aud_position_ids = Tensor::from_vec((0..aud_len as u32).collect::<Vec<_>>(), (1, aud_len), &device)?;

    println!("Running text + audio multimodal forward pass...");
    let audio_emb = model.forward_multimodal(
        &aud_input_ids,
        &aud_position_ids,
        None,
        None,
        None,
        None,
        Some(&input_features),
        None,
    )?;

    assert_eq!(audio_emb.dims2()?, (1, 768));
    let aud_norm = audio_emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    assert!((aud_norm - 1.0).abs() < 1e-4);

    println!("All multimodal forward passes (text, image, video, audio) successfully verified!");
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

#[test]
fn test_multimodal_embedding_sensitivity_real_weights() -> Result<()> {
    let snapshot_dir = match find_snapshot_dir() {
        Some(dir) => dir,
        None => return Ok(()),
    };
    let config_path = snapshot_dir.join("config.json");
    let weights_path = snapshot_dir.join("model.safetensors");
    if !config_path.exists() || !weights_path.exists() {
        return Ok(());
    }

    let device = Device::Cpu;
    let config_file = File::open(&config_path)?;
    let wrapper: EmbeddingGemma2ConfigWrapper = serde_json::from_reader(config_file)?;
    let text_config = wrapper.text_config();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
    };
    let model = EmbeddingGemma2Model::load(vb, text_config, ModelType::Embedding(Pool::Mean))?;

    // 1. Image Sensitivity: Verify dark image vs bright image produce distinct embeddings
    let image_tokens = build_multimodal_tokens(&[2], 1, 0, &[101, 1]);
    let num_tokens = image_tokens.len();
    let mm_input_ids = Tensor::from_vec(image_tokens, (1, num_tokens), &device)?;
    let mm_position_ids = Tensor::from_vec((0..num_tokens as u32).collect::<Vec<_>>(), (1, num_tokens), &device)?;

    let dark_pixels = vec![0.05f32; 48 * 48 * 3];
    let (patches1, positions1, _) = patchify(&dark_pixels, 48, 48, 3, 16, DEFAULT_MAX_PATCHES, 3);
    let (pv1, ppi1) = create_vision_tensors(patches1, positions1, DEFAULT_MAX_PATCHES, 768, &device)?;

    let bright_pixels = vec![0.95f32; 48 * 48 * 3];
    let (patches2, positions2, _) = patchify(&bright_pixels, 48, 48, 3, 16, DEFAULT_MAX_PATCHES, 3);
    let (pv2, ppi2) = create_vision_tensors(patches2, positions2, DEFAULT_MAX_PATCHES, 768, &device)?;

    let emb1 = model.forward_multimodal(
        &mm_input_ids,
        &mm_position_ids,
        Some(&pv1),
        Some(&ppi1),
        None,
        None,
        None,
        None,
    )?;

    let emb2 = model.forward_multimodal(
        &mm_input_ids,
        &mm_position_ids,
        Some(&pv2),
        Some(&ppi2),
        None,
        None,
        None,
        None,
    )?;

    let v1 = emb1.to_vec2::<f32>()?[0].clone();
    let v2 = emb2.to_vec2::<f32>()?[0].clone();
    let sim_img = cosine_similarity(&v1, &v2);
    println!("Cosine similarity between dark and bright images: {sim_img}");
    assert!(sim_img < 0.999, "Different images must yield distinct embeddings (sim: {sim_img})");

    // 2. Audio Sensitivity: Verify silence vs 1kHz tone produce distinct embeddings
    let silence = vec![0.001f32; 16000];
    let tone: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * (i as f32) / 16000.0).sin() * 0.8)
        .collect();

    let (fbank_silence, frames_s) = extract_fbank_features(&silence, 16000, 128, 25.0, 10.0);
    let (fbank_tone, frames_t) = extract_fbank_features(&tone, 16000, 128, 25.0, 10.0);

    let feat_silence = Tensor::from_vec(fbank_silence, (1, frames_s, 128), &device)?;
    let feat_tone = Tensor::from_vec(fbank_tone, (1, frames_t, 128), &device)?;

    let audio_tokens = build_multimodal_tokens(&[2], 0, 10, &[101, 1]);
    let aud_len = audio_tokens.len();
    let aud_input_ids = Tensor::from_vec(audio_tokens, (1, aud_len), &device)?;
    let aud_position_ids = Tensor::from_vec((0..aud_len as u32).collect::<Vec<_>>(), (1, aud_len), &device)?;

    let aud_emb1 = model.forward_multimodal(
        &aud_input_ids,
        &aud_position_ids,
        None,
        None,
        None,
        None,
        Some(&feat_silence),
        None,
    )?;

    let aud_emb2 = model.forward_multimodal(
        &aud_input_ids,
        &aud_position_ids,
        None,
        None,
        None,
        None,
        Some(&feat_tone),
        None,
    )?;

    let av1 = aud_emb1.to_vec2::<f32>()?[0].clone();
    let av2 = aud_emb2.to_vec2::<f32>()?[0].clone();
    let sim_aud = cosine_similarity(&av1, &av2);
    println!("Cosine similarity between silence and 1kHz tone: {sim_aud}");
    assert!(sim_aud < 0.999, "Different audio signals must yield distinct embeddings (sim: {sim_aud})");

    Ok(())
}

#[test]
fn test_all_modalities_simultaneously_fused() -> Result<()> {
    let snapshot_dir = match find_snapshot_dir() {
        Some(dir) => dir,
        None => return Ok(()),
    };
    let config_path = snapshot_dir.join("config.json");
    let weights_path = snapshot_dir.join("model.safetensors");
    if !config_path.exists() || !weights_path.exists() {
        return Ok(());
    }

    let device = Device::Cpu;
    let config_file = File::open(&config_path)?;
    let wrapper: EmbeddingGemma2ConfigWrapper = serde_json::from_reader(config_file)?;
    let text_config = wrapper.text_config();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
    };
    let model = EmbeddingGemma2Model::load(vb, text_config, ModelType::Embedding(Pool::Mean))?;

    // Create prompt with Text + Image (280 tokens) + Audio (10 tokens)
    let prompt_tokens = build_multimodal_tokens(&[2], 1, 10, &[101, 1]);
    let num_tokens = prompt_tokens.len();
    let input_ids = Tensor::from_vec(prompt_tokens, (1, num_tokens), &device)?;
    let position_ids = Tensor::from_vec((0..num_tokens as u32).collect::<Vec<_>>(), (1, num_tokens), &device)?;

    let dummy_pixels = vec![0.3f32; 48 * 48 * 3];
    let (patches, positions, _) = patchify(&dummy_pixels, 48, 48, 3, 16, DEFAULT_MAX_PATCHES, 3);
    let (pixel_values, pixel_position_ids) = create_vision_tensors(
        patches,
        positions,
        DEFAULT_MAX_PATCHES,
        768,
        &device,
    )?;

    let dummy_audio = vec![0.2f32; 16000];
    let (fbank, num_frames) = extract_fbank_features(&dummy_audio, 16000, 128, 25.0, 10.0);
    let input_features = Tensor::from_vec(fbank, (1, num_frames, 128), &device)?;

    println!("Executing omni-modal fusion forward pass (Text + Image + Audio)...");
    let omni_emb = model.forward_multimodal(
        &input_ids,
        &position_ids,
        Some(&pixel_values),
        Some(&pixel_position_ids),
        None,
        None,
        Some(&input_features),
        None,
    )?;

    assert_eq!(omni_emb.dims2()?, (1, 768));
    let norm = omni_emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    assert!((norm - 1.0).abs() < 1e-4, "Omni-modal embedding must be unit normalized: got {norm}");

    let vec = omni_emb.to_vec2::<f32>()?;
    for &val in &vec[0] {
        assert!(!val.is_nan(), "Embedding contains NaN");
        assert!(!val.is_infinite(), "Embedding contains Inf");
    }

    println!("Omni-modal forward pass successfully verified!");
    Ok(())
}

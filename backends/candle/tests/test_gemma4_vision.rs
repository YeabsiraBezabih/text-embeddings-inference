use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use text_embeddings_backend_candle::models::gemma4_vision::*;

fn sample_vision_config() -> Gemma4VisionConfig {
    Gemma4VisionConfig {
        head_dim: 64,
        hidden_size: 768,
        intermediate_size: 3072,
        num_attention_heads: 12,
        num_hidden_layers: 2, // 2 layers for fast testing
        num_key_value_heads: 12,
        patch_size: 16,
        position_embedding_size: 10240,
        pooling_kernel_size: 3,
        rms_norm_eps: 1e-6,
        default_output_length: 280,
        rope_parameters: HashMap::new(),
        standardize: false,
    }
}

#[test]
fn test_vision_rotary_embedding_shapes_and_recomposition() -> Result<()> {
    let device = Device::Cpu;
    let config = sample_vision_config();
    let rope = Gemma4VisionRotaryEmbedding::new(&config, &device)?;

    // 4 patches with coordinates (0, 0), (1, 0), (0, 1), (1, 1)
    let pos_data: Vec<i64> = vec![0, 0, 1, 0, 0, 1, 1, 1];
    let pos_tensor = Tensor::from_vec(pos_data, (1, 4, 2), &device)?;

    let (cos, sin) = rope.forward(&pos_tensor, DType::F32)?;

    // Shape: (B, 1, N, head_dim) = (1, 1, 4, 64)
    assert_eq!(cos.dims4()?, (1, 1, 4, 64));
    assert_eq!(sin.dims4()?, (1, 1, 4, 64));

    // cos^2 + sin^2 should be close to 1.0 for each element
    let sum_sq = (cos.sqr()? + sin.sqr()?)?;
    let ones = Tensor::ones_like(&sum_sq)?;
    let diff = (sum_sq - ones)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-5, "Trig identity cos^2 + sin^2 = 1 violated, diff: {diff}");

    Ok(())
}

#[test]
fn test_vision_spatial_pooler_280_tokens() -> Result<()> {
    let device = Device::Cpu;
    let config = sample_vision_config();
    let pooler = Gemma4VisionPooler::new(&config);

    let b_sz = 1;
    let max_patches = 2520;
    let hidden_size = 768;

    // Fill hidden states with 1.0
    let hidden_states = Tensor::ones((b_sz, max_patches, hidden_size), DType::F32, &device)?;

    // 2520 patches generated on grid
    let mut pos_data = Vec::with_capacity(max_patches * 2);
    for row in 0..60 {
        for col in 0..42 {
            pos_data.push(col as i64);
            pos_data.push(row as i64);
        }
    }
    let pixel_position_ids = Tensor::from_vec(pos_data, (b_sz, max_patches, 2), &device)?;
    let padding_positions = Tensor::zeros((b_sz, max_patches), DType::U8, &device)?;

    let (pooled, mask) = pooler.forward(
        &hidden_states,
        &pixel_position_ids,
        &padding_positions,
        280,
    )?;

    assert_eq!(pooled.dims3()?, (b_sz, 280, hidden_size));
    assert_eq!(mask.dims2()?, (b_sz, 280));

    // Check no NaNs or Infs
    let max_val = pooled.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(!max_val.is_nan());
    assert!(!max_val.is_infinite());
    assert!(max_val > 0.0);

    Ok(())
}

#[test]
fn test_vision_projector_mapping() -> Result<()> {
    let device = Device::Cpu;
    let text_hidden_size = 512;
    let vision_hidden_size = 768;

    // Create random weights in a HashMap for VarBuilder
    let mut weights = HashMap::new();
    let proj_data = vec![0.01f32; text_hidden_size * vision_hidden_size];
    weights.insert(
        "embedding_projection.weight".to_string(),
        Tensor::from_vec(proj_data, (text_hidden_size, vision_hidden_size), &device)?,
    );

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let projector = Gemma4VisionProjector::load(vb, vision_hidden_size, text_hidden_size, 1e-6)?;

    // Input: (1, 280, 768)
    let vision_tokens = Tensor::ones((1, 280, vision_hidden_size), DType::F32, &device)?;
    let projected = projector.forward(&vision_tokens)?;

    // Output must be exactly (1, 280, 512)
    assert_eq!(projected.dims3()?, (1, 280, text_hidden_size));

    let val = projected.i((0, 0, 0))?.to_scalar::<f32>()?;
    assert!(!val.is_nan());
    assert!(!val.is_infinite());

    Ok(())
}

#[test]
fn test_vision_patch_embedder_lookup() -> Result<()> {
    let device = Device::Cpu;
    let config = sample_vision_config();
    let patch_dim = 3 * config.patch_size * config.patch_size; // 768

    let mut weights = HashMap::new();
    weights.insert(
        "input_proj.weight".to_string(),
        Tensor::ones((config.hidden_size, patch_dim), DType::F32, &device)?,
    );
    weights.insert(
        "position_embedding_table".to_string(),
        Tensor::full(0.05f32, (2, config.position_embedding_size, config.hidden_size), &device)?,
    );

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let embedder = Gemma4VisionPatchEmbedder::load(vb, &config)?;

    let num_patches = 9;
    let pixel_values = Tensor::full(0.5f32, (1, num_patches, patch_dim), &device)?;
    let pos_data: Vec<i64> = (0..num_patches).flat_map(|i| vec![i as i64, 0i64]).collect();
    let pixel_pos = Tensor::from_vec(pos_data, (1, num_patches, 2), &device)?;
    let padding = Tensor::zeros((1, num_patches), DType::U8, &device)?;

    let out = embedder.forward(&pixel_values, &pixel_pos, &padding)?;
    assert_eq!(out.dims3()?, (1, num_patches, config.hidden_size));

    // Scaled pixels: 2 * (0.5 - 0.5) = 0.0 -> input_proj gives 0.0
    // Result is purely pos_emb = 0.05 (x) + 0.05 (y) = 0.10
    let sample = out.i((0, 0, 0))?.to_scalar::<f32>()?;
    let diff = (sample - 0.10f32).abs();
    assert!(diff < 1e-4, "Expected pos_emb 0.10, got {sample}");

    Ok(())
}

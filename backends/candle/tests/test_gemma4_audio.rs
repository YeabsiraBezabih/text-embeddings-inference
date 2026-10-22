use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use text_embeddings_backend_candle::models::gemma4_audio::*;

fn sample_audio_config() -> Gemma4AudioConfig {
    Gemma4AudioConfig {
        hidden_size: 1024,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        output_proj_dims: 1536,
        attention_chunk_size: 12,
        attention_context_left: 13,
        attention_context_right: 0,
        attention_logit_cap: 50.0,
        rms_norm_eps: 1e-6,
        residual_weight: 0.5,
        subsampling_conv_channels: vec![128, 32],
        conv_kernel_size: 5,
    }
}

#[test]
fn test_audio_rel_positional_encoding() -> Result<()> {
    let device = Device::Cpu;
    let config = sample_audio_config();
    let rel_enc = Gemma4AudioRelPositionalEncoding::new(&config, &device)?;

    let pos_emb = rel_enc.forward(&device, DType::F32)?;
    // max_pos = context_size / 2 = (12 + 13 - 1) / 2 = 12 -> 13 positions
    let (_, num_pos, hidden_dim) = pos_emb.dims3()?;
    assert_eq!(hidden_dim, config.hidden_size);
    assert_eq!(num_pos, 13);

    // Verify values are bounded in [-1, 1] (sin and cos)
    let max_abs = pos_emb.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_abs <= 1.0 + 1e-5);

    Ok(())
}

#[test]
fn test_audio_projector_mapping() -> Result<()> {
    let device = Device::Cpu;
    let audio_hidden_size = 1536;
    let text_hidden_size = 512;

    let mut weights = HashMap::new();
    let proj_data = vec![0.02f32; text_hidden_size * audio_hidden_size];
    weights.insert(
        "embedding_projection.weight".to_string(),
        Tensor::from_vec(proj_data, (text_hidden_size, audio_hidden_size), &device)?,
    );

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let projector = Gemma4AudioProjector::load(vb, audio_hidden_size, text_hidden_size, 1e-6)?;

    // Input: (1, 50, 1536)
    let audio_tokens = Tensor::ones((1, 50, audio_hidden_size), DType::F32, &device)?;
    let projected = projector.forward(&audio_tokens)?;

    // Output must be exactly (1, 50, 512)
    assert_eq!(projected.dims3()?, (1, 50, text_hidden_size));

    let sample = projected.i((0, 0, 0))?.to_scalar::<f32>()?;
    assert!(!sample.is_nan());
    assert!(!sample.is_infinite());

    Ok(())
}

#[test]
fn test_audio_subsample_layer_shapes() -> Result<()> {
    let device = Device::Cpu;
    let c_in = 1;
    let c_out = 128;

    let mut weights = HashMap::new();
    weights.insert(
        "conv.weight".to_string(),
        Tensor::full(0.01f32, (c_out, c_in, 3, 3), &device)?,
    );
    weights.insert(
        "norm.weight".to_string(),
        Tensor::ones(c_out, DType::F32, &device)?,
    );

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let layer = Gemma4AudioSubSampleLayer::load(vb, c_in, c_out, 1e-6)?;

    // Input: (B, 1, 100 frames, 128 mel bins)
    let input = Tensor::ones((1, 1, 100, 128), DType::F32, &device)?;
    let out = layer.forward(&input)?;

    // Stride 2 downsamples (100 -> 50, 128 -> 64)
    assert_eq!(out.dims4()?, (1, 128, 50, 64));

    Ok(())
}

#[test]
fn test_audio_light_conv1d_forward() -> Result<()> {
    let device = Device::Cpu;
    let config = sample_audio_config();
    let hidden = config.hidden_size;

    let mut weights = HashMap::new();
    weights.insert(
        "linear_start.linear.weight".to_string(),
        Tensor::full(0.01f32, (hidden * 2, hidden), &device)?,
    );
    weights.insert(
        "linear_end.linear.weight".to_string(),
        Tensor::full(0.01f32, (hidden, hidden), &device)?,
    );
    weights.insert(
        "depthwise_conv1d.weight".to_string(),
        Tensor::full(0.2f32, (hidden, 1, 5), &device)?,
    );
    weights.insert("pre_layer_norm.weight".to_string(), Tensor::ones(hidden, DType::F32, &device)?);
    weights.insert("conv_norm.weight".to_string(), Tensor::ones(hidden, DType::F32, &device)?);

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let lconv = Gemma4AudioLightConv1d::load(vb, &config)?;

    // Input: (1, 20, 1024)
    let x = Tensor::ones((1, 20, hidden), DType::F32, &device)?;
    let out = lconv.forward(&x)?;

    assert_eq!(out.dims3()?, (1, 20, hidden));
    let val = out.i((0, 0, 0))?.to_scalar::<f32>()?;
    assert!(!val.is_nan());
    assert!(!val.is_infinite());

    Ok(())
}

use candle::{DType, Device, Result, Tensor};
use text_embeddings_backend_candle::models::embedding_gemma2::scatter_soft_tokens;
use text_embeddings_backend_candle::models::multimodal_preprocessing::{
    AUDIO_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID,
};

#[test]
fn test_scatter_soft_tokens_single_image() -> Result<()> {
    let device = Device::Cpu;
    let b_sz = 1;
    let seq_len = 10;
    let hidden_size = 8;

    // Create input_ids with 3 image placeholders at indices 2, 3, 4
    let ids_vec: Vec<u32> = vec![1, 2, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 5, 6, 7, 8, 9];
    let input_ids = Tensor::from_vec(ids_vec, (b_sz, seq_len), &device)?;

    // Initial embeddings filled with 0.1
    let inputs_embeds = Tensor::full(0.1f32, (b_sz, seq_len, hidden_size), &device)?;

    // 3 soft tokens, filled with distinct values: 10.0, 20.0, 30.0
    let mut soft_vec = Vec::with_capacity(3 * hidden_size);
    for val in [10.0f32, 20.0f32, 30.0f32] {
        for _ in 0..hidden_size {
            soft_vec.push(val);
        }
    }
    let soft_tokens = Tensor::from_vec(soft_vec, (1, 3, hidden_size), &device)?;

    let fused = scatter_soft_tokens(&inputs_embeds, &input_ids, IMAGE_TOKEN_ID, &soft_tokens)?;
    assert_eq!(fused.dims3()?, (b_sz, seq_len, hidden_size));

    let fused_vec = fused.to_vec3::<f32>()?;

    // Non-placeholder indices (0, 1, 5..9) must still equal 0.1
    for s in [0, 1, 5, 6, 7, 8, 9] {
        for d in 0..hidden_size {
            assert_eq!(fused_vec[0][s][d], 0.1f32, "Non-placeholder token at {s} was modified");
        }
    }

    // Placeholder indices (2, 3, 4) must match soft tokens 10.0, 20.0, 30.0
    for d in 0..hidden_size {
        assert_eq!(fused_vec[0][2][d], 10.0f32);
        assert_eq!(fused_vec[0][3][d], 20.0f32);
        assert_eq!(fused_vec[0][4][d], 30.0f32);
    }

    Ok(())
}

#[test]
fn test_scatter_soft_tokens_audio_and_video() -> Result<()> {
    let device = Device::Cpu;
    let b_sz = 1;
    let seq_len = 8;
    let hidden_size = 4;

    // input_ids with 2 audio tokens at 1, 2 and 2 video tokens at 5, 6
    let ids_vec: Vec<u32> = vec![0, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 3, 4, VIDEO_TOKEN_ID, VIDEO_TOKEN_ID, 7];
    let input_ids = Tensor::from_vec(ids_vec, (b_sz, seq_len), &device)?;

    let mut embeds = Tensor::zeros((b_sz, seq_len, hidden_size), DType::F32, &device)?;

    // Audio soft tokens: 7.0
    let audio_soft = Tensor::full(7.0f32, (1, 2, hidden_size), &device)?;
    embeds = scatter_soft_tokens(&embeds, &input_ids, AUDIO_TOKEN_ID, &audio_soft)?;

    // Video soft tokens: 9.0
    let video_soft = Tensor::full(9.0f32, (1, 2, hidden_size), &device)?;
    embeds = scatter_soft_tokens(&embeds, &input_ids, VIDEO_TOKEN_ID, &video_soft)?;

    let res = embeds.to_vec3::<f32>()?;
    for d in 0..hidden_size {
        assert_eq!(res[0][0][d], 0.0);
        assert_eq!(res[0][1][d], 7.0);
        assert_eq!(res[0][2][d], 7.0);
        assert_eq!(res[0][3][d], 0.0);
        assert_eq!(res[0][4][d], 0.0);
        assert_eq!(res[0][5][d], 9.0);
        assert_eq!(res[0][6][d], 9.0);
        assert_eq!(res[0][7][d], 0.0);
    }

    Ok(())
}

#[test]
fn test_scatter_soft_tokens_batch() -> Result<()> {
    let device = Device::Cpu;
    let b_sz = 2;
    let seq_len = 4;
    let hidden_size = 4;

    let ids_vec: Vec<u32> = vec![
        IMAGE_TOKEN_ID, 1, 2, 3,
        0, 1, IMAGE_TOKEN_ID, 3,
    ];
    let input_ids = Tensor::from_vec(ids_vec, (b_sz, seq_len), &device)?;
    let inputs_embeds = Tensor::zeros((b_sz, seq_len, hidden_size), DType::F32, &device)?;

    // 2 soft tokens across batch
    let soft_tokens = Tensor::from_vec(
        vec![
            1.0, 1.0, 1.0, 1.0, // for batch 0 pos 0
            2.0, 2.0, 2.0, 2.0, // for batch 1 pos 2
        ],
        (2, 1, hidden_size),
        &device,
    )?;

    let fused = scatter_soft_tokens(&inputs_embeds, &input_ids, IMAGE_TOKEN_ID, &soft_tokens)?;
    let res = fused.to_vec3::<f32>()?;

    assert_eq!(res[0][0], vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(res[0][1], vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(res[1][2], vec![2.0, 2.0, 2.0, 2.0]);
    assert_eq!(res[1][0], vec![0.0, 0.0, 0.0, 0.0]);

    Ok(())
}

#[test]
fn test_scatter_soft_tokens_no_placeholder_noop() -> Result<()> {
    let device = Device::Cpu;
    let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device)?;
    let inputs_embeds = Tensor::full(5.0f32, (1, 4, 8), &device)?;
    let soft_tokens = Tensor::full(99.0f32, (1, 2, 8), &device)?;

    let fused = scatter_soft_tokens(&inputs_embeds, &input_ids, IMAGE_TOKEN_ID, &soft_tokens)?;
    let res = fused.to_vec3::<f32>()?;

    for s in 0..4 {
        for d in 0..8 {
            assert_eq!(res[0][s][d], 5.0f32);
        }
    }

    Ok(())
}

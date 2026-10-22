use crate::layers::Linear;
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

fn default_head_dim_64() -> usize {
    64
}

fn default_patch_size_16() -> usize {
    16
}

fn default_position_embedding_size_10240() -> usize {
    10240
}

fn default_pooling_kernel_size_3() -> usize {
    3
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_output_length_280() -> usize {
    280
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4VisionConfig {
    #[serde(default = "default_head_dim_64")]
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_patch_size_16")]
    pub patch_size: usize,
    #[serde(default = "default_position_embedding_size_10240")]
    pub position_embedding_size: usize,
    #[serde(default = "default_pooling_kernel_size_3")]
    pub pooling_kernel_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_output_length_280")]
    pub default_output_length: usize,
    #[serde(default)]
    pub rope_parameters: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub standardize: bool,
}

#[derive(Debug)]
pub struct Gemma4VisionRMSNorm {
    weight: Option<Tensor>,
    epsilon: f32,
}

impl Gemma4VisionRMSNorm {
    pub fn load(vb: VarBuilder, dim: usize, eps: f32, with_scale: bool) -> Result<Self> {
        let weight = if with_scale {
            Some(vb.get(dim, "weight").or_else(|_| vb.get(dim, "gamma"))?)
        } else {
            None
        };
        Ok(Self {
            weight,
            epsilon: eps,
        })
    }

    pub fn new(weight: Option<Tensor>, epsilon: f32) -> Self {
        Self { weight, epsilon }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let rsqrt = (variance + self.epsilon as f64)?.sqrt()?.recip()?;
        let norm = x_f32.broadcast_mul(&rsqrt)?.to_dtype(in_dtype)?;
        match &self.weight {
            Some(w) => norm.broadcast_mul(w),
            None => Ok(norm),
        }
    }
}

#[derive(Debug)]
pub struct Gemma4VisionPatchEmbedder {
    input_proj: Linear,
    position_embedding_table: Tensor,
}

impl Gemma4VisionPatchEmbedder {
    pub fn load(vb: VarBuilder, config: &Gemma4VisionConfig) -> Result<Self> {
        let patch_dim = 3 * config.patch_size * config.patch_size;
        let proj_w = vb.get((config.hidden_size, patch_dim), "input_proj.weight")?;
        let input_proj = Linear::new(proj_w, None, None);

        let position_embedding_table = vb.get(
            (2, config.position_embedding_size, config.hidden_size),
            "position_embedding_table",
        )?;

        Ok(Self {
            input_proj,
            position_embedding_table,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
    ) -> Result<Tensor> {
        let in_dtype = pixel_values.dtype();
        // Gemma4 scales raw pixel values: 2 * (pixel_values - 0.5)
        let scaled_pixels = (((pixel_values.to_dtype(DType::F32)? - 0.5)? * 2.0)?
            .to_dtype(in_dtype))?;
        let hidden_states = self.input_proj.forward(&scaled_pixels)?;

        // Position embeddings: sum of x and y coordinate lookups
        // pixel_position_ids: (B, N, 2)
        let clamped_positions = pixel_position_ids.clamp(0i64, 10239i64)?;
        let pos_x = clamped_positions.i((.., .., 0))?.to_dtype(DType::U32)?;
        let pos_y = clamped_positions.i((.., .., 1))?.to_dtype(DType::U32)?;

        let table_x = self.position_embedding_table.i(0)?.contiguous()?; // (10240, hidden_size)
        let table_y = self.position_embedding_table.i(1)?.contiguous()?; // (10240, hidden_size)

        let x_emb = candle_nn::Embedding::new(table_x, self.position_embedding_table.dim(2)?)
            .forward(&pos_x)?;
        let y_emb = candle_nn::Embedding::new(table_y, self.position_embedding_table.dim(2)?)
            .forward(&pos_y)?;

        let mut pos_emb = (x_emb + y_emb)?;

        // Zero out embeddings for padding patches
        let pad_mask = padding_positions.unsqueeze(D::Minus1)?.broadcast_as(pos_emb.shape())?;
        let zeros = Tensor::zeros_like(&pos_emb)?;
        pos_emb = pad_mask.where_cond(&zeros, &pos_emb)?;

        hidden_states + pos_emb
    }
}

#[derive(Debug)]
pub struct Gemma4VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl Gemma4VisionRotaryEmbedding {
    pub fn new(config: &Gemma4VisionConfig, device: &Device) -> Result<Self> {
        let rope_theta = config
            .rope_parameters
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(100.0) as f32;
        let head_dim = config.head_dim;
        let spatial_dim = head_dim / 2;
        let num_freqs = spatial_dim / 2; // 16 for head_dim=64

        let mut freqs = Vec::with_capacity(num_freqs);
        for i in 0..num_freqs {
            let exponent = (2 * i) as f32 / spatial_dim as f32;
            let val = 1.0 / rope_theta.powf(exponent);
            freqs.push(val);
        }

        let inv_freq = Tensor::from_vec(freqs, (1, 1, num_freqs), device)?;
        Ok(Self { inv_freq })
    }

    pub fn forward(
        &self,
        pixel_position_ids: &Tensor,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        // pixel_position_ids: (B, N, 2)
        let clamped = pixel_position_ids.clamp(0i64, 10239i64)?.to_dtype(DType::F32)?;
        let pos_w = clamped.i((.., .., 0))?.unsqueeze(D::Minus1)?; // x (B, N, 1)
        let pos_h = clamped.i((.., .., 1))?.unsqueeze(D::Minus1)?; // y (B, N, 1)

        let freqs_w = pos_w.broadcast_mul(&self.inv_freq)?; // (B, N, 16)
        let freqs_h = pos_h.broadcast_mul(&self.inv_freq)?; // (B, N, 16)

        let cos_w = freqs_w.cos()?;
        let sin_w = freqs_w.sin()?;
        let cos_h = freqs_h.cos()?;
        let sin_h = freqs_h.sin()?;

        // Recomposition: [H, H, W, W] (each 16 dims, total 64 = head_dim)
        let cos = Tensor::cat(&[&cos_h, &cos_h, &cos_w, &cos_w], D::Minus1)?
            .unsqueeze(1)? // (B, 1, N, head_dim)
            .to_dtype(dtype)?;
        let sin = Tensor::cat(&[&sin_h, &sin_h, &sin_w, &sin_w], D::Minus1)?
            .unsqueeze(1)? // (B, 1, N, head_dim)
            .to_dtype(dtype)?;

        Ok((cos, sin))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let half = last_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, &x1], D::Minus1)
}

fn apply_vision_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotated = rotate_half(x)?;
    let x_cos = x.broadcast_mul(cos)?;
    let rot_sin = rotated.broadcast_mul(sin)?;
    x_cos + rot_sin
}

#[derive(Debug)]
pub struct Gemma4VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Gemma4VisionRMSNorm,
    k_norm: Gemma4VisionRMSNorm,
    v_norm: Gemma4VisionRMSNorm,
    num_heads: usize,
    head_dim: usize,
}

impl Gemma4VisionAttention {
    pub fn load(vb: VarBuilder, config: &Gemma4VisionConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let num_heads = config.num_attention_heads;

        let q_proj = Linear::new(
            vb.get((num_heads * head_dim, hidden_size), "q_proj.linear.weight")?,
            None,
            None,
        );
        let k_proj = Linear::new(
            vb.get((num_heads * head_dim, hidden_size), "k_proj.linear.weight")?,
            None,
            None,
        );
        let v_proj = Linear::new(
            vb.get((num_heads * head_dim, hidden_size), "v_proj.linear.weight")?,
            None,
            None,
        );
        let o_proj = Linear::new(
            vb.get((hidden_size, num_heads * head_dim), "o_proj.linear.weight")?,
            None,
            None,
        );

        let q_norm = Gemma4VisionRMSNorm::load(vb.pp("q_norm"), head_dim, config.rms_norm_eps, true)?;
        let k_norm = Gemma4VisionRMSNorm::load(vb.pp("k_norm"), head_dim, config.rms_norm_eps, true)?;
        let v_norm = Gemma4VisionRMSNorm::new(None, config.rms_norm_eps);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (B, H, N, D)
        let k = k
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (B, H, N, D)
        let v = v
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (B, H, N, D)

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;

        let q = apply_vision_rope(&q, cos, sin)?;
        let k = apply_vision_rope(&k, cos, sin)?;

        // Dot product attention with scale = 1.0 (since Q and K have head-norm)
        let mut att = q.matmul(&k.t()?)?;
        if let Some(mask) = attention_mask {
            att = att.broadcast_add(mask)?;
        }

        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?; // (B, H, N, D)
        let out = out.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;
        self.o_proj.forward(&out)
    }
}

#[derive(Debug)]
pub struct Gemma4VisionMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma4VisionMLP {
    pub fn load(vb: VarBuilder, config: &Gemma4VisionConfig) -> Result<Self> {
        let gate_proj = Linear::new(
            vb.get((config.intermediate_size, config.hidden_size), "gate_proj.linear.weight")?,
            None,
            None,
        );
        let up_proj = Linear::new(
            vb.get((config.intermediate_size, config.hidden_size), "up_proj.linear.weight")?,
            None,
            None,
        );
        let down_proj = Linear::new(
            vb.get((config.hidden_size, config.intermediate_size), "down_proj.linear.weight")?,
            None,
            None,
        );

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let act = gate.gelu()?;
        let hidden = (act * up)?;
        self.down_proj.forward(&hidden)
    }
}

#[derive(Debug)]
pub struct Gemma4VisionEncoderLayer {
    self_attn: Gemma4VisionAttention,
    mlp: Gemma4VisionMLP,
    input_layernorm: Gemma4VisionRMSNorm,
    post_attention_layernorm: Gemma4VisionRMSNorm,
    pre_feedforward_layernorm: Gemma4VisionRMSNorm,
    post_feedforward_layernorm: Gemma4VisionRMSNorm,
}

impl Gemma4VisionEncoderLayer {
    pub fn load(vb: VarBuilder, config: &Gemma4VisionConfig) -> Result<Self> {
        let self_attn = Gemma4VisionAttention::load(vb.pp("self_attn"), config)?;
        let mlp = Gemma4VisionMLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = Gemma4VisionRMSNorm::load(vb.pp("input_layernorm"), config.hidden_size, config.rms_norm_eps, true)?;
        let post_attention_layernorm = Gemma4VisionRMSNorm::load(vb.pp("post_attention_layernorm"), config.hidden_size, config.rms_norm_eps, true)?;
        let pre_feedforward_layernorm = Gemma4VisionRMSNorm::load(vb.pp("pre_feedforward_layernorm"), config.hidden_size, config.rms_norm_eps, true)?;
        let post_feedforward_layernorm = Gemma4VisionRMSNorm::load(vb.pp("post_feedforward_layernorm"), config.hidden_size, config.rms_norm_eps, true)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let norm_x = self.input_layernorm.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(&norm_x, cos, sin, attention_mask)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let hidden_states = (residual + attn_out)?;

        let residual = &hidden_states;
        let norm_x = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&norm_x)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        residual + mlp_out
    }
}

#[derive(Debug)]
pub struct Gemma4VisionPooler {
    hidden_size: usize,
    root_hidden_size: f64,
}

impl Gemma4VisionPooler {
    pub fn new(config: &Gemma4VisionConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            root_hidden_size: (config.hidden_size as f64).sqrt(),
        }
    }

    /// Spatial 2D average pooling over 3x3 patch kernels to output_length tokens
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
        output_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, input_seq_len, _) = hidden_states.dims3()?;
        let k = ((input_seq_len / output_length) as f64).sqrt() as usize;
        let k_squared = k * k;

        // Mask out padding positions with zeros
        let pad_mask = padding_positions.unsqueeze(D::Minus1)?.broadcast_as(hidden_states.shape())?;
        let zeros = Tensor::zeros_like(hidden_states)?;
        let masked_states = pad_mask.where_cond(&zeros, hidden_states)?;

        let clamped_positions = pixel_position_ids.clamp(0i64, 10239i64)?;
        let pos_x = clamped_positions.i((.., .., 0))?;
        let pos_y = clamped_positions.i((.., .., 1))?;

        // max_x per batch
        let max_x = (pos_x.max_keepdim(D::Minus1)? + 1.0)?;
        let k_f = k as f64;
        let kx = (pos_x.to_dtype(DType::F64)? / k_f)?.floor()?;
        let ky = (pos_y.to_dtype(DType::F64)? / k_f)?.floor()?;
        let max_x_div_k = (max_x.to_dtype(DType::F64)? / k_f)?.floor()?;

        let kernel_idxs = (kx + (ky.broadcast_mul(&max_x_div_k))?)?.to_dtype(DType::U32)?; // (B, N)

        let mut pooled_batches = Vec::with_capacity(b_sz);
        let mut valid_masks = Vec::with_capacity(b_sz);

        for b in 0..b_sz {
            let b_states = masked_states.i(b)?; // (N, D)
            let b_idxs = kernel_idxs.i(b)?.to_vec1::<u32>()?;
            let b_pad = padding_positions.i(b)?.to_vec1::<u8>()?;

            let mut sums = vec![0.0f32; output_length * self.hidden_size];
            let mut counts = vec![0usize; output_length];

            let b_states_f32 = b_states.to_dtype(DType::F32)?.to_vec2::<f32>()?;

            for (i, (&idx, &pad)) in b_idxs.iter().zip(b_pad.iter()).enumerate() {
                let out_idx = idx as usize;
                if out_idx < output_length && pad == 0 {
                    counts[out_idx] += 1;
                    let row = &b_states_f32[i];
                    let sum_offset = out_idx * self.hidden_size;
                    for d in 0..self.hidden_size {
                        sums[sum_offset + d] += row[d];
                    }
                }
            }

            let mut pooled_vec = vec![0.0f32; output_length * self.hidden_size];
            let mut is_valid = vec![0u8; output_length];

            for out_idx in 0..output_length {
                let cnt = counts[out_idx];
                if cnt > 0 {
                    is_valid[out_idx] = 1;
                    let sum_offset = out_idx * self.hidden_size;
                    for d in 0..self.hidden_size {
                        // Average pooling over the k^2 patch grid, then scaled by sqrt(hidden_size)
                        pooled_vec[sum_offset + d] = (sums[sum_offset + d] / (k_squared as f32)) * (self.root_hidden_size as f32);
                    }
                }
            }

            let b_pooled = Tensor::from_vec(pooled_vec, (output_length, self.hidden_size), hidden_states.device())?
                .to_dtype(hidden_states.dtype())?;
            let b_mask = Tensor::from_vec(is_valid, (output_length,), hidden_states.device())?;

            pooled_batches.push(b_pooled);
            valid_masks.push(b_mask);
        }

        let pooled_tensor = Tensor::stack(&pooled_batches, 0)?;
        let mask_tensor = Tensor::stack(&valid_masks, 0)?;

        Ok((pooled_tensor, mask_tensor))
    }
}

#[derive(Debug)]
pub struct Gemma4VisionModel {
    patch_embedder: Gemma4VisionPatchEmbedder,
    rotary_embedding: Gemma4VisionRotaryEmbedding,
    layers: Vec<Gemma4VisionEncoderLayer>,
    pooler: Gemma4VisionPooler,
    default_output_length: usize,
}

impl Gemma4VisionModel {
    pub fn load(vb: VarBuilder, config: &Gemma4VisionConfig) -> Result<Self> {
        let patch_embedder = Gemma4VisionPatchEmbedder::load(vb.pp("patch_embedder"), config)?;
        let rotary_embedding = Gemma4VisionRotaryEmbedding::new(config, vb.device())?;
        let pooler = Gemma4VisionPooler::new(config);

        let vb_layers = vb.pp("encoder.layers");
        let layers = (0..config.num_hidden_layers)
            .map(|i| Gemma4VisionEncoderLayer::load(vb_layers.pp(i), config))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            patch_embedder,
            rotary_embedding,
            layers,
            pooler,
            default_output_length: config.default_output_length,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, num_patches, _) = pixel_position_ids.dims3()?;
        let pad_positions = pixel_position_ids.i((.., .., 0))?.eq(-1i64)?;

        let mut hidden_states = self.patch_embedder.forward(
            pixel_values,
            pixel_position_ids,
            &pad_positions,
        )?;

        let (cos, sin) = self.rotary_embedding.forward(pixel_position_ids, hidden_states.dtype())?;

        // Encoder attention mask
        let valid_mask = pad_positions.eq(0u8)?;
        let att_mask = {
            let mask = valid_mask.unsqueeze(1)?.unsqueeze(2)?; // (B, 1, 1, N)
            let zeros = Tensor::zeros((b_sz, 1, 1, num_patches), hidden_states.dtype(), hidden_states.device())?;
            let min_val = match hidden_states.dtype() {
                DType::F32 => f32::MIN,
                _ => -65504.0,
            };
            let negatives = Tensor::full(min_val, (b_sz, 1, 1, num_patches), hidden_states.device())?
                .to_dtype(hidden_states.dtype())?;
            mask.where_cond(&zeros, &negatives)?
        };

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                Some(&att_mask),
            )?;
        }

        let (pooled_features, _mask) = self.pooler.forward(
            &hidden_states,
            pixel_position_ids,
            &pad_positions,
            self.default_output_length,
        )?;

        Ok(pooled_features)
    }
}

#[derive(Debug)]
pub struct Gemma4VisionProjector {
    pre_projection_norm: Gemma4VisionRMSNorm,
    embedding_projection: Linear,
}

impl Gemma4VisionProjector {
    pub fn load(vb: VarBuilder, vision_hidden_size: usize, text_hidden_size: usize, eps: f32) -> Result<Self> {
        let pre_projection_norm = Gemma4VisionRMSNorm::new(None, eps);
        let proj_w = vb.get(
            (text_hidden_size, vision_hidden_size),
            "embedding_projection.weight",
        )?;
        let embedding_projection = Linear::new(proj_w, None, None);

        Ok(Self {
            pre_projection_norm,
            embedding_projection,
        })
    }

    pub fn forward(&self, vision_tokens: &Tensor) -> Result<Tensor> {
        let normed = self.pre_projection_norm.forward(vision_tokens)?;
        self.embedding_projection.forward(&normed)
    }
}

use crate::layers::Linear;
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use serde::Deserialize;

fn default_audio_hidden_size() -> usize {
    1024
}

fn default_audio_num_heads() -> usize {
    8
}

fn default_audio_num_layers() -> usize {
    12
}

fn default_audio_output_proj_dims() -> usize {
    1536
}

fn default_audio_chunk_size() -> usize {
    12
}

fn default_audio_context_left() -> usize {
    13
}

fn default_audio_rms_norm_eps() -> f32 {
    1e-6
}

fn default_audio_residual_weight() -> f32 {
    0.5
}

fn default_audio_logit_cap() -> f32 {
    50.0
}

fn default_subsampling_channels() -> Vec<usize> {
    vec![128, 32]
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Gemma4AudioConfig {
    #[serde(default = "default_audio_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_audio_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_audio_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_audio_output_proj_dims")]
    pub output_proj_dims: usize,
    #[serde(default = "default_audio_chunk_size")]
    pub attention_chunk_size: usize,
    #[serde(default = "default_audio_context_left")]
    pub attention_context_left: usize,
    #[serde(default)]
    pub attention_context_right: usize,
    #[serde(default = "default_audio_logit_cap")]
    pub attention_logit_cap: f32,
    #[serde(default = "default_audio_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_audio_residual_weight")]
    pub residual_weight: f32,
    #[serde(default = "default_subsampling_channels")]
    pub subsampling_conv_channels: Vec<usize>,
    #[serde(default)]
    pub conv_kernel_size: usize,
}

#[derive(Debug)]
pub struct Gemma4AudioRMSNorm {
    weight: Option<Tensor>,
    epsilon: f32,
}

impl Gemma4AudioRMSNorm {
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
pub struct Gemma4AudioSubSampleLayer {
    conv: Conv2d,
    norm_weight: Tensor,
    eps: f32,
}

impl Gemma4AudioSubSampleLayer {
    pub fn load(vb: VarBuilder, in_c: usize, out_c: usize, eps: f32) -> Result<Self> {
        let weight = vb.get((out_c, in_c, 3, 3), "conv.weight")?;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
        };
        let conv = Conv2d::new(weight, None, conv_cfg);
        let norm_weight = vb.get(out_c, "norm.weight")?;
        Ok(Self {
            conv,
            norm_weight,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?; // (B, C, H, W)
        // LayerNorm over channel dimension C: permute to (B, H, W, C)
        let h = h.permute((0, 2, 3, 1))?;
        let in_dtype = h.dtype();
        let h_f32 = h.to_dtype(DType::F32)?;
        let mean = h_f32.mean_keepdim(D::Minus1)?;
        let diff = h_f32.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
        let rsqrt = (var + self.eps as f64)?.sqrt()?.recip()?;
        let normed = diff.broadcast_mul(&rsqrt)?.to_dtype(in_dtype)?;
        let normed = normed.broadcast_mul(&self.norm_weight)?;
        let relu = normed.relu()?;
        relu.permute((0, 3, 1, 2))
    }
}

#[derive(Debug)]
pub struct Gemma4AudioSubSampleConvProjection {
    layer0: Gemma4AudioSubSampleLayer,
    layer1: Gemma4AudioSubSampleLayer,
    input_proj_linear: Linear,
}

impl Gemma4AudioSubSampleConvProjection {
    pub fn load(vb: VarBuilder, config: &Gemma4AudioConfig) -> Result<Self> {
        let c0 = config.subsampling_conv_channels[0];
        let c1 = config.subsampling_conv_channels[1];
        let layer0 = Gemma4AudioSubSampleLayer::load(vb.pp("layer0"), 1, c0, config.rms_norm_eps)?;
        let layer1 = Gemma4AudioSubSampleLayer::load(vb.pp("layer1"), c0, c1, config.rms_norm_eps)?;

        // Mel feature bins are 128. After layer0: 128 / 2 = 64. After layer1: 64 / 2 = 32.
        // Flattened channel dimension = 32 * 32 = 1024
        let proj_dim = (c0 / 4) * c1;
        let proj_w = vb.get((config.hidden_size, proj_dim), "input_proj_linear.weight")?;
        let input_proj_linear = Linear::new(proj_w, None, None);

        Ok(Self {
            layer0,
            layer1,
            input_proj_linear,
        })
    }

    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        // input_features: (B, seq_len, 128)
        let (b_sz, _, _) = input_features.dims3()?;
        // Treat as 2D image: (B, 1, seq_len, 128)
        let x = input_features.unsqueeze(1)?;
        let x = self.layer0.forward(&x)?;
        let x = self.layer1.forward(&x)?; // (B, 32, seq_len/4, 32)

        let (_, c, s, w) = x.dims4()?;
        let x = x.permute((0, 2, 1, 3))?.reshape((b_sz, s, c * w))?;
        self.input_proj_linear.forward(&x)
    }
}

#[derive(Debug)]
pub struct Gemma4AudioRelPositionalEncoding {
    inv_timescales: Tensor,
    context_size: usize,
}

impl Gemma4AudioRelPositionalEncoding {
    pub fn new(config: &Gemma4AudioConfig, device: &Device) -> Result<Self> {
        let context_size = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right;
        let num_timescales = config.hidden_size / 2;
        let min_timescale = 1.0f32;
        let max_timescale = 10000.0f32;
        let log_increment = (max_timescale / min_timescale).ln() / (num_timescales.saturating_sub(1).max(1) as f32);

        let mut timescales = Vec::with_capacity(num_timescales);
        for i in 0..num_timescales {
            timescales.push(min_timescale * (-(i as f32) * log_increment).exp());
        }
        let inv_timescales = Tensor::from_vec(timescales, (1, 1, num_timescales), device)?;

        Ok(Self {
            inv_timescales,
            context_size,
        })
    }

    pub fn forward(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        let max_pos = (self.context_size / 2) as i64;
        let mut pos_ids = Vec::with_capacity((max_pos + 1) as usize);
        for p in (0..=max_pos).rev() {
            pos_ids.push(p as f32);
        }
        let pos_tensor = Tensor::from_vec(pos_ids, (1, (max_pos + 1) as usize, 1), device)?;
        let scaled_time = pos_tensor.broadcast_mul(&self.inv_timescales)?;
        let sin = scaled_time.sin()?;
        let cos = scaled_time.cos()?;
        Tensor::cat(&[&sin, &cos], D::Minus1)?.to_dtype(dtype)
    }
}

#[derive(Debug)]
pub struct Gemma4AudioFeedForward {
    ffw1: Linear,
    ffw2: Linear,
    pre_norm: Gemma4AudioRMSNorm,
    post_norm: Gemma4AudioRMSNorm,
    residual_weight: f32,
}

impl Gemma4AudioFeedForward {
    pub fn load(vb: VarBuilder, config: &Gemma4AudioConfig) -> Result<Self> {
        let ffw1_w = vb.get((config.hidden_size * 4, config.hidden_size), "ffw_layer_1.linear.weight")?;
        let ffw2_w = vb.get((config.hidden_size, config.hidden_size * 4), "ffw_layer_2.linear.weight")?;
        let ffw1 = Linear::new(ffw1_w, None, None);
        let ffw2 = Linear::new(ffw2_w, None, None);

        let pre_norm = Gemma4AudioRMSNorm::load(vb.pp("pre_layer_norm"), config.hidden_size, config.rms_norm_eps, true)?;
        let post_norm = Gemma4AudioRMSNorm::load(vb.pp("post_layer_norm"), config.hidden_size, config.rms_norm_eps, true)?;

        Ok(Self {
            ffw1,
            ffw2,
            pre_norm,
            post_norm,
            residual_weight: config.residual_weight,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let normed = self.pre_norm.forward(x)?;
        let h = self.ffw1.forward(&normed)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.ffw2.forward(&h)?;
        let h = self.post_norm.forward(&h)?;
        let h = (h * (self.residual_weight as f64))?;
        residual + h
    }
}

#[derive(Debug)]
pub struct Gemma4AudioLightConv1d {
    linear_start: Linear,
    linear_end: Linear,
    depthwise_w: Tensor, // [1024, 1, 5]
    pre_norm: Gemma4AudioRMSNorm,
    conv_norm: Gemma4AudioRMSNorm,
}

impl Gemma4AudioLightConv1d {
    pub fn load(vb: VarBuilder, config: &Gemma4AudioConfig) -> Result<Self> {
        let linear_start_w = vb.get(
            (config.hidden_size * 2, config.hidden_size),
            "linear_start.linear.weight",
        )?;
        let linear_end_w = vb.get(
            (config.hidden_size, config.hidden_size),
            "linear_end.linear.weight",
        )?;
        let depthwise_w = vb.get(
            (config.hidden_size, 1, 5),
            "depthwise_conv1d.weight",
        )?;

        let linear_start = Linear::new(linear_start_w, None, None);
        let linear_end = Linear::new(linear_end_w, None, None);

        let pre_norm = Gemma4AudioRMSNorm::load(vb.pp("pre_layer_norm"), config.hidden_size, config.rms_norm_eps, true)?;
        let conv_norm = Gemma4AudioRMSNorm::load(vb.pp("conv_norm"), config.hidden_size, config.rms_norm_eps, true)?;

        Ok(Self {
            linear_start,
            linear_end,
            depthwise_w,
            pre_norm,
            conv_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.pre_norm.forward(x)?;
        let projected = self.linear_start.forward(&h)?;

        // GLU activation: act * sigmoid(gate)
        let half = self.depthwise_w.dim(0)?;
        let act = projected.narrow(D::Minus1, 0, half)?;
        let gate = projected.narrow(D::Minus1, half, half)?;
        let glu = (act * candle_nn::ops::sigmoid(&gate)?)?;

        // 1D depthwise causal convolution: kernel=5, left_pad=4
        let glu_t = glu.transpose(1, 2)?; // (B, C, S)
        let (b_sz, c_sz, s_len) = glu_t.dims3()?;

        let left_pad = Tensor::zeros((b_sz, c_sz, 4), glu_t.dtype(), glu_t.device())?;
        let padded = Tensor::cat(&[&left_pad, &glu_t], 2)?; // (B, C, S + 4)

        // Conv1D with groups=C
        let mut conv_out = Vec::with_capacity(s_len);
        let w_f32 = self.depthwise_w.squeeze(1)?.to_dtype(DType::F32)?; // (C, 5)
        let padded_f32 = padded.to_dtype(DType::F32)?;

        for t in 0..s_len {
            let window = padded_f32.narrow(2, t, 5)?; // (B, C, 5)
            let step = window.broadcast_mul(&w_f32.unsqueeze(0)?)?.sum(D::Minus1)?; // (B, C)
            conv_out.push(step.unsqueeze(2)?);
        }

        let conv_res = Tensor::cat(&conv_out, 2)?.to_dtype(x.dtype())?; // (B, C, S)
        let conv_res = conv_res.transpose(1, 2)?; // (B, S, C)

        let normed_conv = self.conv_norm.forward(&conv_res)?;
        let silu_conv = candle_nn::ops::silu(&normed_conv)?;
        let out = self.linear_end.forward(&silu_conv)?;

        residual + out
    }
}

#[derive(Debug)]
pub struct Gemma4AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    post: Linear,
    relative_k_proj: Linear,
    per_dim_scale: Tensor,
    num_heads: usize,
    head_dim: usize,
    q_scale: f64,
    k_scale: f64,
    logit_cap: f32,
}

impl Gemma4AudioAttention {
    pub fn load(vb: VarBuilder, config: &Gemma4AudioConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let head_dim = hidden / heads;

        let q_proj = Linear::new(vb.get((hidden, hidden), "q_proj.linear.weight")?, None, None);
        let k_proj = Linear::new(vb.get((hidden, hidden), "k_proj.linear.weight")?, None, None);
        let v_proj = Linear::new(vb.get((hidden, hidden), "v_proj.linear.weight")?, None, None);
        let post = Linear::new(vb.get((hidden, hidden), "post.linear.weight")?, None, None);

        let rel_k_proj = Linear::new(vb.get((hidden, hidden), "relative_k_proj.weight")?, None, None);
        let per_dim_scale = vb.get(head_dim, "per_dim_scale")?;

        let q_scale = ((head_dim as f64).powf(-0.5)) / 2.0f64.ln();
        let k_scale = (1.0f64 + std::f64::consts::E).ln() / 2.0f64.ln();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            post,
            relative_k_proj: rel_k_proj,
            per_dim_scale,
            num_heads: heads,
            head_dim,
            q_scale,
            k_scale,
            logit_cap: config.attention_logit_cap,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rel_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        // Scale Q with per_dim_scale softplus
        let scale_factor = (self.per_dim_scale.to_dtype(DType::F32)?.exp()? + 1.0)?.log()?;
        let scale_factor = scale_factor.to_dtype(x.dtype())?;
        let q_scaled = (q * self.q_scale)?.broadcast_mul(&scale_factor.unsqueeze(0)?.unsqueeze(1)?.unsqueeze(2)?)?;
        let k_scaled = (k * self.k_scale)?;

        let mut att = q_scaled.matmul(&k_scaled.t()?)?;

        // Relative position contribution
        let rel_k = self.relative_k_proj.forward(rel_pos_emb)?;
        let rel_k = rel_k.reshape(((), self.num_heads, self.head_dim))?.transpose(0, 1)?; // (H, Pos, D)

        let rel_scores = q_scaled.matmul(&rel_k.t()?.unsqueeze(0)?)?; // (B, H, S, Pos)
        if rel_scores.dim(D::Minus1)? >= att.dim(D::Minus1)? {
            let rel_cropped = rel_scores.narrow(D::Minus1, 0, att.dim(D::Minus1)?)?;
            att = (att + rel_cropped)?;
        }

        // Tanh softcap
        let att_f32 = att.to_dtype(DType::F32)?;
        let capped = ((att_f32 / (self.logit_cap as f64))?.tanh()? * (self.logit_cap as f64))?
            .to_dtype(x.dtype())?;

        let att_probs = candle_nn::ops::softmax(&capped, D::Minus1)?;
        let out = att_probs.matmul(&v)?; // (B, H, S, D)

        let out = out.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;
        self.post.forward(&out)
    }
}

#[derive(Debug)]
pub struct Gemma4AudioLayer {
    feed_forward1: Gemma4AudioFeedForward,
    feed_forward2: Gemma4AudioFeedForward,
    self_attn: Gemma4AudioAttention,
    lconv1d: Gemma4AudioLightConv1d,
    norm_pre_attn: Gemma4AudioRMSNorm,
    norm_post_attn: Gemma4AudioRMSNorm,
    norm_out: Gemma4AudioRMSNorm,
}

impl Gemma4AudioLayer {
    pub fn load(vb: VarBuilder, config: &Gemma4AudioConfig) -> Result<Self> {
        let feed_forward1 = Gemma4AudioFeedForward::load(vb.pp("feed_forward1"), config)?;
        let feed_forward2 = Gemma4AudioFeedForward::load(vb.pp("feed_forward2"), config)?;
        let self_attn = Gemma4AudioAttention::load(vb.pp("self_attn"), config)?;
        let lconv1d = Gemma4AudioLightConv1d::load(vb.pp("lconv1d"), config)?;

        let norm_pre_attn = Gemma4AudioRMSNorm::load(vb.pp("norm_pre_attn"), config.hidden_size, config.rms_norm_eps, true)?;
        let norm_post_attn = Gemma4AudioRMSNorm::load(vb.pp("norm_post_attn"), config.hidden_size, config.rms_norm_eps, true)?;
        let norm_out = Gemma4AudioRMSNorm::load(vb.pp("norm_out"), config.hidden_size, config.rms_norm_eps, true)?;

        Ok(Self {
            feed_forward1,
            feed_forward2,
            self_attn,
            lconv1d,
            norm_pre_attn,
            norm_post_attn,
            norm_out,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rel_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let h = self.feed_forward1.forward(x)?;

        let residual = &h;
        let normed = self.norm_pre_attn.forward(&h)?;
        let attn_out = self.self_attn.forward(&normed, rel_pos_emb)?;
        let attn_out = self.norm_post_attn.forward(&attn_out)?;
        let h = (residual + attn_out)?;

        let h = self.lconv1d.forward(&h)?;
        let h = self.feed_forward2.forward(&h)?;
        self.norm_out.forward(&h)
    }
}

#[derive(Debug)]
pub struct Gemma4AudioModel {
    subsample_conv_projection: Gemma4AudioSubSampleConvProjection,
    rel_pos_enc: Gemma4AudioRelPositionalEncoding,
    layers: Vec<Gemma4AudioLayer>,
    output_proj: Linear,
}

impl Gemma4AudioModel {
    pub fn load(vb: VarBuilder, config: &Gemma4AudioConfig) -> Result<Self> {
        let subsample_conv_projection = Gemma4AudioSubSampleConvProjection::load(
            vb.pp("subsample_conv_projection"),
            config,
        )?;
        let rel_pos_enc = Gemma4AudioRelPositionalEncoding::new(config, vb.device())?;

        let vb_layers = vb.pp("layers");
        let layers = (0..config.num_hidden_layers)
            .map(|i| Gemma4AudioLayer::load(vb_layers.pp(i), config))
            .collect::<Result<Vec<_>>>()?;

        let out_w = vb.get((config.output_proj_dims, config.hidden_size), "output_proj.weight")?;
        let out_b = vb.get(config.output_proj_dims, "output_proj.bias")?;
        let output_proj = Linear::new(out_w, Some(out_b), None);

        Ok(Self {
            subsample_conv_projection,
            rel_pos_enc,
            layers,
            output_proj,
        })
    }

    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        let mut h = self.subsample_conv_projection.forward(input_features)?;
        let rel_pos = self.rel_pos_enc.forward(h.device(), h.dtype())?;

        for layer in &self.layers {
            h = layer.forward(&h, &rel_pos)?;
        }

        self.output_proj.forward(&h)
    }
}

#[derive(Debug)]
pub struct Gemma4AudioProjector {
    pre_projection_norm: Gemma4AudioRMSNorm,
    embedding_projection: Linear,
}

impl Gemma4AudioProjector {
    pub fn load(vb: VarBuilder, audio_hidden_size: usize, text_hidden_size: usize, eps: f32) -> Result<Self> {
        let pre_projection_norm = Gemma4AudioRMSNorm::new(None, eps);
        let proj_w = vb.get(
            (text_hidden_size, audio_hidden_size),
            "embedding_projection.weight",
        )?;
        let embedding_projection = Linear::new(proj_w, None, None);

        Ok(Self {
            pre_projection_norm,
            embedding_projection,
        })
    }

    pub fn forward(&self, audio_tokens: &Tensor) -> Result<Tensor> {
        let normed = self.pre_projection_norm.forward(audio_tokens)?;
        self.embedding_projection.forward(&normed)
    }
}

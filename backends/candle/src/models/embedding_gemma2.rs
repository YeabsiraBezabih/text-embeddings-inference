use crate::layers::{apply_rotary, get_cos_sin, get_inv_freqs, HiddenAct, Linear};
use crate::models::Model;
use super::gemma4_audio::{Gemma4AudioConfig, Gemma4AudioModel, Gemma4AudioProjector};
use super::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionModel, Gemma4VisionProjector};

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2LayerOverride {
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParam {
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: String,
}

fn default_hidden_act() -> HiddenAct {
    HiddenAct::Gelu
}

fn default_hidden_size_per_layer_input() -> usize {
    512
}

fn default_head_dim() -> usize {
    256
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2Config {
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub per_layer_config: HashMap<String, EmbeddingGemma2LayerOverride>,
    #[serde(default)]
    pub rope_parameters: HashMap<String, RopeParam>,
    pub sliding_window: Option<usize>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    #[serde(default = "default_hidden_size_per_layer_input")]
    pub hidden_size_per_layer_input: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingGemma2ConfigWrapper {
    Outer {
        text_config: EmbeddingGemma2Config,
        #[serde(default)]
        vision_config: Option<Gemma4VisionConfig>,
        #[serde(default)]
        audio_config: Option<Gemma4AudioConfig>,
        #[serde(default)]
        image_token_id: Option<u32>,
        #[serde(default)]
        video_token_id: Option<u32>,
        #[serde(default)]
        audio_token_id: Option<u32>,
    },
    Inner(EmbeddingGemma2Config),
}

impl EmbeddingGemma2ConfigWrapper {
    pub fn into_inner(self) -> EmbeddingGemma2Config {
        match self {
            Self::Outer { text_config, .. } => text_config,
            Self::Inner(config) => config,
        }
    }

    pub fn text_config(&self) -> &EmbeddingGemma2Config {
        match self {
            Self::Outer { text_config, .. } => text_config,
            Self::Inner(config) => config,
        }
    }

    pub fn vision_config(&self) -> Option<&Gemma4VisionConfig> {
        match self {
            Self::Outer { vision_config, .. } => vision_config.as_ref(),
            Self::Inner(_) => None,
        }
    }

    pub fn audio_config(&self) -> Option<&Gemma4AudioConfig> {
        match self {
            Self::Outer { audio_config, .. } => audio_config.as_ref(),
            Self::Inner(_) => None,
        }
    }

    pub fn image_token_id(&self) -> u32 {
        match self {
            Self::Outer { image_token_id, .. } => image_token_id.unwrap_or(258880),
            Self::Inner(_) => 258880,
        }
    }

    pub fn video_token_id(&self) -> u32 {
        match self {
            Self::Outer { video_token_id, .. } => video_token_id.unwrap_or(258884),
            Self::Inner(_) => 258884,
        }
    }

    pub fn audio_token_id(&self) -> u32 {
        match self {
            Self::Outer { audio_token_id, .. } => audio_token_id.unwrap_or(258881),
            Self::Inner(_) => 258881,
        }
    }
}

#[derive(Debug)]
pub struct EmbeddingGemma2RMSNorm {
    weight: Option<Tensor>,
    epsilon: f32,
    span: tracing::Span,
}

impl EmbeddingGemma2RMSNorm {
    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32, with_scale: bool) -> Result<Self> {
        let weight = if with_scale {
            Some(
                vb.get(hidden_size, "weight")
                    .or_else(|_| vb.get(hidden_size, "gamma"))?,
            )
        } else {
            None
        };
        Ok(Self {
            weight,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let hidden_states_dtype = hidden_states.dtype();
        let internal_dtype = match hidden_states_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let hidden_size = hidden_states.dim(D::Minus1)?;
        let hidden_states_f32 = hidden_states.to_dtype(internal_dtype)?;
        let norm_hidden_states =
            (hidden_states_f32.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let hidden_states_normed = hidden_states_f32
            .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?;

        let output = match &self.weight {
            Some(w) => hidden_states_normed.broadcast_mul(&w.to_dtype(internal_dtype)?)?,
            None => hidden_states_normed,
        };

        output.to_dtype(hidden_states_dtype)
    }
}

pub struct EmbeddingGemma2Embedding {
    embedding: Embedding,
    scale: f64,
    span: tracing::Span,
}

impl EmbeddingGemma2Embedding {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2Config) -> Result<Self> {
        let embedding = Embedding::new(
            vb.get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );
        let scale = (config.hidden_size as f64).sqrt();

        Ok(Self {
            embedding,
            scale,
            span: tracing::span!(tracing::Level::TRACE, "embed_tokens"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.embedding.forward(input_ids)?;
        hidden_states * self.scale
    }
}

pub struct EmbeddingGemma2TextPLE {
    per_layer_model_projection: Linear,
    scale: f64,
    per_layer_projection_norm: EmbeddingGemma2RMSNorm,
    num_hidden_layers: usize,
    hidden_size_per_layer_input: usize,
    span: tracing::Span,
}

impl EmbeddingGemma2TextPLE {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2Config) -> Result<Self> {
        let weight = vb.get(
            (
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                config.hidden_size,
            ),
            "per_layer_model_projection.weight",
        )?;
        let per_layer_model_projection = Linear::new(weight, None, None);
        let scale = (config.hidden_size as f64).powf(-0.5);
        let per_layer_projection_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("per_layer_projection_norm"),
            config.hidden_size_per_layer_input,
            config.rms_norm_eps,
            true,
        )?;

        Ok(Self {
            per_layer_model_projection,
            scale,
            per_layer_projection_norm,
            num_hidden_layers: config.num_hidden_layers,
            hidden_size_per_layer_input: config.hidden_size_per_layer_input,
            span: tracing::span!(tracing::Level::TRACE, "ple"),
        })
    }

    pub fn forward(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let proj = self.per_layer_model_projection.forward(inputs_embeds)?;
        let proj = (proj * self.scale)?;
        let dims = inputs_embeds.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let reshaped = proj.reshape((
            batch_size,
            seq_len,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        ))?;
        self.per_layer_projection_norm.forward(&reshaped)
    }
}

pub struct EmbeddingGemma2TextPLEBlock {
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: EmbeddingGemma2RMSNorm,
    hidden_activation: HiddenAct,
    span: tracing::Span,
}

impl EmbeddingGemma2TextPLEBlock {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2Config) -> Result<Self> {
        let gate_w = vb.get(
            (config.hidden_size_per_layer_input, config.hidden_size),
            "per_layer_input_gate.weight",
        )?;
        let per_layer_input_gate = Linear::new(gate_w, None, None);

        let proj_w = vb.get(
            (config.hidden_size, config.hidden_size_per_layer_input),
            "per_layer_projection.weight",
        )?;
        let per_layer_projection = Linear::new(proj_w, None, None);

        let post_per_layer_input_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("post_per_layer_input_norm"),
            config.hidden_size,
            config.rms_norm_eps,
            true,
        )?;

        Ok(Self {
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            hidden_activation: config.hidden_activation.clone(),
            span: tracing::span!(tracing::Level::TRACE, "ple_block"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = hidden_states.clone();
        let gate = self.per_layer_input_gate.forward(hidden_states)?;
        let gate = self.hidden_activation.forward(&gate)?;
        let gated = (gate * per_layer_input)?;
        let proj = self.per_layer_projection.forward(&gated)?;
        let normed = self.post_per_layer_input_norm.forward(&proj)?;
        residual.broadcast_add(&normed)
    }
}

pub struct EmbeddingGemma2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    q_norm: EmbeddingGemma2RMSNorm,
    k_norm: EmbeddingGemma2RMSNorm,
    v_norm: EmbeddingGemma2RMSNorm,

    attention_head_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2Attention {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2Config,
        layer_idx: usize,
        _is_sliding: bool,
    ) -> Result<Self> {
        let layer_key = format!("{layer_idx:02}");
        let (attention_head_size, num_key_value_heads) = if let Some(override_cfg) = config
            .per_layer_config
            .get(&layer_key)
        {
            (
                override_cfg.head_dim.unwrap_or(config.head_dim),
                override_cfg
                    .num_key_value_heads
                    .unwrap_or(config.num_key_value_heads),
            )
        } else if let Some(override_cfg) = config.per_layer_config.get(&layer_idx.to_string()) {
            (
                override_cfg.head_dim.unwrap_or(config.head_dim),
                override_cfg
                    .num_key_value_heads
                    .unwrap_or(config.num_key_value_heads),
            )
        } else {
            (config.head_dim, config.num_key_value_heads)
        };

        let num_attention_heads = config.num_attention_heads;
        let hidden_size = config.hidden_size;

        let q_w = vb.get(
            (num_attention_heads * attention_head_size, hidden_size),
            "q_proj.weight",
        )?;
        let k_w = vb.get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "k_proj.weight",
        )?;
        let v_w = vb.get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "v_proj.weight",
        )?;
        let o_w = vb.get(
            (hidden_size, num_attention_heads * attention_head_size),
            "o_proj.weight",
        )?;

        let q_proj = Linear::new(q_w, None, None);
        let k_proj = Linear::new(k_w, None, None);
        let v_proj = Linear::new(v_w, None, None);
        let o_proj = Linear::new(o_w, None, None);

        let q_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("q_norm"),
            attention_head_size,
            config.rms_norm_eps,
            true,
        )?;
        let k_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("k_norm"),
            attention_head_size,
            config.rms_norm_eps,
            true,
        )?;
        let v_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("v_norm"),
            attention_head_size,
            config.rms_norm_eps,
            false,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            attention_head_size,
            num_attention_heads,
            num_key_value_heads,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_dims = hidden_states.dims();
        let batch_size = input_dims[0];
        let seq_len = input_dims[1];

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let k = k.reshape((
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.attention_head_size,
        ))?;
        let v = v.reshape((
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.attention_head_size,
        ))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = apply_rotary(&q, cos, sin, self.attention_head_size)?;
        let k = apply_rotary(&k, cos, sin, self.attention_head_size)?;

        // Repeat KV for GQA if needed
        let k = if self.num_key_value_heads != self.num_attention_heads {
            let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
            let (b, h, s, d) = k.shape().dims4()?;
            k.unsqueeze(2)?
                .expand((b, h, repeat_factor, s, d))?
                .reshape((b, h * repeat_factor, s, d))?
        } else {
            k
        };

        let v = if self.num_key_value_heads != self.num_attention_heads {
            let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
            let (b, h, s, d) = v.shape().dims4()?;
            v.unsqueeze(2)?
                .expand((b, h, repeat_factor, s, d))?
                .reshape((b, h * repeat_factor, s, d))?
        } else {
            v
        };

        let mut attn_weights = q.matmul(&k.t()?)?;

        if let Some(bias) = attention_bias {
            attn_weights = attn_weights.broadcast_add(bias)?;
        }

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let context = attn_probs.matmul(&v.contiguous()?)?;

        let context = context.transpose(1, 2)?.flatten_from(D::Minus2)?;
        self.o_proj.forward(&context)
    }
}

pub struct EmbeddingGemma2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    hidden_activation: HiddenAct,
    span: tracing::Span,
}

impl EmbeddingGemma2MLP {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2Config) -> Result<Self> {
        let gate_w = vb.get(
            (config.intermediate_size, config.hidden_size),
            "gate_proj.weight",
        )?;
        let up_w = vb.get(
            (config.intermediate_size, config.hidden_size),
            "up_proj.weight",
        )?;
        let down_w = vb.get(
            (config.hidden_size, config.intermediate_size),
            "down_proj.weight",
        )?;

        Ok(Self {
            gate_proj: Linear::new(gate_w, None, None),
            up_proj: Linear::new(up_w, None, None),
            down_proj: Linear::new(down_w, None, None),
            hidden_activation: config.hidden_activation.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(hidden_states)?;
        let gate = self.hidden_activation.forward(&gate)?;
        let up = self.up_proj.forward(hidden_states)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

pub struct EmbeddingGemma2Layer {
    input_layernorm: EmbeddingGemma2RMSNorm,
    self_attn: EmbeddingGemma2Attention,
    post_attention_layernorm: EmbeddingGemma2RMSNorm,
    pre_feedforward_layernorm: EmbeddingGemma2RMSNorm,
    mlp: EmbeddingGemma2MLP,
    post_feedforward_layernorm: EmbeddingGemma2RMSNorm,
    ple_block: EmbeddingGemma2TextPLEBlock,
    layer_scalar: Tensor,
    is_sliding: bool,
    span: tracing::Span,
}

impl EmbeddingGemma2Layer {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2Config,
        layer_idx: usize,
    ) -> Result<Self> {
        let is_sliding = config
            .layer_types
            .get(layer_idx)
            .map(|s| s.as_str())
            != Some("full_attention");

        let input_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
            true,
        )?;
        let self_attn = EmbeddingGemma2Attention::load(
            vb.pp("self_attn"),
            config,
            layer_idx,
            is_sliding,
        )?;
        let post_attention_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
            true,
        )?;
        let pre_feedforward_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("pre_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
            true,
        )?;
        let mlp = EmbeddingGemma2MLP::load(vb.pp("mlp"), config)?;
        let post_feedforward_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("post_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
            true,
        )?;
        let ple_block = EmbeddingGemma2TextPLEBlock::load(vb.pp("ple_block"), config)?;
        let layer_scalar = vb.get(1, "layer_scalar")?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            mlp,
            post_feedforward_layernorm,
            ple_block,
            layer_scalar,
            is_sliding,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        per_layer_input: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(&normed, attention_bias, cos, sin)?;
        let post_attn = self.post_attention_layernorm.forward(&attn_out)?;
        let hidden_states = residual.broadcast_add(&post_attn)?;

        let residual = hidden_states.clone();
        let pre_ffn = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_out = self.mlp.forward(&pre_ffn)?;
        let post_ffn = self.post_feedforward_layernorm.forward(&mlp_out)?;
        let hidden_states = residual.broadcast_add(&post_ffn)?;

        let hidden_states = self.ple_block.forward(&hidden_states, per_layer_input)?;
        hidden_states.broadcast_mul(&self.layer_scalar)
    }
}

pub fn scatter_soft_tokens(
    inputs_embeds: &Tensor,
    input_ids: &Tensor,
    target_token_id: u32,
    soft_tokens: &Tensor,
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_size) = inputs_embeds.dims3()?;
    let mut embeds_vec = inputs_embeds.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let ids_vec = input_ids.to_vec2::<u32>()?;
    let soft_flat = soft_tokens
        .to_dtype(DType::F32)?
        .reshape(((), hidden_size))?
        .to_vec2::<f32>()?;

    let mut soft_idx = 0;
    for b in 0..b_sz {
        for s in 0..seq_len {
            if ids_vec[b][s] == target_token_id && soft_idx < soft_flat.len() {
                embeds_vec[b][s].copy_from_slice(&soft_flat[soft_idx]);
                soft_idx += 1;
            }
        }
    }

    let flat_data: Vec<f32> = embeds_vec.into_iter().flatten().flatten().collect();
    Tensor::from_vec(flat_data, (b_sz, seq_len, hidden_size), inputs_embeds.device())?
        .to_dtype(inputs_embeds.dtype())
}

pub struct EmbeddingGemma2Model {
    embed_tokens: EmbeddingGemma2Embedding,
    ple: EmbeddingGemma2TextPLE,
    layers: Vec<EmbeddingGemma2Layer>,
    norm: EmbeddingGemma2RMSNorm,
    embedding_projection: Linear,

    rotary_cache_sliding: (Tensor, Tensor),
    rotary_cache_full: (Tensor, Tensor),
    head_dim_sliding: usize,
    head_dim_full: usize,

    pad_token_id: u32,
    sliding_window: Option<usize>,
    pool: Pool,

    pub vision_tower: Option<Gemma4VisionModel>,
    pub embed_vision: Option<Gemma4VisionProjector>,
    pub audio_tower: Option<Gemma4AudioModel>,
    pub embed_audio: Option<Gemma4AudioProjector>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub audio_token_id: u32,

    dtype: DType,
    device: Device,
    span: tracing::Span,
}

impl EmbeddingGemma2Model {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2Config, model_type: ModelType) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for EmbeddingGemma2")
            }
            ModelType::Embedding(pool) => pool,
        };

        // Official checkpoints prefix language tensors with "language_model."
        let vb_lm = if vb.contains_tensor("language_model.norm.weight") {
            vb.pp("language_model")
        } else {
            vb.clone()
        };

        let embed_tokens = EmbeddingGemma2Embedding::load(vb_lm.pp("embed_tokens"), config)?;
        let ple = EmbeddingGemma2TextPLE::load(vb_lm.pp("ple"), config)?;

        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| {
                EmbeddingGemma2Layer::load(
                    vb_lm.pp(format!("layers.{layer_idx}")),
                    config,
                    layer_idx,
                )
            })
            .collect::<Result<Vec<EmbeddingGemma2Layer>>>()?;

        let norm = EmbeddingGemma2RMSNorm::load(
            vb_lm.pp("norm"),
            config.hidden_size,
            config.rms_norm_eps,
            true,
        )?;
        let proj_w = vb_lm.get(
            (config.embedding_dim, config.hidden_size),
            "embedding_projection.weight",
        )?;
        let embedding_projection = Linear::new(proj_w, None, None);

        let head_dim_sliding = config.head_dim;
        let head_dim_full = config
            .per_layer_config
            .values()
            .find_map(|c| c.head_dim)
            .unwrap_or(config.head_dim * 2);

        let rope_theta_sliding = config
            .rope_parameters
            .get("sliding_attention")
            .map(|r| r.rope_theta)
            .unwrap_or(10000.0);
        let rope_theta_full = config
            .rope_parameters
            .get("full_attention")
            .map(|r| r.rope_theta)
            .unwrap_or(1000000.0);

        let inv_freqs_sliding =
            get_inv_freqs(head_dim_sliding, rope_theta_sliding, vb_lm.device(), None)?;
        let rotary_cache_sliding =
            get_cos_sin(config.max_position_embeddings, &inv_freqs_sliding, vb_lm.dtype(), true)?;

        let inv_freqs_full =
            get_inv_freqs(head_dim_full, rope_theta_full, vb_lm.device(), None)?;
        let rotary_cache_full =
            get_cos_sin(config.max_position_embeddings, &inv_freqs_full, vb_lm.dtype(), true)?;

        let vision_tower = if vb.contains_tensor("vision_tower.encoder.layers.0.input_layernorm.weight") {
            let v_cfg = Gemma4VisionConfig {
                head_dim: 64,
                hidden_size: 768,
                intermediate_size: 3072,
                num_attention_heads: 12,
                num_hidden_layers: 16,
                num_key_value_heads: 12,
                patch_size: 16,
                position_embedding_size: 10240,
                pooling_kernel_size: 3,
                rms_norm_eps: 1e-6,
                default_output_length: 280,
                rope_parameters: HashMap::new(),
                standardize: false,
            };
            Some(Gemma4VisionModel::load(vb.pp("vision_tower"), &v_cfg)?)
        } else {
            None
        };

        let embed_vision = if vb.contains_tensor("embed_vision.embedding_projection.weight") {
            Some(Gemma4VisionProjector::load(
                vb.pp("embed_vision"),
                768,
                config.hidden_size,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };

        let disable_audio = std::env::var("TEI_DISABLE_AUDIO")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let audio_tower = if !disable_audio && vb.contains_tensor("audio_tower.output_proj.weight") {
            tracing::info!("Loading audio encoder for EmbeddingGemma2");
            let a_cfg = Gemma4AudioConfig {
                hidden_size: 1024,
                num_hidden_layers: 12,
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
            };
            Some(Gemma4AudioModel::load(vb.pp("audio_tower"), &a_cfg)?)
        } else {
            if disable_audio {
                tracing::info!("Audio encoder pruned (TEI_DISABLE_AUDIO=1) to conserve RAM");
            }
            None
        };

        let embed_audio = if !disable_audio && vb.contains_tensor("embed_audio.embedding_projection.weight") {
            Some(Gemma4AudioProjector::load(
                vb.pp("embed_audio"),
                1536,
                config.hidden_size,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            ple,
            layers,
            norm,
            embedding_projection,
            rotary_cache_sliding,
            rotary_cache_full,
            head_dim_sliding,
            head_dim_full,
            pool,
            pad_token_id: config.pad_token_id,
            sliding_window: config.sliding_window,
            vision_tower,
            embed_vision,
            audio_tower,
            embed_audio,
            image_token_id: 258880,
            video_token_id: 258884,
            audio_token_id: 258881,
            dtype: vb_lm.dtype(),
            device: vb_lm.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        pixel_position_ids: Option<&Tensor>,
        pixel_values_videos: Option<&Tensor>,
        video_position_ids: Option<&Tensor>,
        input_features: Option<&Tensor>,
        attention_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, max_length) = input_ids.dims2()?;
        let mut inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // Early fusion: Image
        if let (Some(pv), Some(ppi), Some(vt), Some(ev)) = (
            pixel_values,
            pixel_position_ids,
            &self.vision_tower,
            &self.embed_vision,
        ) {
            let vision_tokens = vt.forward(pv, ppi)?;
            let projected = ev.forward(&vision_tokens)?;
            inputs_embeds = scatter_soft_tokens(
                &inputs_embeds,
                input_ids,
                self.image_token_id,
                &projected,
            )?;
        }

        // Early fusion: Video
        if let (Some(pvv), Some(vpi), Some(vt), Some(ev)) = (
            pixel_values_videos,
            video_position_ids,
            &self.vision_tower,
            &self.embed_vision,
        ) {
            let video_tokens = vt.forward(pvv, vpi)?;
            let projected = ev.forward(&video_tokens)?;
            inputs_embeds = scatter_soft_tokens(
                &inputs_embeds,
                input_ids,
                self.video_token_id,
                &projected,
            )?;
        }

        // Early fusion: Audio
        if let (Some(af), Some(at), Some(ea)) = (
            input_features,
            &self.audio_tower,
            &self.embed_audio,
        ) {
            let audio_tokens = at.forward(af)?;
            let projected = ea.forward(&audio_tokens)?;
            inputs_embeds = scatter_soft_tokens(
                &inputs_embeds,
                input_ids,
                self.audio_token_id,
                &projected,
            )?;
        }

        let per_layer_inputs = self.ple.forward(&inputs_embeds)?;

        let cos_s = self
            .rotary_cache_sliding
            .0
            .index_select(&position_ids.flatten_all()?, 0)?;
        let cos_s = cos_s.reshape((batch_size, 1, max_length, self.head_dim_sliding))?;
        let sin_s = self
            .rotary_cache_sliding
            .1
            .index_select(&position_ids.flatten_all()?, 0)?;
        let sin_s = sin_s.reshape((batch_size, 1, max_length, self.head_dim_sliding))?;

        let cos_f = self
            .rotary_cache_full
            .0
            .index_select(&position_ids.flatten_all()?, 0)?;
        let cos_f = cos_f.reshape((batch_size, 1, max_length, self.head_dim_full))?;
        let sin_f = self
            .rotary_cache_full
            .1
            .index_select(&position_ids.flatten_all()?, 0)?;
        let sin_f = sin_f.reshape((batch_size, 1, max_length, self.head_dim_full))?;

        let mut hidden_states = inputs_embeds;

        for (i, layer) in self.layers.iter().enumerate() {
            let per_layer_input = per_layer_inputs.i((.., .., i, ..))?;
            let (cos, sin) = if layer.is_sliding {
                (&cos_s, &sin_s)
            } else {
                (&cos_f, &sin_f)
            };
            hidden_states = layer.forward(
                &hidden_states,
                &per_layer_input,
                attention_bias,
                cos,
                sin,
            )?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let outputs = self.embedding_projection.forward(&hidden_states)?;

        let pooled = outputs.mean(1)?;
        let norm = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        pooled.broadcast_div(&norm)
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let shape = (batch_size, max_length);

        let (input_ids, position_ids, input_lengths, attention_bias) = if batch_size > 1 {
            let elems = batch_size * max_length;

            let mut input_ids = Vec::with_capacity(elems);
            let mut position_ids = Vec::with_capacity(elems);
            let mut attention_bias = Vec::with_capacity(elems);
            let mut input_lengths = Vec::with_capacity(batch_size);
            let mut masking = false;

            for i in 0..batch_size {
                let start = batch.cumulative_seq_lengths[i] as usize;
                let end = batch.cumulative_seq_lengths[i + 1] as usize;
                let seq_length = end - start;
                input_lengths.push(seq_length);

                for j in start..end {
                    input_ids.push(batch.input_ids[j]);
                    position_ids.push(batch.position_ids[j]);
                    attention_bias.push(0.0);
                }

                let padding = max_length - seq_length;
                if padding > 0 {
                    masking = true;
                    for _ in 0..padding {
                        input_ids.push(self.pad_token_id);
                        position_ids.push(0);
                        attention_bias.push(f32::NEG_INFINITY);
                    }
                }
            }

            let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
            let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;

            let attention_bias = if masking {
                let attention_bias =
                    Tensor::from_vec(attention_bias, (batch_size, 1, 1, max_length), &self.device)?
                        .to_dtype(self.dtype)?;
                Some(attention_bias)
            } else {
                None
            };

            (input_ids, position_ids, input_lengths, attention_bias)
        } else {
            let input_ids = Tensor::from_vec(
                batch.input_ids.clone(),
                (1, batch.input_ids.len()),
                &self.device,
            )?;
            let position_ids = Tensor::from_vec(
                batch.position_ids.clone(),
                (1, batch.position_ids.len()),
                &self.device,
            )?;
            let input_lengths = vec![batch.input_ids.len()];

            (input_ids, position_ids, input_lengths, None)
        };

        let attention_bias_full = attention_bias;
        let attention_bias_sliding = match (&attention_bias_full, self.sliding_window) {
            (_, Some(window_size)) if max_length > window_size => {
                let mask: Vec<u8> = (0..max_length)
                    .flat_map(|i| {
                        (0..max_length).map(move |j| (i.abs_diff(j) <= window_size) as u8)
                    })
                    .collect();
                let mask_tensor =
                    Tensor::from_slice(&mask, (1, 1, max_length, max_length), &self.device)?;
                let zeros = Tensor::zeros(
                    (1, 1, max_length, max_length),
                    self.dtype,
                    &self.device,
                )?;
                let min_val = match self.dtype {
                    DType::F32 => f32::MIN,
                    _ => -65504.0,
                };
                let negatives = Tensor::full(
                    min_val,
                    (1, 1, max_length, max_length),
                    &self.device,
                )?
                .to_dtype(self.dtype)?;
                let window_bias = mask_tensor.where_cond(&zeros, &negatives)?;
                match &attention_bias_full {
                    Some(bias) => Some(bias.broadcast_add(&window_bias)?),
                    None => Some(window_bias),
                }
            }
            (bias, _) => bias.clone(),
        };

        let inputs_embeds = self.embed_tokens.forward(&input_ids)?;
        let per_layer_inputs = self.ple.forward(&inputs_embeds)?;

        let cos_s = self
            .rotary_cache_sliding
            .0
            .index_select(&position_ids.flatten_all()?, 0)?;
        let cos_s = cos_s.reshape((batch_size, 1, max_length, self.head_dim_sliding))?;
        let sin_s = self
            .rotary_cache_sliding
            .1
            .index_select(&position_ids.flatten_all()?, 0)?;
        let sin_s = sin_s.reshape((batch_size, 1, max_length, self.head_dim_sliding))?;

        let cos_f = self
            .rotary_cache_full
            .0
            .index_select(&position_ids.flatten_all()?, 0)?;
        let cos_f = cos_f.reshape((batch_size, 1, max_length, self.head_dim_full))?;
        let sin_f = self
            .rotary_cache_full
            .1
            .index_select(&position_ids.flatten_all()?, 0)?;
        let sin_f = sin_f.reshape((batch_size, 1, max_length, self.head_dim_full))?;

        let mut hidden_states = inputs_embeds;

        for (i, layer) in self.layers.iter().enumerate() {
            let per_layer_input = per_layer_inputs.i((.., .., i, ..))?;
            let (cos, sin, bias) = if layer.is_sliding {
                (&cos_s, &sin_s, attention_bias_sliding.as_ref())
            } else {
                (&cos_f, &sin_f, attention_bias_full.as_ref())
            };
            hidden_states = layer.forward(
                &hidden_states,
                &per_layer_input,
                bias,
                cos,
                sin,
            )?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let outputs = self.embedding_projection.forward(&hidden_states)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                Pool::Cls | Pool::Splade => {
                    candle::bail!("Only Mean and Last-Token pooling are supported for EmbeddingGemma2, no CLS or SPLADE");
                }
                Pool::Mean => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .iter()
                            .map(|&i| {
                                let i = i as usize;
                                let length = input_lengths[i];
                                let embeddings = outputs.i((i, ..length))?;
                                embeddings.sum_keepdim(0)? / (length as f64)
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        let length = input_lengths[0];
                        let embeddings = outputs.i((0, ..length))?;
                        Some((embeddings.sum_keepdim(0)? / (length as f64))?)
                    }
                }
                Pool::LastToken => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .iter()
                            .map(|&i| {
                                let i = i as usize;
                                outputs.i((i..i + 1, input_lengths[i] - 1))
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some(outputs.i((.., max_length - 1))?)
                    }
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                let mut final_embeddings = Vec::new();
                for &i in &batch.raw_indices {
                    let i = i as usize;
                    let length = input_lengths[i];
                    final_embeddings.push(outputs.i((i, ..length))?);
                }
                Some(Tensor::cat(&final_embeddings, 0)?)
            } else if batch_size == 1 {
                let length = input_lengths[0];
                Some(outputs.i((0, ..length))?)
            } else {
                let mut all_embeddings = Vec::new();
                for (i, &length) in input_lengths.iter().enumerate().take(batch_size) {
                    all_embeddings.push(outputs.i((i, ..length))?);
                }
                Some(Tensor::cat(&all_embeddings, 0)?)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for EmbeddingGemma2Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}

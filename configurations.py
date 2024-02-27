
configurations = {
	'default': {
		'vocab_size': None, # default to tokenizer.vocab_size
		'hidden_size': 4096,
		'intermediate_size': 11008,
		'num_hidden_layers': 32,
		'num_attention_heads': 32,
		'num_key_value_heads': None,
		'hidden_act':  'silu',
		'max_position_embeddings': None, # default to settings.MAX_LENGTH
		'initializer_range': 0.02,
		'rms_norm_eps': 1e-06,
		'use_cache': True,
		'pad_token_id': None,
		'bos_token_id': 1,
		'eos_token_id': 2,
		'pretraining_tp': 1,
		'tie_word_embeddings': False,
		'rope_theta': 10000.0,
		'rope_scaling': None,
		'attention_bias': False
	},
	'llama-7b': {
		'hidden_size': 4096,
		'intermediate_size': 11008,
		'num_hidden_layers': 32,
		'num_attention_heads': 32,
	},
	'llama-3b': {
		'hidden_size': 2048,
		'intermediate_size': 11008,
		'num_hidden_layers': 32,
		'num_attention_heads': 32,
	},
	'llama-1500m': {
		'hidden_size': 1536,
		'intermediate_size': 11008,
		'num_hidden_layers': 24,
		'num_attention_heads': 32,
	},
	'llama-1b': {
		'hidden_size': 1536,
		'intermediate_size': 11008//2,
		'num_hidden_layers': 24,
		'num_attention_heads': 32,
	},
	'llama-300m': {
		'hidden_size': 480,
		'intermediate_size': 11008,
		'num_hidden_layers': 16,
		'num_attention_heads': 24,
	},
	'llama2c-110m': {
		'tie_word_embeddings': True,
		'hidden_size': 768,
		'intermediate_size': 2048,
		'num_hidden_layers': 12,
		'num_attention_heads': 12,
		'num_key_value_heads': 12,
		'max_position_embeddings': 1024,
	},
	'smol-llama-62m-tied': {
		'tie_word_embeddings': True,
		'hidden_size': 768,
		'intermediate_size': 2048,
		'num_hidden_layers': 6,
		'num_attention_heads': 24,
		'num_key_value_heads': 8,
		'max_position_embeddings': 1024,
	},
	'smol-llama-81m-tied': {
		'tie_word_embeddings': True,
		'hidden_size': 768,
		'intermediate_size': 3072,
		'num_hidden_layers': 6,
		'num_attention_heads': 24,
		'num_key_value_heads': 24,
		'max_position_embeddings': 1024,
	},
	'smol-llama-101m': {
		'hidden_size': 768,
		'intermediate_size': 3072,
		'num_hidden_layers': 6,
		'num_attention_heads': 24,
		'num_key_value_heads': 8,
		'max_position_embeddings': 1024,
	},
	'smol-llama-220m': {
		'hidden_size': 1024,
		'intermediate_size': 4096,
		'num_hidden_layers': 10,
		'num_attention_heads': 32,
		'num_key_value_heads': 8,
		'max_position_embeddings': 2048,
	},
	'llama-86m': {
		'hidden_size': 256,
		'intermediate_size': 11008,
		'num_hidden_layers': 8,
		'num_attention_heads': 8,
	},
	'llama-51m': {
		'hidden_size': 256,
		'intermediate_size': 11008,
		'num_hidden_layers': 4,
		'num_attention_heads': 4,
	},
	'llama2c-42m': {
		'tie_word_embeddings': True,
		'hidden_size': 512,
		'intermediate_size': 1376,
		'num_hidden_layers': 8,
		'num_attention_heads': 8,
		'max_position_embeddings': 1024,
	},
	'llama-34m': {
		'hidden_size': 256,
		'intermediate_size': 11008//2,
		'num_hidden_layers': 4,
		'num_attention_heads': 4,
	},
	'llama-25m': {
		'hidden_size': 256,
		'intermediate_size': 11008//4,
		'num_hidden_layers': 4,
		'num_attention_heads': 4,
	},
	'llama-25m': {
		'hidden_size': 256,
		'intermediate_size': 11008//4,
		'num_hidden_layers': 2,
		'num_attention_heads': 2,
	},
	'llama-19m': {
		'hidden_size': 256,
		'intermediate_size': 11008//8,
		'num_hidden_layers': 2,
		'num_attention_heads': 2,
	},
	'llama2c-15m': {
		'tie_word_embeddings': True,
		'hidden_size': 288,
		'intermediate_size': 768,
		'num_hidden_layers': 6,
		'num_attention_heads': 6,
		'max_position_embeddings': 256,
	},
	'llama-9m': {
		'hidden_size': 128,
		'intermediate_size': 11008//8,
		'num_hidden_layers': 2,
		'num_attention_heads': 2,
	}
}

def get_configuration(settings, tokenizer):
	config = configurations['default'].copy()
	config.update(configurations[settings.configuration])

	if config['vocab_size'] is None: config['vocab_size'] = tokenizer.vocab_size
	if config['max_position_embeddings'] is None: config['max_position_embeddings'] = settings.MAX_LENGTH

	return config

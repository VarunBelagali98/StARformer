from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

#python run_star_atari.py --seed 123 --data_dir_prefix /data/add_disk0/varun/atari/dqn_replay/ --epochs 10 --num_steps 500000 --num_buffers 50 --batch_size 16 -tee ./logs/ttm_v1_rwd_breakout_b16_sq30.txt
#python run_star_atari.py --seed 123 --data_dir_prefix /data/add_disk0/varun/atari/dqn_replay/ --epochs 10 --num_steps 500000 --num_buffers 50 --batch_size 16 --seq_len 30 --model_type 'star' --game 'Breakout' --save_dir /data/add_disk0/varun/atari/save/ttm_v2 tee 2>&1 | tee ./logs/ttm_v2_rwd_breakout_b16_sq30.txt
class GELU(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, input):
		return F.gelu(input) 

class CausalSelfAttention(nn.Module):
	def __init__(self, config, N_head=6, D=128, T=30):
		super().__init__()
		assert D % N_head == 0
		self.config = config
		

		self.N_head = N_head
		
		self.key = nn.Linear(D, D)
		self.query = nn.Linear(D, D)
		self.value = nn.Linear(D, D)
		
		self.attn_drop = nn.Dropout(config.attn_pdrop)
		self.resd_drop = nn.Dropout(config.resid_pdrop)
		
		self.proj = nn.Linear(D, D)
	def forward(self, x, mask=None):
		# x: B * N * D
		B, N, D = x.size()
		
		q = self.query(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
		k = self.key(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
		v = self.value(x.view(B*N, -1)).view(B, N, self.N_head, D//self.N_head).transpose(1, 2)
		
		A = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
		if mask is not None:
			A = A.masked_fill(mask[:,:,:N,:N] == 0, float('-inf'))
		A = F.softmax(A, dim=-1)
		A_drop = self.attn_drop(A)
		y = (A_drop @ v).transpose(1, 2).contiguous().view(B, N, D)
		y = self.resd_drop(self.proj(y))
		return y, A


	
class PatchEmb(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		D, iD = config.D, config.local_D
		p1, p2 = config.patch_size
		c, h, w = config.img_size

		maxt = config.maxT
		self.patch_embedding = nn.Sequential(
			Rearrange('b t c (h p1) (w p2) -> (b t) (h w) (p1 p2 c)', p1 = p1, p2 = p2),
			nn.Linear(p1*p2*c, iD)
		)
		
		if 'continuous' in config.action_type:
			self.action_emb = nn.Sequential(nn.Linear(config.vocab_size, iD), nn.Tanh())
		else:
			# +1 for mask
			self.action_emb = nn.Embedding(config.vocab_size+1, iD)
		
		if 'rwd' in config.model_type:
			  self.reward_emb = nn.Sequential(nn.Linear(1, iD), nn.Tanh())
		self.spatial_emb = nn.Parameter(torch.zeros(1, h*w//p1//p2, iD))
		
		
		self.temporal_emb = nn.Parameter(torch.zeros(1, maxt, D))
		if 'xconv' not in config.model_type:
			if 'vit' in config.model_type:
				self.conv_emb = nn.Sequential(
							Rearrange('bt c (h p1) (w p2) -> bt (h w) (p1 p2 c)', p1 = p1, p2 = p2),
							nn.Linear(p1*p2*c, iD),
							Rearrange('b n c -> b (n c)'),
							nn.Linear(h*w//p1//p2*iD, D),
							nn.Tanh()
						)
			else:
				self.conv_emb = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
										 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
										 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
										 nn.Flatten(), nn.Linear(3136, D), nn.Tanh())
	
	@torch.jit.ignore
	def no_weight_decay(self):
		return {'spatial_emb', 'temporal_emb'}
	
	def forward(self, states, actions, rewards=None):
		B, T, C, H, W = states.size()
		local_state_tokens = (self.patch_embedding(states)+ self.spatial_emb).reshape(B, T, -1, self.config.local_D)

		if 'xconv' in self.config.model_type or 'stack' in self.config.model_type:
			global_state_tokens = 0
		else:
			global_state_tokens = self.conv_emb(states.reshape(-1, C, H, W)).reshape(B, T, -1) + self.temporal_emb[:, :T]
		local_action_tokens = self.action_emb(actions.reshape(-1, 1)).reshape(B, T, -1).unsqueeze(2) # B T 1 iD

		if 'rwd' in self.config.model_type:
			local_reward_tokens = self.reward_emb(rewards.reshape(-1, 1)).reshape(B, T, -1).unsqueeze(2)
			local_tokens = torch.cat((local_action_tokens, local_state_tokens, local_reward_tokens), dim=2)
		else:
			local_tokens = torch.cat((local_action_tokens, local_state_tokens), dim=2)
		return local_tokens, global_state_tokens, self.temporal_emb[:, :T]
	
class _SABlock(nn.Module):
	def __init__(self, config, N_head, D):
		super().__init__()
		self.ln1 = nn.LayerNorm(D)
		self.ln2 = nn.LayerNorm(D)
		self.attn = CausalSelfAttention(config, N_head=N_head, D=D)
		self.mlp = nn.Sequential(
				nn.Linear(D, 4*D),
				GELU(),
				nn.Linear(4*D, D),
				nn.Dropout(config.resid_pdrop)
			)

		
	def forward(self, x, mask=None):
		y, att = self.attn(self.ln1(x), mask)
		x = x + y
		x = x + self.mlp(self.ln2(x))
		return x, att
	
	
class SABlock(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		p1, p2 = config.patch_size
		c, h, w = config.img_size
		patch_count = h*w//p1//p2
		self.local_block = _SABlock(config, config.local_N_head, config.local_D)
		self.global_block = _SABlock(config, config.N_head, config.D)
		if 'fusion' in config.model_type:
			self.register_buffer("mask", torch.tril(torch.ones(config.maxT*2, config.maxT*2))
									 .view(1, 1, config.maxT*2, config.maxT*2))
		else:
			self.register_buffer("mask", torch.tril(torch.ones(config.maxT*2, config.maxT*2))
									 .view(1, 1, config.maxT*2, config.maxT*2))
			for i in range(0, config.maxT*2, 2):
				self.mask[0, 0, i, i-1] = 1.
		self.local_norm = nn.LayerNorm(config.local_D)
		if 'stack' not in config.model_type:
			if 'rwd' in config.model_type:
				self.local_global_proj = nn.Sequential(
					nn.Linear((patch_count+1+1)*config.local_D, config.D), # +1 for action token +1 for reward token
					nn.LayerNorm(config.D)
				)
			else:
				self.local_global_proj = nn.Sequential(
						nn.Linear((patch_count+1)*config.local_D, config.D), # +1 for action token
						nn.LayerNorm(config.D)
				)
		
		
	def forward(self, local_tokens, global_tokens, temporal_emb=None):
		B, T, P, d = local_tokens.size()
		local_tokens, local_att = self.local_block(local_tokens.reshape(-1, P, d))
		local_tokens = local_tokens.reshape(B, T, P, d)
		lt_tmp = self.local_norm(local_tokens.reshape(-1, d)).reshape(B*T, P*d)
		lt_tmp = self.local_global_proj(lt_tmp).reshape(B, T, -1)
		if 'fusion' in self.config.model_type or 'xconv' in self.config.model_type:
			global_tokens += lt_tmp
			if ('xconv' in self.config.model_type) and (temporal_emb is not None):
				global_tokens += temporal_emb
			global_tokens, global_att = self.global_block(global_tokens, self.mask)
			return local_tokens, global_tokens, local_att, global_att
		else:
			#if temporal_emb is not None:
				#lt_tmp += temporal_emb
			#global_tokens = torch.stack((lt_tmp, global_tokens), dim=2).view(B, -1, self.config.D)
			#global_tokens, global_att = self.global_block(global_tokens, self.mask)
			return local_tokens, lt_tmp, None, None
				
				
	
class Starformer(nn.Module):
	
	def __init__(self, config):
		super().__init__()
		self.config = config

		D, iD = config.D, config.local_D
		p1, p2 = config.patch_size
		c, h, w = config.img_size
		patch_count = h*w//p1//p2
		maxt = config.maxT
		
		self.token_emb = PatchEmb(config)
		self.blocks = nn.ModuleList([SABlock(config) for _ in range(config.n_layer)])
		
		self.local_pos_drop = nn.Dropout(config.pos_drop)
		self.global_pos_drop = nn.Dropout(config.pos_drop)
		if 'stack' in config.model_type:
			if 'rwd' in config.model_type:
				self.local_global_proj = nn.Sequential(
					nn.Linear((patch_count+1+1)*config.local_D, config.D), # +1 for action token +1 for reward token
					nn.LayerNorm(config.D)
				)
			else:
				self.local_global_proj = nn.Sequential(
						nn.Linear((patch_count+1)*config.local_D, config.D), # +1 for action token
						nn.LayerNorm(config.D)
				)
		
		self.ln_head = nn.LayerNorm(config.D)
		if 'continuous' in config.action_type:
			self.head = nn.Sequential(
					*([nn.Linear(config.D, config.vocab_size)] + [nn.Tanh()])
				)
		else:
			self.head = nn.Linear(config.D, config.vocab_size)
		
		self.apply(self._init_weights)
		#self.ttm_block = TTM_Block(config, 10, 16, config.local_N_head, config.local_D)
		self.ttm_block = TTM_Block(config, 96, 24, config.local_N_head, config.local_D)
		
		print("number of parameters: %d" % sum(p.numel() for p in self.parameters()))
		
	def configure_optimizers(self, train_config):
		"""
		This long function is unfortunately doing something very simple and is being very defensive:
		We are separating out all parameters of the model into two buckets: those that will experience
		weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
		We are then returning the PyTorch optimizer object.
		"""

		# separate out all parameters to those that will and won't experience regularizing weight decay
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
		for mn, m in self.named_modules():
			for pn, p in m.named_parameters():
				fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

				if pn.endswith('bias'):
					# all biases will not be decayed
					no_decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					# weights of whitelist modules will be weight decayed
					decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					# weights of blacklist modules will NOT be weight decayed
					no_decay.add(fpn)

		no_decay.add('token_emb.spatial_emb')
		no_decay.add('token_emb.temporal_emb')

		# validate that we considered every parameter
		param_dict = {pn: p for pn, p in self.named_parameters()}
		inter_params = decay & no_decay
		union_params = decay | no_decay
		assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
		assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
													% (str(param_dict.keys() - union_params), )

		# create the pytorch optimizer object
		optim_groups = [
			{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
			{"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
		]
		optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
		return optimizer
		
	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
			
	def get_loss(self, pred, target):
		if 'continuous' in self.config.action_type:
			return F.mse_loss(pred, target, reduction='none')
		else:
			return F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1), reduction='none')

	
	def forward(self, states, actions, targets=None, rewards=None):
		# actions should be already padded by dataloader

		local_tokens, global_state_tokens, temporal_emb = self.token_emb(states, actions, rewards=rewards)
		local_tokens = self.local_pos_drop(local_tokens)
		if ('xconv' not in self.config.model_type) and ('stack' not in self.config.model_type):
			global_state_tokens = self.global_pos_drop(global_state_tokens)
		
		B, T, P, d = local_tokens.size()
		local_atts, global_atts = [], []
		if 'stack' in self.config.model_type:
			for i, blk in enumerate(self.blocks):
				local_tokens, local_att = blk.local_block(local_tokens.reshape(-1, P, d))
				local_att = local_att.detach()
				# for return attention maps
				# local_atts.append(local_att)
			global_state_tokens = self.local_global_proj(local_tokens.reshape(B, T, -1)) + temporal_emb
			for i, blk in enumerate(self.blocks):
				global_state_tokens, global_att = blk.global_block(global_state_tokens, blk.mask)
				global_att = global_att.detach()
				# for return attention maps
				# global_atts.append(global_att)
		else:
			for i, blk in enumerate(self.blocks):
				if i == 0:
					local_tokens, global_state_tokens, local_att, global_att = blk(local_tokens, global_state_tokens, temporal_emb)
				else:
					local_tokens, global_state_tokens, local_att, global_att = blk(local_tokens, global_state_tokens, temporal_emb)
				local_att = local_att #.detach()
				global_att = global_att #.detach()
				# for return attention maps
				# local_atts.append(local_att)
				# global_atts.append(global_att)
		
		#y = self.head(self.ln_head(global_state_tokens))
		x = self.ttm_block(local_tokens)
		x = x + global_state_tokens
		y = self.head(x)
		#print(x.shape, y.shape)
		loss = None
		if targets is not None:
			loss = self.get_loss(y, targets)
		return y[:, -1], (local_atts, global_atts), loss

class TTM_Unit(nn.Module):
	def __init__(self, Mem_len, sum_len, D):
		super().__init__()
		
		self.read_alpha = nn.Linear(D, sum_len)
		self.write_alpha = nn.Linear(D, Mem_len)
		self.sum_len = sum_len
		self.Mem_len = Mem_len

	def summary(self, x, op="read"):
		B, N, D = x.shape

		if op == "read":
			alpha = self.read_alpha(x)
		elif op == "write":
			alpha = self.write_alpha(x)
		else:
			print("unknown op")
			return None

		alpha = F.softmax(alpha, dim=-1)  #B X N X sum_len 
		x = torch.transpose(x, -1, -2)    # B X D X N

		summary = x @ alpha               # B X D X sum_len
		summary = torch.transpose(summary, -1, -2)   #B X sum_len X D 
		#print(op, x.shape, summary.shape)
		return summary
	
	def forward(self, x):
		return self.summary(x, "read")
	
	def read(self, x):
		return self.summary(x, "read")

	def write(self, x):
		return  self.summary(x, "write")

class TTM_Block(nn.Module):
	# how to back prop ?
	# what is memeory for first time step ?
	# use TTM at each layer or only final layer of stepformer ?
	def __init__(self, config, Mem_len, sum_lem, N_head, local_D):
		super().__init__()
		self.ttm = TTM_Unit(Mem_len, sum_lem, local_D)
		self.mem = None
		self.processor = _SABlock(config, N_head=N_head, D=local_D)
		
		self.local_proj = nn.Sequential(
				#nn.Linear(sum_lem*config.local_D, config.D), # +1 for action token +1 for reward token
				nn.Linear(sum_lem*config.local_D, config.D),
				nn.LayerNorm(config.D)
			)

	def forward(self, x):
		B, T, P, d = x.size()
		global_tokens = []
		projs = []
		for i in range(0, T):
			#print("time step", i)
			#first read, there is no memory avail
			if i == 0:
				#get summary
				summary = self.ttm.read(x[:, i])
				self.mem = summary
				#process summary
				#print(x[:,i].shape)
				#out, attn = self.processor(torch.cat([summary,x[:, i]], dim=-2))
				out, attn = self.processor(summary)
				proj = self.projector(out, B)
				projs.append(proj)
				#print(out.shape)
				#write to memory
				self.mem = self.ttm.write(torch.cat([self.mem,x[:, i], out], dim=-2))
			else:
				#get summary
				summary = self.ttm.read(x[:, i])
				#process memory
				#out, attn = self.processor(torch.cat([summary,x[:, i]], dim=-2))
				out, attn = self.processor(summary)
				proj = self.projector(out, B)
				projs.append(proj)
				#print(out.shape)
				#write to memory
				self.mem = self.ttm.write(torch.cat([self.mem,x[:, i], out], dim=-2))
		projs = torch.cat(projs, dim=1)
		return projs

	def projector(self, x, B):
		#print("proj", x.shape)
		x = self.local_proj(x.reshape(B, 1,-1))
		#x = torch.mean(x, dim=1, keepdim=True)
		#x = self.local_proj(x)
		#print("ttm proj", x.shape)
		return x
	
#------------------------------------------------------------------------
	
		
class StarformerConfig:
	embd_pdrop = 0.1
	resid_pdrop = 0.1
	attn_pdrop = 0.1
	action_type = "discrete"

	def __init__(self, vocab_size, **kwargs):
		self.vocab_size = vocab_size
		for k,v in kwargs.items():
			setattr(self, k, v)
		assert self.img_size is not None and self.patch_size is not None
		assert self.D % self.N_head == 0
		C, H, W = self.img_size
		pH, pW = self.patch_size
		
		
if __name__ == "__main__":
	mconf = StarformerConfig(4, img_size = (4, 84, 84), patch_size = (7, 7), context_length=30, pos_drop=0.1, resid_drop=0.1,
						  N_head=8, D=192, local_N_head=4, local_D=64, model_type='star', max_timestep=100, n_layer=6, C=4, maxT=30)


	model = Starformer(mconf)
	model = model #.cuda()
	dummy_states = torch.randn(3, 28, 4, 84, 84) #.cuda()
	dummy_actions = torch.randint(0, 4, (3, 28, 1), dtype=torch.long) #.cuda()
	output, atn, loss = model(dummy_states, dummy_actions, None)
	print (output.size(), output)


	#dummy_states = torch.randn(3, 1, 4, 84, 84).cuda()
	#dummy_actions = torch.randint(0, 4, (3, 1, 1), dtype=torch.long).cuda()
	#output, atn, loss = model(dummy_states, dummy_actions, None)
	#print (output.size(), output)

	# other test codes
#     mconf = StarformerConfig(4, img_size = (4, 84, 84), patch_size = (7, 7), context_length=30, pos_drop=0.1, resid_drop=0.1,
#                           N_head=8, D=192, local_N_head=4, local_D=64, model_type='star', max_timestep=100, n_layer=6, C=4, maxT = 30)

#     model = Starformer(mconf)
#     model = model.cuda()
#     dummy_states = torch.randn(3, 30, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 30, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)
	
#     dummy_states = torch.randn(3, 1, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 1, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)

	
	
#     mconf = StarformerConfig(4, img_size = (4, 84, 84), patch_size = (7, 7), context_length=30, pos_drop=0.1, resid_drop=0.1,
#                           N_head=8, D=192, local_N_head=4, local_D=64, model_type='star_fusion', max_timestep=100, n_layer=6, C=4, maxT=30)
	
#     model = Starformer(mconf)
#     model = model.cuda()
#     dummy_states = torch.randn(3, 28, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 28, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)
	
#     dummy_states = torch.randn(3, 1, 4, 84, 84).cuda()
#     dummy_actions = torch.randint(0, 4, (3, 1, 1), dtype=torch.long).cuda()
#     output, atn, loss = model(dummy_states, dummy_actions, None)
#     print (output.size(), output)
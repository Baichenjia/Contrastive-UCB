import torch
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple
from collections import namedtuple
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.utils.tensor import select_at_indexes, valid_mean
import numpy as np
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.logging import logger
from src.rlpyt_buffer import AsyncPrioritizedSequenceReplayFrameBufferExtended, \
	AsyncUniformSequenceReplayFrameBufferExtended
from src.models import from_categorical, to_categorical
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
	["observation", "action", "reward", "done"])
ModelSamplesToBuffer = namedarraytuple("SamplesToBuffer",
	["observation", "action", "reward", "done", "value"])

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr"])
ModelOptInfo = namedtuple("OptInfo", ["loss", "gradNorm",
									  "tdAbsErr",
									  "modelRLLoss",
									  "RewardLoss",
									  "modelGradNorm",
									  "SPRLoss",
									  "ModelSPRLoss",
									  "bonus"])

EPS = 1e-6  # (NaN-guard)


class SPRCategoricalDQN(CategoricalDQN):
	"""Distributional DQN with fixed probability bins for the Q-value of each
	action, a.k.a. categorical."""

	def __init__(self,
				 bonus_factor=0.1,       # TODO
				 count_action=False,     # TODO
				 t0_spr_loss_weight=1.,
				 model_rl_weight=1.,
				 reward_loss_weight=1.,
				 model_spr_weight=1.,
				 time_offset=0,
				 distributional=1,
				 jumps=0,
				 **kwargs):
		super().__init__(**kwargs)
		self.opt_info_fields = tuple(f for f in ModelOptInfo._fields)  # copy
		self.t0_spr_loss_weight = t0_spr_loss_weight
		self.model_spr_weight = model_spr_weight

		self.reward_loss_weight = reward_loss_weight
		self.model_rl_weight = model_rl_weight
		self.time_offset = time_offset
		self.jumps = jumps

		if not distributional:
			self.rl_loss = self.dqn_rl_loss
		else:
			self.rl_loss = self.dist_rl_loss

		# TODO
		self.bonus_factor = bonus_factor
		self.bonus_return_ = np.zeros((16, 32))
		self.bonus_flag = bonus_factor > 0.0
		self.count_action = count_action

	def initialize_replay_buffer(self, examples, batch_spec, async_=False):
		example_to_buffer = ModelSamplesToBuffer(
			observation=examples["observation"],
			action=examples["action"],
			reward=examples["reward"],
			done=examples["done"],
			value=examples["agent_info"].p,
		)
		replay_kwargs = dict(
			example=example_to_buffer,
			size=self.replay_size,
			B=batch_spec.B,
			batch_T=self.jumps+1+self.time_offset,
			discount=self.discount,
			n_step_return=self.n_step_return,
			rnn_state_interval=0,
		)

		if self.prioritized_replay:
			replay_kwargs['alpha'] = self.pri_alpha
			replay_kwargs['beta'] = self.pri_beta_init
			# replay_kwargs["input_priorities"] = self.input_priorities
			buffer = AsyncPrioritizedSequenceReplayFrameBufferExtended(**replay_kwargs)
		else:
			buffer = AsyncUniformSequenceReplayFrameBufferExtended(**replay_kwargs)

		self.replay_buffer = buffer

	def optim_initialize(self, rank=0):
		"""Called in initilize or by async runner after forking sampler."""
		self.rank = rank
		try:
			# We're probably dealing with DDP
			self.optimizer = self.OptimCls(self.agent.model.module.parameters(),
				lr=self.learning_rate, **self.optim_kwargs)
			self.model = self.agent.model.module
		except:
			self.optimizer = self.OptimCls(self.agent.model.parameters(),
				lr=self.learning_rate, **self.optim_kwargs)
			self.model = self.agent.model
		if self.initial_optim_state_dict is not None:
			self.optimizer.load_state_dict(self.initial_optim_state_dict)
		if self.prioritized_replay:
			self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)

		# TODO
		# initialize an diagonal matrix
		if self.bonus_flag:
			if self.count_action:
				self.Lambda_total = torch.diag(torch.zeros(self.agent.n_atoms)+1e-5).to(self.agent.device)
			else:
				self.Lambda_total = torch.diag(torch.zeros(self.agent.n_atoms*self.model.num_actions)+1e-5).to(self.agent.device)

	def samples_to_buffer(self, samples):
		"""Defines how to add data from sampler into the replay buffer. Called
		in optimize_agent() if samples are provided to that method.  In
		asynchronous mode, will be called in the memory_copier process."""
		return ModelSamplesToBuffer(
			observation=samples.env.observation,
			action=samples.agent.action,
			reward=samples.env.reward,
			done=samples.env.done,
			value=samples.agent.agent_info.p,
		)

	def optimize_agent(self, itr, samples=None, sampler_itr=None):
		""" 主要用于训练的函数
		Extracts the needed fields from input samples and stores them in the
		replay buffer.  Then samples from the replay buffer to train the agent
		by gradient updates (with the number of updates determined by replay
		ratio, sampler batch size, and training batch size).  If using prioritized
		replay, updates the priorities for sampled training batches.
		"""
		itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.=
		if samples is not None:
			samples_to_buffer = self.samples_to_buffer(samples)
			self.replay_buffer.append_samples(samples_to_buffer)
		opt_info = ModelOptInfo(*([] for _ in range(len(ModelOptInfo._fields))))
		if itr < self.min_itr_learn:
			return opt_info
		for _ in range(self.updates_per_optimize):
			# TODO: 计算特征，累加 Lambda 矩阵.
			# sample. all_observation.shape=[16, 32, 4, 1, 84, 84], all_reward.shape=[16, 32]
			# return_.shape=(6, 32), done.shape=(6, 32), done_n.shape=(6, 32). 原因是 n_step_return=10
			samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
			if self.bonus_flag:
				with torch.no_grad():
					# print("---begin")
					# print(samples_from_replay.done.shape)
					# calculate the Lambda matrix in this iteration
					B1, B2 = samples_from_replay.all_observation.squeeze().shape[:2]        # 16, 32
					img_size = samples_from_replay.all_observation.squeeze().shape[-3:]
					obs_reshape = samples_from_replay.all_observation.squeeze().reshape(B1*B2, *img_size)   # (512, 4, 84, 84)
					obs_reshape_transform = self.agent.model.transform(obs_reshape, augment=False)          # (512, 4, 84, 84)
					obs_features = self.agent.model.stem_forward(obs_reshape_transform.to(self.agent.device))   # conv features (512, 64, 7, 7)
					obs_features = self.agent.model.head_forward(obs_features, None, None, False)           # (512, num_action, 51)
					assert (obs_features >= 0.0).all()

					if self.count_action:       # extract features of the action
						# (512, 51). after select the action
						obs_features = select_at_indexes(samples_from_replay.all_action.reshape(B1*B2).to(self.agent.device), obs_features)

					obs_features = obs_features.reshape(B1*B2, -1)
					# print("** obs_features:", obs_features.shape)
					Lambda_iter = torch.matmul(obs_features.unsqueeze(-1), obs_features.unsqueeze(1))   # (512, 51*n_action, 51*n_action)
					# print("** Lambda_iter:", Lambda_iter.shape)
					Lambda_iter_sum = torch.sum(Lambda_iter, dim=0)         # (306, 306)

					# calculate the bonus
					self.Lambda_total += Lambda_iter_sum      # update the global Lambda matrix
					bonus_phi_lambda = torch.matmul(obs_features.unsqueeze(1), torch.inverse(self.Lambda_total))  # (512, 1, 51*n_action) * (51*n_action, 51*n_action) = (512, 1, 51*n_action)
					# print("isnan 0:", torch.isnan(torch.inverse(self.Lambda_total)).any())
					# print("isnan 0 Lambda < 0:", (self.Lambda_total < 0).sum())
					# print("isnan 0 Lambda inverse < 0:", (torch.inverse(self.Lambda_total) < 0).sum())

					bonus_phi_lambda_phi = torch.matmul(bonus_phi_lambda, obs_features.unsqueeze(-1))   # (512, 1, 51*n_action) * (512, 51*n_action, 1) = (512, 1, 1)

					# print("isnan 1:", torch.isnan(bonus_phi_lambda_phi).any())
					# print("isnan 1 < 0:", (bonus_phi_lambda_phi < 0).sum())
					# print("isnan 1 diagonal < 0:", (torch.diagonal(bonus_phi_lambda_phi) < 0).sum())

					bonus_sqrt = torch.sqrt(bonus_phi_lambda_phi).squeeze()     # (512,)
					# print("isnan sqrt:", torch.isnan(bonus_sqrt).any())
					bonus_sqrt = torch.nan_to_num(bonus_sqrt, nan=0.0)          # nan to zero
					# assert not torch.isnan(bonus_sqrt).any()

					bonus_weight = self.bonus_factor * bonus_sqrt               # (512,)
					bonus_final = bonus_weight.reshape(B1, B2).cpu().numpy()    # (B1, B2)  (16, 32)
					# print("bonus:", bonus_final.shape, np.mean(bonus_final))

					# multi-step bonus
					return_ = np.zeros_like(bonus_final)
					for n1 in range(0, B1-self.n_step_return):
						for n2 in range(0, self.n_step_return):
							return_[n1, :] += (self.discount ** n2) * bonus_final[n1+n2, :] * (1 - samples_from_replay.done.cpu().numpy()[n1, :])

					self.bonus_return_ = return_
					# print("original rewards:", samples_from_replay.all_reward.shape, samples_from_replay.all_reward.mean())
					# samples_from_replay.all_reward += bonus_final
					# print("rewards with bonus:", samples_from_replay.all_reward.shape, samples_from_replay.all_reward.mean())
					# print("---end \n")

			loss, td_abs_errors, model_rl_loss, reward_loss,\
			t0_spr_loss, model_spr_loss = self.loss(samples_from_replay)
			spr_loss = self.t0_spr_loss_weight*t0_spr_loss + self.model_spr_weight*model_spr_loss
			total_loss = loss + self.model_rl_weight*model_rl_loss + self.reward_loss_weight*reward_loss
			total_loss = total_loss + spr_loss
			self.optimizer.zero_grad()
			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(
				self.model.stem_parameters(), self.clip_grad_norm)
			if len(list(self.model.dynamics_model.parameters())) > 0:
				model_grad_norm = torch.nn.utils.clip_grad_norm_(
					self.model.dynamics_model.parameters(), self.clip_grad_norm)
			else:
				model_grad_norm = 0
			self.optimizer.step()
			if self.prioritized_replay:
				self.replay_buffer.update_batch_priorities(td_abs_errors)
			opt_info.loss.append(loss.item())
			opt_info.gradNorm.append(torch.tensor(grad_norm).item())  # grad_norm is a float sometimes, so wrap in tensor
			opt_info.modelRLLoss.append(model_rl_loss.item())
			opt_info.RewardLoss.append(reward_loss.item())
			opt_info.modelGradNorm.append(torch.tensor(model_grad_norm).item())
			opt_info.SPRLoss.append(spr_loss.item())
			opt_info.ModelSPRLoss.append(model_spr_loss.item())
			opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.
			if self.bonus_flag:
				opt_info.bonus.append(self.bonus_return_[0].mean())   # TODO: log
			# opt_info.bonus
			self.update_counter += 1
			if self.update_counter % self.target_update_interval == 0:
				self.agent.update_target(self.target_update_tau)
		self.update_itr_hyperparams(itr)
		return opt_info

	def dqn_rl_loss(self, qs, samples, index):
		"""
		Computes the Q-learning loss, based on: 0.5 * (Q - target_Q) ^ 2.
		Implements regular DQN or Double-DQN for computing target_Q values
		using the agent's target network.  Computes the Huber loss using
		``delta_clip``, or if ``None``, uses MSE.  When using prioritized
		replay, multiplies losses by importance sample weights.

		Input ``samples`` have leading batch dimension [B,..] (but not time).

		Calls the agent to compute forward pass on training inputs, and calls
		``agent.target()`` to compute target values.

		Returns loss and TD-absolute-errors for use in prioritization.

		Warning:
			If not using mid_batch_reset, the sampler will only reset environments
			between iterations, so some samples in the replay buffer will be
			invalid.  This case is not supported here currently.
		"""
		q = select_at_indexes(samples.all_action[index+1], qs).cpu()
		with torch.no_grad():
			target_qs = self.agent.target(samples.all_observation[index + self.n_step_return],
										  samples.all_action[index + self.n_step_return],
										  samples.all_reward[index + self.n_step_return])  # [B,A,P']
			if self.double_dqn:
				next_qs = self.agent(samples.all_observation[index + self.n_step_return],
									 samples.all_action[index + self.n_step_return],
									 samples.all_reward[index + self.n_step_return])  # [B,A,P']
				next_a = torch.argmax(next_qs, dim=-1)
				target_q = select_at_indexes(next_a, target_qs)
			else:
				target_q = torch.max(target_qs, dim=-1).values

			disc_target_q = (self.discount ** self.n_step_return) * target_q
			y = samples.return_[index] + (1 - samples.done_n[index].float()) * disc_target_q

		delta = y - q
		losses = 0.5 * delta ** 2
		abs_delta = abs(delta)
		if self.delta_clip > 0:  # Huber loss.
			b = self.delta_clip * (abs_delta - self.delta_clip / 2)
			losses = torch.where(abs_delta <= self.delta_clip, losses, b)
		td_abs_errors = abs_delta.detach()
		if self.delta_clip > 0:
			td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
		return losses, td_abs_errors

	def dist_rl_loss(self, log_pred_ps, samples, index):
		# print("dist_rl_loss:")
		# log_pred_ps.shape = (32, num_action, 51), index=0
		# samples.all_observation.shape = (16, 32, 4, 1, 84, 84), samples.all_reward.shape = (16, 32)
		# samples.return_.shape = (6, 32), samples.done_n.shape = (6, 32). 6 = 16-n_step+1
		delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
		z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
		# Make 2-D tensor of contracted z_domain for each data point,
		# with zeros where next value should not be added.
		next_z = z * (self.discount ** self.n_step_return)             # [P']      shape=[51,], discount=0.99, n_step_return=10
		next_z = torch.ger(1 - samples.done_n[index].float(), next_z)  # [B,P']    (32, 51)
		ret = samples.return_[index].unsqueeze(1)                      # [B,1]     (32, 1)  index处的样本
		# print("Return:", ret.shape, torch.mean(ret))
		# TODO: add bonus
		if self.bonus_flag:
			# bonus_return_.shape=(16, 32), bonus.shape=(32,1)
			bonus = torch.from_numpy(self.bonus_return_[index]).unsqueeze(1)
			#print("return:", torch.mean(ret).item(), "Bonus:", bonus.shape, torch.mean(bonus).item())
			ret = ret + bonus       # add to return

		next_z = torch.clamp(ret + next_z, self.V_min, self.V_max)     # [B,P']    (32, 51)

		z_bc = z.view(1, -1, 1)                                        # [1,P,1]
		next_z_bc = next_z.unsqueeze(1)                                # [B,1,P']
		abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
		projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)   # Most 0.
		# projection_coeffs is a 3-D tensor: [B,P,P']
		# dim-0: independent data entries
		# dim-1: base_z atoms (remains after projection)
		# dim-2: next_z atoms (summed in projection)

		with torch.no_grad():
			target_ps = self.agent.target(samples.all_observation[index + self.n_step_return],
										  samples.all_action[index + self.n_step_return],
										  samples.all_reward[index + self.n_step_return])  # [B,A,P']
			if self.double_dqn:
				next_ps = self.agent(samples.all_observation[index + self.n_step_return],
									 samples.all_action[index + self.n_step_return],
									 samples.all_reward[index + self.n_step_return])       # [B,A,P']
				next_qs = torch.tensordot(next_ps, z, dims=1)                              # [B,A]
				next_a = torch.argmax(next_qs, dim=-1)                                     # [B]
			else:
				target_qs = torch.tensordot(target_ps, z, dims=1)        # [B,A]
				next_a = torch.argmax(target_qs, dim=-1)                 # [B]
			target_p_unproj = select_at_indexes(next_a, target_ps)       # [B,P']
			target_p_unproj = target_p_unproj.unsqueeze(1)               # [B,1,P']
			target_p = (target_p_unproj * projection_coeffs).sum(-1)     # [B,P]
		p = select_at_indexes(samples.all_action[index + 1].squeeze(-1), log_pred_ps.cpu())  # [B,P]
		# p = torch.clamp(p, EPS, 1)  # NaN-guard.
		losses = -torch.sum(target_p * p, dim=1)             # Cross-entropy.

		target_p = torch.clamp(target_p, EPS, 1)
		KL_div = torch.sum(target_p * (torch.log(target_p) - p.detach()), dim=1)
		KL_div = torch.clamp(KL_div, EPS, 1 / EPS)           # Avoid <0 from NaN-guard.

		return losses, KL_div.detach()

	def loss(self, samples):
		"""
		Computes the Distributional Q-learning loss, based on projecting the
		discounted rewards + target Q-distribution into the current Q-domain,
		with cross-entropy loss.

		Returns loss and KL-divergence-errors for use in prioritization.
		"""
		if self.model.noisy:
			self.model.head.reset_noise()
		log_pred_ps, pred_rew, spr_loss\
			= self.agent(samples.all_observation.to(self.agent.device),
						 samples.all_action.to(self.agent.device),
						 samples.all_reward.to(self.agent.device),
						 train=True)                             # [B,A,P]
		rl_loss, KL = self.rl_loss(log_pred_ps[0], samples, 0)   # 调用 dqn_rl_loss
		if len(pred_rew) > 0:
			pred_rew = torch.stack(pred_rew, 0)
			with torch.no_grad():
				reward_target = to_categorical(samples.all_reward[:self.jumps+1].flatten().to(self.agent.device), limit=1).view(*pred_rew.shape)
			reward_loss = -torch.sum(reward_target * pred_rew, 2).mean(0).cpu()
		else:
			reward_loss = torch.zeros(samples.all_observation.shape[1],)
		model_rl_loss = torch.zeros_like(reward_loss)

		if self.model_rl_weight > 0:
			for i in range(1, self.jumps+1):
				jump_rl_loss, model_KL = self.rl_loss(log_pred_ps[i], samples, i)
				model_rl_loss = model_rl_loss + jump_rl_loss

		nonterminals = 1. - torch.sign(torch.cumsum(samples.done.to(self.agent.device), 0)).float()
		nonterminals = nonterminals[self.model.time_offset: self.jumps + self.model.time_offset+1]
		spr_loss = spr_loss*nonterminals
		if self.jumps > 0:
			model_spr_loss = spr_loss[1:].mean(0)
			spr_loss = spr_loss[0]
		else:
			spr_loss = spr_loss[0]
			model_spr_loss = torch.zeros_like(spr_loss)
		spr_loss = spr_loss.cpu()
		model_spr_loss = model_spr_loss.cpu()
		reward_loss = reward_loss.cpu()
		if self.prioritized_replay:
			weights = samples.is_weights
			spr_loss = spr_loss * weights
			model_spr_loss = model_spr_loss * weights
			reward_loss = reward_loss * weights

			# RL losses are no longer scaled in the c51 function
			rl_loss = rl_loss * weights
			model_rl_loss = model_rl_loss * weights

		return rl_loss.mean(), KL, \
			   model_rl_loss.mean(),\
			   reward_loss.mean(), \
			   spr_loss.mean(), \
			   model_spr_loss.mean(),

import torch
from typing import Optional

class POMDP:
    def __init__(
        self, 
        generative_model, 
        num_belief_particles = 1000,
        num_envs = 1,
        device = None,
        ):
        self._generative_model = generative_model
        self._num_belief_particles = num_belief_particles
        self._num_envs = num_envs
        self._device = device
        self._belief_particles = None

    def reset(self):
        self._generative_model.reset()
        init_states = self._generative_model.sample_initial_belief(num_samples=self._num_belief_particles)
        self._update_belief_distr(init_states)        

    @property
    def state_shape(self):
        return self._generative_model.state_shape

    @property
    def obs_shape(self):
        return self._generative_model.obs_shape

    @property
    def belief_particles(self):
        return self._belief_particles

    def step(self, state: torch.Tensor, action: torch.Tensor, active_env_ids: torch.Tensor | None=None, **kwargs):        
        if active_env_ids is None:
            active_env_ids = torch.ones((state.shape[0]), dtype=torch.bool, device=self._device)
        return self._generative_model(state, action, active_env_ids, **kwargs)

    def heuristics(self, state: torch.Tensor, action: torch.tensor, current_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._generative_model.heuristics(state, action, current_nodes=current_nodes)

    def is_goal(self, state: torch.Tensor) -> torch.Tensor:
        return self._generative_model.is_goal(state)

    def sample_belief(self, **kwargs):
        """Sample from current belief"""
        return self._belief_particles[self._belief_distr.sample((self._num_envs,))]

    def _update_belief_discrete_obs(self, action: torch.Tensor, observation: torch.Tensor, **kwargs):        
        states = torch.empty((0, self.state_shape[0]), dtype=self._belief_particles.dtype, device=self._device)
        while states.shape[0] < self._num_belief_particles:
            sampled_states = self.sample_belief(**kwargs)
            states = torch.cat((states, sampled_states), dim=0)

        # Sample next states
        active_env_ids = torch.ones((states.shape[0],), dtype=torch.bool, device=self._device)        
        next_states = torch.empty((0, self.state_shape[0]), dtype=self._belief_particles.dtype, device=self._device)
        terminals = torch.empty((0,), dtype=torch.bool, device=self._device)

        while next_states.shape[0] < self._num_belief_particles:
            n_iters = -(-self._num_belief_particles // self._num_envs)
            for i in range(n_iters):
                _active_env_ids = active_env_ids[i*self._num_envs:(i+1)*self._num_envs]
                _state = states[i*self._num_envs:(i+1)*self._num_envs]
                _action = action.repeat(self._num_envs, 1)
                outputs = self.step(_state, _action, _active_env_ids, belief_update=True)

                _next_states = outputs['next_state']
                _observations = outputs['observation']
                mask = _observations == observation # Filter out states that are not in the observation bucket  
                _next_states = _next_states[mask]          
                _terminals = outputs['terminal'][mask]
                next_states = torch.cat((next_states, _next_states), dim=0)
                terminals = torch.cat((terminals, _terminals), dim=0)

        next_states = next_states[:self._num_belief_particles]
        terminals = terminals[:self._num_belief_particles]

        # Filter out terminal states
        next_states = next_states[~terminals]
        if next_states.shape[0] == 0:
            print("COULDN'T SAMPLE NON-TERMINAL STATES")
            input('go on')

        self._update_belief_distr(next_states, **kwargs)

    def update_belief(self, action: torch.Tensor, observation: torch.Tensor, **kwargs):
        #print("=======================\nUPDATE BELIEF")        
        states = torch.empty((0, self.state_shape[0]), dtype=self._belief_particles.dtype, device=self._device)
        while states.shape[0] < self._num_belief_particles:
            sampled_states = self.sample_belief(**kwargs)            
            states = torch.cat((states, sampled_states), dim=0)   

        # Sample next states
        active_env_ids = torch.ones((states.shape[0],), dtype=torch.bool, device=self._device)        
        next_states = torch.empty((0, self.state_shape[0]), dtype=self._belief_particles.dtype, device=self._device)
        terminals = torch.empty((0,), dtype=torch.bool, device=self._device)

        n_iters = -(-self._num_belief_particles // self._num_envs)
        for i in range(n_iters):
            _active_env_ids = active_env_ids[i*self._num_envs:(i+1)*self._num_envs]
            _state = states[i*self._num_envs:(i+1)*self._num_envs]
            _action = action.repeat(self._num_envs, 1)
            outputs = self.step(_state, _action, _active_env_ids, belief_update=True)

            _next_states = outputs['next_state']
            _terminals = outputs['terminal']
            next_states = torch.cat((next_states, _next_states), dim=0)
            terminals = torch.cat((terminals, _terminals), dim=0)

        next_states = next_states[:self._num_belief_particles]
        terminals = terminals[:self._num_belief_particles]

        # Filter out terminal states
        next_states = next_states[~terminals]
        if next_states.shape[0] == 0:
            print("COULDN'T SAMPLE NON-TERMINAL STATES")
            input('go on')        

        # Compute the weights of next states             
        log_weights = self._generative_model.likelihood(
            observation,
            states, 
            next_states, 
            action.repeat(next_states.shape[0], 1),
            log_likelihood=True,
            is_encoded_observation=False,
        )

        weights = torch.exp(log_weights)
        sum_weights = weights.sum()

        if sum_weights > 0:
            weights /= sum_weights

        # ---- ESS check ----
        '''
        # Not being used right now
        B = weights.shape[0]
        ess = 1.0 / torch.sum(weights**2)        
        #if sum_weights < 1e-10:
        print(f"ess / B = {ess / B}, sum weights = {sum_weights}")              
        if (ess / B < 0.1) or (sum_weights < 1e-10):
            print("\n\n===============\nRegenerate")
            particles, log_weights = self._generative_model.regenerate_particles(
                next_states, 
                action, 
                observation
            )

            if particles is None:
                print("Couldn't regenerate particles")
                return False'''           
            

        # Resample particles proportional to log_weights
        distr = torch.distributions.Categorical(logits=log_weights) 
        indices = distr.sample((self._num_belief_particles,))
        next_belief_particles = next_states[indices]
        next_belief_particles = self._generative_model.postprocess_belief_particles(
            states, 
            action,
            observation,
            next_belief_particles,
        ) 
        self._update_belief_distr(next_belief_particles, **kwargs)
        return True

    def _update_belief_distr(self, states: torch.Tensor, **kwargs):        
        self._belief_particles = states
        self._belief_distr = torch.distributions.Categorical(
            torch.ones(
                (self._belief_particles.shape[0],), 
                dtype=torch.float32, 
                device=self._device
            )
        )
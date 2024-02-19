import numpy as np
# import bc_agent
# import bcq_agent
# import brac_dual_agent
# import brac_primal_agent
# import sac_agent

class group_set:
    
    def __init__(self, batch_size, agent_datasets, group_dataset) -> None:
        self.batch_size = batch_size
        self.agents_datasets = agent_datasets
        self.group_dataset = group_dataset
    
    def _get_action(self, agent, state):
        pass
        # actions, _ = agent.test_policies()['main'](state)
        # if isinstance(agent, bc_agent):
        #     pass
        # elif isinstance(agent, bcq_agent):
        #     pass
        # elif isinstance(agent, brac_dual_agent):
        #     pass
        # elif isinstance(agent, brac_primal_agent):
        #     pass
        # else:
        #     raise ValueError('the agent not belongs to the known agent types')
    
    def get_batch_samples(self):
        batch_indices = np.random.choice(self.group_dataset.size, self.batch_size)
        agent_datasets = self.agents_datasets
        group_dataset = self.group_dataset
        group_batch = group_dataset.get_batch(batch_indices)
        agents_batch = []
        for data_set in agent_datasets:
            agent_batch = data_set.get_batch(batch_indices)
            batch = dict(
                s1=agent_batch.s1,
                s2=agent_batch.s2,
                r=agent_batch.reward,
                dsc=agent_batch.discount,
                a1=agent_batch.a1,
                a2=agent_batch.a2,
                gs1=group_batch.s1,
                gs2=group_batch.s2,
                ga1=group_batch.a1,
                ga2=group_batch.a2
                )
            agents_batch.append(batch)
        
        return agents_batch   
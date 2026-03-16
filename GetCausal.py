import glob
import os
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from tigramite import data_processing
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
class CausalPrecursors():
    def __init__(self, name='crypto_data',
                 cond_ind_test='parcorr',#conditional independence test method, currently only support partial correlation test (ParCorr)
                 window=10,#maximum time lag to consider for causal links
                 sig_thres=0.05, #significance level for conditional independence test
                 var_names=['btc_vol','eth_vol'],#names of the features we want to analyze
                 depth=2,#maximum number of parents to consider for each variable
                 num_features=2):
            if cond_ind_test=='parcorr':
                self.cond_ind_test = ParCorr()
            elif cond_ind_test == '':
                raise ValueError('Not support yet!')
            if len(var_names) != num_features:
                raise ValueError('Give coincidence number and name of features!')
            self.name = name
            self.sig_thres = sig_thres
            self.window = window
            self.var_names = var_names
            self.depth = depth
            self.num_features = num_features

    def __call__(self, data):
       '''main function to get causal precursors for the target variable'''
       print('1. get causaity based on PCMCI for {} features with {} window'.\
            format(self.num_features, self.window))
       self.get_causal_precursors(data)
       print('2. plot line chart of causal strength for {} features with {} window'.\
             format(self.num_features, self.window))
       self.plot_heatmap()
       self.plot_line_chart()
       print('3. group causal drivers for the same causal activation time')
       self.group_causal_prescursors()
       print('5. get group {} trees'.\
            format(len(self.causal_link_groups.keys())))
       self.get_group_trees()
       return self
    def get_causal_precursors(self, data):
         '''1. Format Data to a 3D array of shape 
         2. Run PCMCI to get causal precursors for the target variable.'''
         dataframe = data_processing.DataFrame(np.array(data))
         self.pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=self.cond_ind_test)
         self.pcmci.run_pcmci(
            tau_max=self.window,
            pc_alpha=self.sig_thres)
    def plot_heatmap(self):
                print(f"Drawing causality heatmap for {self.name}...")
                val_matrix = self.pcmci.val_matrix
                combined_matrix = np.mean(np.abs(val_matrix), axis=2)
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(combined_matrix, cmap='YlOrRd') # 使用黄-橙-红渐变
                ax.set_xticks(np.arange(len(self.var_names)))
                ax.set_yticks(np.arange(len(self.var_names)))
                ax.set_xticklabels(self.var_names)
                ax.set_yticklabels(self.var_names)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                for i in range(len(self.var_names)):
                    for j in range(len(self.var_names)):
                        val = combined_matrix[i, j]
                        if val >= self.sig_thres: # 只标注显著的
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

                ax.set_title(f"Causal Strength Heatmap (Mean over {self.window} lags)")
                fig.colorbar(im, ax=ax, label='Causal Strength (abs val)')
                fig.tight_layout()
                save_path = f"{self.name}_causality_structure.png"
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
                plt.show()
    def plot_line_chart(self):
        val_matrix = np.abs(self.pcmci.val_matrix)
        target_idx = self.num_features - 1
        
        fig, ax = plt.subplots(figsize=(10, 5))
        lags = np.arange(self.window + 1)
        
        for i in range(self.num_features):
            label = f"{self.var_names[i]} -> {self.var_names[target_idx]}"
            ax.plot(lags[1:], val_matrix[i, target_idx, 1:], label=label, marker='o')
        
        ax.axhline(y=self.sig_thres, color='r', linestyle='--')
        ax.legend()
        plt.show()
    def group_causal_prescursors(self):
         impact_matrix = np.abs(self.pcmci.val_matrix)
         link_matrix=impact_matrix >= self.sig_thres
         self.causal_link_groups = {}
         self.causal_impact_groups = {}
         for tau in range(1, self.window + 1):
              #link for each timestamp
              link_at_timestep = link_matrix[:, -1,tau]
              driver_at_timestep = list(np.where(link_at_timestep >= self.sig_thres)[0])
              impact_at_timestep = impact_matrix[driver_at_timestep, -1, tau]
              driver_at_timestep = [i for i in driver_at_timestep if i != self.num_features-1]
              if len(driver_at_timestep) != 0:
                   self.causal_link_groups[str(tau)] = driver_at_timestep
                   self.causal_impact_groups[str(tau)] = impact_at_timestep
    def get_group_trees(self):
         self.group_nodes={}
         self.group_num_chid_nodes={}
         self.group_input_idx = {} 
         self.group_child_state_idx = {}
         for (timestep, causal_link) in self.causal_link_groups.items():
              node_dict, num_child_nodes, input_idx, child_state_idx = \
                self._get_one_tree(causal_link)
              self.group_nodes[timestep] = node_dict
              self.group_num_chid_nodes[timestep] = num_child_nodes
              self.group_input_idx[timestep] = input_idx
              self.group_child_state_idx[timestep] = child_state_idx
    def _get_num_child_nodes(self, node_dict):
           num_child_nodes = []
           for level in np.arange(self.depth, 0, -1):
                if level == self.depth:
                     for node_level in node_dict[str(level)]:
                       for node in node_level:
                               num_child_nodes.append(0)
                else:
                      for node_level in node_dict[str(level+1)]:
                        num_child_nodes.append(len(node_level)) 
           return num_child_nodes
    def _get_input_idx(self, node_dict):
        """get input index of node """
        input_idx = []
        
        for level in np.arange(self.depth,0,-1):
            for node_level in node_dict[str(level)]:
                for node in node_level:
                    input_idx.append([int(node)])   

        return input_idx

    def _get_child_state_idx(self, node_dict):
        """get state of child nodes of each node."""
        child_state_idx = []
        count = -1
        for level in np.arange(self.depth,0,-1):
            if level == self.depth:
                for node_level in node_dict[str(level)]:
                    for node in node_level:
                        child_state_idx.append([])
            else:
                for node_level in node_dict[str(level+1)]:
                    _child_state_idx = []

                    for node in node_level:
                        count += 1
                        _child_state_idx.append(count)
                    
                    child_state_idx.append(_child_state_idx)

        return child_state_idx

    def _get_one_tree(self, causal_link):
        """generate inputs for Causal LSTM for each single tree in group."""
        # generate feature dict for each level.
        node_dict = {}
        node_dict['1'] = [[self.num_features-1]]
        node_dict['2'] = [causal_link] # may change

        # generate inputs of CLSTM using node_dict
        num_child_nodes = self._get_num_child_nodes(node_dict)
        input_idx = self._get_input_idx(node_dict)
        child_state_idx = self._get_child_state_idx(node_dict)

        return node_dict, num_child_nodes, input_idx, child_state_idx
    
           
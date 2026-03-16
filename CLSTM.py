import torch
import torch.nn as nn

class NodeCell(nn.Module):
    def __init__(self, dim_hidden=16, dim_children=0):
        super(NodeCell, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_children = dim_children
        self.input_mapping = nn.Linear(1, dim_hidden)
        self._ifo_x = nn.Linear(dim_hidden, 3 * dim_hidden)
        self._ifo_h = nn.Linear(dim_hidden, 3 * dim_hidden)
        self._a_x = nn.Linear(dim_hidden, dim_hidden)
        self._a_h = nn.Linear(dim_hidden, dim_hidden)
        if dim_children > 0:
            self.child_x = nn.Linear(dim_hidden, dim_children * dim_hidden)
            self.child_h = nn.Linear(dim_hidden, dim_children * dim_hidden)
            
        self._1_x = nn.Linear(dim_hidden, dim_hidden)
        self._1_h = nn.Linear(dim_hidden, dim_hidden)
        self._2_x = nn.Linear(dim_hidden, dim_hidden)
        self._2_h = nn.Linear(dim_hidden, dim_hidden)

    def _horizontal_forward(self, inputs, h_prev, c_prev):
        # inputs: [batch, hidden], h_prev: [batch, hidden]
        ifo = torch.sigmoid(self._ifo_x(inputs) + self._ifo_h(h_prev))
        i, f, o = torch.split(ifo, self.dim_hidden, dim=-1)
        
        a = torch.tanh(self._a_x(inputs) + self._a_h(h_prev))
        c = i * a + f * c_prev
        h = o * torch.tanh(c)
        return h, c

    def _vertical_forward(self, inputs, h_prev, h_curr, child_n=None):
        if self.dim_children == 0 or child_n is None:
            neighborhood_influence = 0
        else:
            # child_n: [num_children, batch, hidden]
            child_r_gates = torch.sigmoid(self.child_x(inputs) + self.child_h(h_prev))
            child_r_gates = child_r_gates.view(self.dim_children, -1, self.dim_hidden)
            neighborhood_influence = torch.sum(child_r_gates * child_n, dim=0)

        n_1 = torch.sigmoid(self._1_x(inputs) + self._1_h(h_prev))
        n_2 = torch.sigmoid(self._2_x(inputs) + self._2_h(h_curr))
        n = n_1 * neighborhood_influence + n_2 * h_curr
        return n

    def forward(self, inputs, h_prev, c_prev, child_n=None):
        mapped_inputs = torch.relu(self.input_mapping(inputs))
        h, c = self._horizontal_forward(mapped_inputs, h_prev, c_prev)
        n = self._vertical_forward(mapped_inputs, h_prev, h, child_n)
        return h, c, n
class CausalCell(nn.Module):
    def __init__(self, dim_hidden, num_nodes, dim_child_nodes, input_idx, child_state_idx):
        super(CausalCell, self).__init__()
        self.num_nodes = num_nodes
        self.input_idx = input_idx
        self.child_state_idx = child_state_idx

        self.nodes = nn.ModuleList([
            NodeCell(dim_hidden, dim_children=dim_child_nodes[i]) 
            for i in range(num_nodes)
        ])

    def forward(self, inputs, h, c, n):

        new_h, new_c, new_n = h.clone(), c.clone(), n.clone()
        
        for i in range(self.num_nodes):
 
            _in_x = inputs[:, self.input_idx[i]].view(inputs.size(0), -1)
            
            _h_prev, _c_prev = h[i], c[i]
            
            if len(self.child_state_idx[i]) == 0:
                h_i, c_i, n_i = self.nodes[i](_in_x, _h_prev, _c_prev)
            else:
                child_n_states = torch.stack([new_n[j] for j in self.child_state_idx[i]], dim=0)
                h_i, c_i, n_i = self.nodes[i](_in_x, _h_prev, _c_prev, child_n_states)
            
            new_h[i], new_c[i], new_n[i] = h_i, c_i, n_i
            
        return new_n, new_h, new_c
class CLSTM(nn.Module):
    def __init__(self, num_nodes, num_hiddens, num_child_nodes, input_idx, child_state_idx, input_len, batch_size):
        super(CLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.input_len = input_len

        self.cell_stack = nn.ModuleList([
            CausalCell(num_hiddens, num_nodes, num_child_nodes, input_idx, child_state_idx)
            for _ in range(input_len)
        ])
        self.dense = nn.Linear(num_hiddens, 1)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        device = inputs.device
 
        h = torch.zeros(self.num_nodes, batch_size, self.num_hiddens, device=device)
        c = torch.zeros(self.num_nodes, batch_size, self.num_hiddens, device=device)
        n = torch.zeros(self.num_nodes, batch_size, self.num_hiddens, device=device)
        
        for t in range(self.input_len):
            n, h, c = self.cell_stack[t](inputs[:, t, :], h, c, n)

        return self.dense(n[-1])
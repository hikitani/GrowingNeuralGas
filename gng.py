import numpy as np
from tqdm import tqdm

from utils import exception

class GrowingNeuralGas():
    def __init__(self, init_weights: np.array, max_nodes: int = 150, max_age: int = 100,
                 lr_winner: np.float64 = 0.1, lr_nearest: np.float64 = 0.0006, step_of_needing_node: int = 300,
                 alpha: np.float64 = 0.5, betta: np.float64 = 0.0005, minkowski_p: np.float64 = 2, history_divisor: int = 10):
        
        if len(init_weights) != 2:
            raise AssertionError('During init Must be 2 vectors of weights')
        if not ((lr_nearest > 0 and lr_nearest < 1) and (lr_winner > 0 and lr_winner < 1)):
            raise AssertionError('Learning rate must belongs (0, 1)')
        if step_of_needing_node <= 0:
            raise AssertionError('step_of_needing_node must be > 0')
        if not (alpha > 0 and alpha < 1):
            raise AssertionError('alpha must belongs (0, 1)')
        if not (betta > 0 and betta < 1):
            raise AssertionError('betta must belongs (0, 1)')
        if minkowski_p <= 0:
            raise AssertionError('minkowski_p must be > 0')
        if max_nodes < 2:
            raise AssertionError('max_nodes must be >= 2')
        if max_age < 1:
            raise AssertionError('max_age must be >= 1')
        if history_divisor <= 0:
            raise AssertionError('history_divisor must be > 0')
            
        self.history_divisor = history_divisor
        self.history = dict()
        self.history['err'] = []
        
        self.n_features = len(init_weights[0])
        self.step_of_needing_node = step_of_needing_node
        self.alpha = alpha
        self.betta = betta
        self.max_nodes = max_nodes
        self.max_age = max_age + 1
        self.minkowski_p = np.float64(minkowski_p)
        self.lr_nearest = lr_nearest
        self.lr_winner = lr_winner
        
        self.vec_weights = np.ma.array(np.zeros(shape=(max_nodes, self.n_features), dtype=np.float64), mask=False)
        self.acc_errors = np.ma.array(np.zeros(shape=(max_nodes, 1), dtype=np.float64), mask=False)
        self.adj_matrix = np.ma.array(np.zeros(shape=(max_nodes, max_nodes), dtype=np.int), mask=False)
        
        self.vec_weights[[0, 1]] = init_weights
        self.vec_weights.mask[2:] = True
        self.acc_errors.mask[2:] = True
        self.adj_matrix.mask[2:] = True
        self.node_count = 2
        
        self._set_age(node_idx1=0, node_idx2=1, value=1)
        
    def _get_age(self, node_idx1: int, node_idx2: int) -> int:
        return self.adj_matrix[node_idx1, node_idx2] - 1
    
    def _set_age(self, node_idx1: int, node_idx2: int, value: int):
        self.adj_matrix[node_idx1, node_idx2] = value
        self.adj_matrix[node_idx2, node_idx1] = value
        
    def _set_ages(self, node_idx1: int, nodes_ids: np.array, value):
        self.adj_matrix[node_idx1, nodes_ids] = value
        self.adj_matrix[nodes_ids, node_idx1] = value
        
    def _add_node(self, weights: np.array, error: np.float64 = 0, linked_nodes_ids: np.array = []):
            
        unalloc_ids = np.where(self.vec_weights.mask[:, 0] == True)[0]
        if len(unalloc_ids) == 0:
            return -1
        
        first_unalloc_idx = unalloc_ids[0]
        
        self.vec_weights.mask[first_unalloc_idx] = False
        self.acc_errors.mask[first_unalloc_idx] = False
        self.adj_matrix.mask[first_unalloc_idx] = False
        
        self.vec_weights[first_unalloc_idx] = weights
        self.acc_errors[first_unalloc_idx] = error
        self.adj_matrix[first_unalloc_idx] = np.zeros(self.max_nodes, dtype=np.int)
        self._set_ages(first_unalloc_idx, linked_nodes_ids, 1)
        
        self.node_count += 1
        
        return first_unalloc_idx
        
    def _del_node(self, node_idx: int):
        self.vec_weights.mask[node_idx] = True
        self.acc_errors.mask[node_idx] = True
        self.adj_matrix.mask[node_idx] = True
        
        self.node_count -= 1
    
    def _del_nodes(self, node_ids: np.array):
        self.vec_weights.mask[node_ids] = True
        self.acc_errors.mask[node_ids] = True
        self.adj_matrix.mask[node_ids] = True
        
        self.node_count -= len(node_ids)
        
    def _get_two_neares_nodes_index(self, vec: np.array) -> tuple:
        vec_dist = np.ma.array(self._minkowski_dist_vec(vec, self.vec_weights), mask=False)
        vec_dist.mask = self.vec_weights.mask[:, 0]
        min_index1 = np.argmin(vec_dist)
        dist_node1 = vec_dist[min_index1]
        vec_dist.mask[min_index1] = True
        min_index2 = np.argmin(vec_dist)
        
        return (min_index1, min_index2, dist_node1)
    
    def _need_new_node(self, cur_iteration):
        return (self.node_count < self.max_nodes) and (cur_iteration % self.step_of_needing_node == 0)
    
    def _get_neighbours_nodes_indexes(self, node_idx: int) -> np.array:
        return np.where(self.adj_matrix[node_idx] > 0)[0]
        
    def _get_node_with_largest_error(self, linked_node_idx: int = None):
        if linked_node_idx == None:
            return np.argmax(self.acc_errors)
        
        neighbours_nodes_ids = self._get_neighbours_nodes_indexes(linked_node_idx)
        return neighbours_nodes_ids[np.argmax(self.acc_errors[neighbours_nodes_ids])]
    
    def _insert_node_btw(self, node_idx1, node_idx2):
        self.acc_errors[node_idx1] = self.acc_errors[node_idx1] * self.alpha
        self.acc_errors[node_idx2] = self.acc_errors[node_idx2] * self.alpha
        
        new_weights = (self.vec_weights[node_idx1] + self.vec_weights[node_idx2]) / 2
        new_error = self.acc_errors[node_idx1]
        new_adj_vec = np.zeros(self.max_nodes, dtype=np.int)
        new_node_idx = self._add_node(weights=new_weights, error=new_error, linked_nodes_ids=[node_idx1, node_idx2])
        
        if new_node_idx != -1:
            self._set_age(node_idx1, node_idx2, 0)
        
    def _minkowski_distance(self, vec1: np.array, vec2: np.array) -> np.float:
        return np.linalg.norm(vec1 - vec2, ord=self.minkowski_p)
#         return np.power(np.sum(np.power(np.abs(vec1 - vec2), self.minkowski_p)), 1 / self.minkowski_p)
    
    def _minkowski_dist_vec(self, vec: np.array, vecs: np.array) -> np.array:
        return np.apply_along_axis(lambda el_of_vecs: self._minkowski_distance(el_of_vecs, vec), 1, vecs)
    
    def step(self, x, cur_iteration):
        w_s_idx, w_t_idx, w_s_lp_norm = self._get_two_neares_nodes_index(x)
        self.acc_errors[w_s_idx] = self.acc_errors[w_s_idx] + w_s_lp_norm
        w_n_ids = self._get_neighbours_nodes_indexes(w_s_idx)
    
        self.vec_weights[w_s_idx] = self.vec_weights[w_s_idx] + self.lr_winner * (x - self.vec_weights[w_s_idx])
        for w_n_idx in w_n_ids:
            self.vec_weights[w_n_idx] = self.vec_weights[w_n_idx] + self.lr_nearest * (x - self.vec_weights[w_n_idx])

        self._set_ages(w_s_idx, w_n_ids, self.adj_matrix[w_s_idx, w_n_ids] + 1)
        self._set_age(w_s_idx, w_t_idx, 1)

        old_node_ids = w_n_ids[np.where(self.adj_matrix[w_s_idx, w_n_ids] > self.max_age)[0]]
        self._set_ages(w_s_idx, old_node_ids, 0)
        no_edges_ids = old_node_ids[np.where(np.sum(self.adj_matrix[old_node_ids], axis=-1) == 0)[0]]
        self._del_nodes(no_edges_ids)
        
        if self._need_new_node(cur_iteration):
            w_u_idx = self._get_node_with_largest_error()
            w_v_idx = self._get_node_with_largest_error(w_u_idx)
            self._insert_node_btw(w_u_idx, w_v_idx)
            
        self.acc_errors = self.acc_errors - self.betta * self.acc_errors
        
    def get_info_for_plot(self):
        edge_x = []
        edge_y = []

        text = []

        for i in range(self.max_nodes):
            if self.vec_weights.mask[i, 0] == True:
                continue

            for j in range(i+1, self.max_nodes):
                if self.vec_weights.mask[j, 0] == True:
                    continue

                if self.adj_matrix[i, j] > 0:
                    edge_x.append(self.vec_weights[i, 0])
                    edge_x.append(self.vec_weights[j, 0])
                    edge_x.append(None)

                    edge_y.append(self.vec_weights[i, 1])
                    edge_y.append(self.vec_weights[j, 1])
                    edge_y.append(None)

    #                 text.append((
    #                     (gng.vec_weights[i, 0] + zng.vec_weights[j, 0]) / 2,
    #                     (gng.vec_weights[i, 1] + gng.vec_weights[j, 1]) / 2,
    #                     f'age: {gng.adj_matrix[i, j] - 1}'
    #                 ))

        return edge_x, edge_y, text
    
    @exception(exc_type=KeyboardInterrupt, msg='Stopped')
    def fit(self, X, num_of_iter, verbose=True, pre_step=None, post_step=None):
        _iter = range(num_of_iter)
        if verbose:
            _tqdm_obj = tqdm(_iter)
            _iter = _tqdm_obj.__iter__()
            
        cur_iter = 0
        ids = list(range(len(X)))
        while True:
            random_order = np.random.permutation(ids)
            for x in X[random_order]:                
                if cur_iter >= num_of_iter - 1:
                    return
                
                if pre_step != None:
                    pre_step(self, cur_iter)
                
                self.step(x, cur_iter)
                
                if post_step != None:
                    post_step(self, cur_iter=cur_iter)
                
                if cur_iter % self.history_divisor == 0:
                    self.history['err'].append(np.mean(self.acc_errors))
                    
                _tqdm_obj.desc = 'mean err: {:.2f}'.format(self.history['err'][-1])
                cur_iter = next(_iter)
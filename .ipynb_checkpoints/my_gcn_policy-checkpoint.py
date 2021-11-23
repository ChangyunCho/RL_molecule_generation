import numpy as np
import gym
import tensorflow as tf

from baselines.common.distributions import make_pdtype,MultiCatCategoricalPdType,CategoricalPdType
import baselines.common.tf_util as U

class GCNPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, args, kind='small', atom_type_num = None):
        with tf.compat.v1.variable_scope(name):
            self._init(ob_space, ac_space, kind, atom_type_num, args)
            self.scope = tf.compat.v1.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind, atom_type_num, args):
        self.pdtype = MultiCatCategoricalPdType
        ### 0 Get input
        ob = {'adj': U.get_placeholder(name="adj", dtype=tf.float32, shape=[None,ob_space['adj'].shape[0],None,None]), 
              'node': U.get_placeholder(name="node", dtype=tf.float32, shape=[None,1,None,ob_space['node'].shape[2]])}
        
        # only when evaluating given action, at training time
        self.ac_real = U.get_placeholder(name='ac_real', dtype=tf.int64, shape=[None,4]) # feed groudtruth action
        
        ob_node = tf.keras.layers.Dense(8,activation=None,use_bias=False,name='emb')(ob['node']) # embedding layer
        
        if args.bn==1:
            ob_node = tf.keras.layers.BatchNormalization(axis=-1)(ob_node)
        if args.has_concat==1:
            emb_node = tf.concat((GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate),ob_node),axis=-1)
        else:
            emb_node = GCN_batch(ob['adj'], ob_node, args.emb_size, name='gcn1',aggregate=args.gcn_aggregate)
        if args.bn == 1:
            emb_node = tf.keras.layers.BatchNormalization(axis=-1)(emb_node)
        for i in range(args.layer_num_g-2):
            if args.has_residual==1:
                emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate)+self.emb_node1
            elif args.has_concat==1:
                emb_node = tf.concat((GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_'+str(i+1),aggregate=args.gcn_aggregate),self.emb_node1),axis=-1)
            else:
                emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, name='gcn1_' + str(i + 1),aggregate=args.gcn_aggregate)
            if args.bn == 1:
                emb_node = tf.keras.layers.BatchNormalization(axis=-1)(emb_node)
        emb_node = GCN_batch(ob['adj'], emb_node, args.emb_size, is_act=False, is_normalize=(args.bn == 0), name='gcn2',aggregate=args.gcn_aggregate)
        emb_node = tf.squeeze(emb_node,axis=1)  # B*n*f

        ### 1 only keep effective nodes
        ob_len = tf.reduce_sum(tf.squeeze(tf.cast(tf.cast(tf.reduce_sum(ob['node'], axis=-1),dtype=tf.bool),dtype=tf.float32),axis=-2),axis=-1)
        ob_len_first = ob_len-atom_type_num
        logits_mask = tf.sequence_mask(ob_len, maxlen=tf.shape(ob['node'])[2]) # mask all valid entry
        logits_first_mask = tf.sequence_mask(ob_len_first,maxlen=tf.shape(ob['node'])[2]) # mask valid entry -3 (rm isolated nodes)

        if args.mask_null==1:
            emb_node_null = tf.zeros(tf.shape(emb_node))
            emb_node = tf.where(condition=tf.tile(tf.expand_dims(logits_mask,axis=-1),(1,1,emb_node.get_shape()[-1])), x=emb_node, y=emb_node_null)

        ## get graph embedding
        emb_graph = tf.reduce_sum(emb_node, axis=1, keepdims=True)
        if args.graph_emb == 1:
            emb_graph = tf.tile(emb_graph, [1, tf.shape(emb_node)[1], 1])
            emb_node = tf.concat([emb_node, emb_graph], axis=2)

        ### 2 predict stop
        emb_stop = tf.keras.layers.Dense(args.emb_size, activation=tf.nn.relu, use_bias=False, name='linear_stop1')(emb_node)
        if args.bn==1:
            emb_stop = tf.keras.layers.BatchNormalization(axis=-1)(emb_stop)
        self.logits_stop = tf.reduce_sum(emb_stop,axis=1)
        self.logits_stop = tf.keras.layers.Dense(2, activation=None, name='linear_stop2_1')(self.logits_stop)  # B*2
        
        # explicitly show node num
        stop_shift = tf.constant([[0,args.stop_shift]],dtype=tf.float32)
        pd_stop = CategoricalPdType(-1).pdfromflat(flat=self.logits_stop+stop_shift)
        ac_stop = pd_stop.sample()

        ### 3.1: select first (active) node
        # rules: only select effective nodes
        self.logits_first = tf.keras.layers.Dense(args.emb_size, activation=tf.nn.relu, name='linear_select1')(emb_node)
        self.logits_first = tf.keras.layers.Dense(1, activation=None, name='linear_select2')(self.logits_first)
        self.logits_first = tf.squeeze(self.logits_first, axis=-1) # B*n
        logits_first_null = tf.ones(tf.shape(self.logits_first))*-1000
        self.logits_first = tf.where(condition=logits_first_mask,x=self.logits_first,y=logits_first_null)
        
        # using own prediction
        pd_first = CategoricalPdType(-1).pdfromflat(flat=self.logits_first)
        ac_first = pd_first.sample()
        mask = tf.one_hot(ac_first, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_first = tf.boolean_mask(emb_node, mask)
        emb_first = tf.expand_dims(emb_first,axis=1)
        
        # using groud truth action
        ac_first_real = self.ac_real[:, 0]
        mask_real = tf.one_hot(ac_first_real, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_first_real = tf.boolean_mask(emb_node, mask_real)
        emb_first_real = tf.expand_dims(emb_first_real, axis=1)

        ### 3.2: select second node
        # rules: do not select first node
        
        # using own prediction
        emb_cat = tf.concat([tf.tile(emb_first,[1,tf.shape(emb_node)[1],1]),emb_node],axis=2)
        self.logits_second = tf.keras.layers.Dense(args.emb_size, activation=tf.nn.relu, name='logits_second1')(emb_cat)
        self.logits_second = tf.keras.layers.Dense(1, activation=None, name='logits_second2')(self.logits_second)
        self.logits_second = tf.squeeze(self.logits_second, axis=-1)
        ac_first_mask = tf.one_hot(ac_first, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=False, off_value=True)
        logits_second_mask = tf.logical_and(logits_mask,ac_first_mask)
        logits_second_null = tf.ones(tf.shape(self.logits_second)) * -1000
        self.logits_second = tf.where(condition=logits_second_mask, x=self.logits_second, y=logits_second_null)

        pd_second = CategoricalPdType(-1).pdfromflat(flat=self.logits_second)
        ac_second = pd_second.sample()
        mask = tf.one_hot(ac_second, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_second = tf.boolean_mask(emb_node, mask)
        emb_second = tf.expand_dims(emb_second, axis=1)

        # using groudtruth
        emb_cat = tf.concat([tf.tile(emb_first_real, [1, tf.shape(emb_node)[1], 1]), emb_node], axis=2)
        self.logits_second_real = tf.keras.layers.Dense(args.emb_size, activation=tf.nn.relu, name='logits_second1')(emb_cat)
        self.logits_second_real = tf.keras.layers.Dense(1, activation=None, name='logits_second2')(self.logits_second_real)
        self.logits_second_real = tf.squeeze(self.logits_second_real, axis=-1)
        ac_first_mask_real = tf.one_hot(ac_first_real, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=False, off_value=True)
        logits_second_mask_real = tf.logical_and(logits_mask,ac_first_mask_real)
        self.logits_second_real = tf.where(condition=logits_second_mask_real, x=self.logits_second_real, y=logits_second_null)
        ac_second_real = self.ac_real[:,1]
        mask_real = tf.one_hot(ac_second_real, depth=tf.shape(emb_node)[1], dtype=tf.bool, on_value=True, off_value=False)
        emb_second_real = tf.boolean_mask(emb_node, mask_real)
        emb_second_real = tf.expand_dims(emb_second_real, axis=1)

        ### 3.3 predict edge type
        # using own prediction
        emb_cat = tf.concat([emb_first,emb_second],axis=-1)
        self.logits_edge = tf.keras.layers.Dense(args.emb_size, activation=tf.nn.relu, name='logits_edge1')(emb_cat)
        self.logits_edge = tf.keras.layers.Dense(ob['adj'].get_shape()[1], activation=None, name='logits_edge2')(self.logits_edge)
        self.logits_edge = tf.squeeze(self.logits_edge,axis=1)
        pd_edge = CategoricalPdType(-1).pdfromflat(self.logits_edge)
        ac_edge = pd_edge.sample()

        # using ground truth
        emb_cat = tf.concat([emb_first_real, emb_second_real], axis=-1)
        self.logits_edge_real = tf.keras.layers.Dense(args.emb_size, activation=tf.nn.relu, name='logits_edge1')(emb_cat)
        self.logits_edge_real = tf.keras.layers.Dense(ob['adj'].get_shape()[1], activation=None, name='logits_edge2')(self.logits_edge_real)
        self.logits_edge_real = tf.squeeze(self.logits_edge_real, axis=1)
        self.pd = self.pdtype(-1).pdfromflat([self.logits_first,self.logits_second_real,self.logits_edge_real,self.logits_stop])
        self.vpred = tf.keras.layers.Dense(args.emb_size, use_bias=False, activation=tf.nn.relu, name='value1')(emb_node)
        if args.bn==1:
            self.vpred = tf.keras.layers.BatchNormalization(axis=-1)(self.vpred)
        self.vpred = tf.reduce_max(self.vpred, axis=1)
        self.vpred = tf.keras.layers.Dense(1, activation=None, name='value2')(self.vpred)
        self.state_in = []
        self.state_out = []
        self.ac = tf.concat((tf.expand_dims(ac_first,axis=1),tf.expand_dims(ac_second,axis=1),tf.expand_dims(ac_edge,axis=1),tf.expand_dims(ac_stop,axis=1)),axis=1)

        debug = {}
        debug['ob_node'] = tf.shape(ob['node'])
        debug['ob_adj'] = tf.shape(ob['adj'])
        debug['emb_node'] = emb_node
        debug['logits_stop'] = self.logits_stop
        debug['logits_second'] = self.logits_second
        debug['ob_len'] = ob_len
        debug['logits_first_mask'] = logits_first_mask
        debug['logits_second_mask'] = logits_second_mask
        debug['ac'] = self.ac

        stochastic = tf.compat.v1.placeholder(dtype=tf.bool, shape=())
        self._act = U.function([stochastic, ob['adj'], ob['node']], [self.ac, self.vpred, debug]) # add debug in second arg if needed

    def act(self, stochastic, ob):
        return self._act(stochastic, ob['adj'][None], ob['node'][None])
    
    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)
    
    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    
    def get_initial_state(self):
        return []

def GCN(adj, node_feature, out_channels, is_act=True, is_normalize=False, name='gcn_simple'):
    '''
    state s: (adj,node_feature)
    :param adj: b*n*n
    :param node_feature: 1*n*d
    :param out_channels: scalar
    :param name:
    :return:
    '''
    edge_dim = adj.get_shape()[0]
    in_channels = node_feature.get_shape()[-1]
    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
        W = tf.compat.v1.get_variable("W", [edge_dim, in_channels, out_channels])
        b = tf.compat.v1.get_variable("b", [edge_dim, 1, out_channels])
        node_embedding = adj@tf.tile(node_feature,[edge_dim,1,1])@W+b
        if is_act:
            node_embedding = tf.nn.relu(node_embedding)
        node_embedding = tf.reduce_mean(node_embedding,axis=0,keepdims=True) # mean pooling
        if is_normalize:
            node_embedding = tf.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

def GCN_batch(adj, node_feature, out_channels, is_act=True, is_normalize=False, name='gcn_simple',aggregate='sum'):
    '''
    state s: (adj,node_feature)
    :param adj: none*b*n*n
    :param node_feature: none*1*n*d
    :param out_channels: scalar
    :param name:
    :return:
    '''
    edge_dim = adj.get_shape()[1]
    batch_size = tf.shape(adj)[0]
    in_channels = node_feature.get_shape()[-1]

    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
        W = tf.compat.v1.get_variable("W", [1, edge_dim, in_channels, out_channels],initializer=tf.keras.initializers.GlorotUniform())
        b = tf.compat.v1.get_variable("b", [1, edge_dim, 1, out_channels])
        node_embedding = adj@tf.tile(node_feature,[1,edge_dim,1,1])@tf.tile(W,[batch_size,1,1,1]) 
        if is_act:
            node_embedding = tf.nn.relu(node_embedding)
        if aggregate == 'sum':
            node_embedding = tf.reduce_sum(node_embedding, axis=1, keepdims=True)  # mean pooling
        elif aggregate=='mean':
            node_embedding = tf.reduce_mean(node_embedding,axis=1,keepdims=True) # mean pooling
        elif aggregate=='concat':
            node_embedding = tf.concat(tf.split(node_embedding,axis=1,num_or_size_splits=edge_dim),axis=3)
        else:
            print('GCN aggregate error!')
        if is_normalize:
            node_embedding = tf.nn.l2_normalize(node_embedding,axis=-1)
        return node_embedding

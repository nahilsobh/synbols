from haven import haven_utils as hu

# Define exp groups for parameter search
EXP_GROUPS = {'fewshot_char_protonet':
                hu.cartesian_exp_group({
                    'benchmark':'fewshot',
                    'lr':[0.1],
                    'batch_size':[1],
                    'model': "protonet",
                    'backbone': "resnet18",
                    'max_epoch': 100,
                    'imagenet_pretraining': False,
                    'episodic': True,
                    'dataset': {'path':'/mnt/datasets/public/research/synbols/plain_n=1000000.npz',
                                'name': 'fewshot_synbols',
                                'task': 'char',
                                # start 5-way 5-shot 15-query
                                'nclasses_train': 5, 
                                'nclasses_val': 5,
                                'nclasses_test': 5,
                                'support_size_train': 5,
                                'support_size_val': 5,
                                'support_size_test': 5,
                                'query_size_train': 15,
                                'query_size_val': 15,
                                'query_size_test': 15,
                                # end 5-way 5-shot 15-query
                                'train_iters': 50,
                                'val_iters': 50,
                                'test_iters': 50}}),
                'fewshot_char_maml':hu.cartesian_exp_group({
                    'benchmark':'fewshot',
                    'lr':[0.01],
                    'batch_size':[1],
                    'model': "MAML",
                    'backbone': "conv",
                    'max_epoch': 100,
                    'imagenet_pretraining': False,
                    'episodic': True,
                    'dataset': {'path':'/mnt/datasets/public/research/synbols/plain_n=1000000.npz',
                                'name': 'fewshot_synbols',
                                'task': 'char',
                                # start 5-way 5-shot 15-query
                                'nclasses_train': 5, 
                                'nclasses_val': 5,
                                'nclasses_test': 5,
                                'support_size_train': 5,
                                'support_size_val': 5,
                                'support_size_test': 5,
                                'query_size_train': 15,
                                'query_size_val': 15,
                                'query_size_test': 15,
                                # end 5-way 5-shot 15-query
                                'train_iters': 50,
                                'val_iters': 50,
                                'test_iters': 50}})}
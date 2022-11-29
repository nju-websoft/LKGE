def config(args):
    '''
    Hyperparameters for all models and datasets.
    '''

    '''base model'''
    args.learning_rate = 0.0001
    args.emb_dim = 200
    args.batch_size = 2048

    '''other parameters'''
    if args.dataset == 'ENTITY':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.1
        elif args.lifelong_name == 'SI':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 0.1
    elif args.dataset == 'RELATION':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.1
        elif args.lifelong_name == 'SI':
            args.regular_weight = 1.0
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 0.1
    elif args.dataset == 'FACT':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'SI':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 1.0
    elif args.dataset == 'HYBRID':
        if args.lifelong_name == 'EWC':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'SI':
            args.regular_weight = 0.01
        elif args.lifelong_name == 'LKGE':
            args.regular_weight = 0.01
            args.reconstruct_weight = 0.1




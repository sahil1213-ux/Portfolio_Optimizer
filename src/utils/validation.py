

def validate_arguments(args):
    if args.max_weight <= 0 or args.max_weight > 1:
        raise ValueError("max_weight must be between 0 and 1.")
    if args.amount <= 0:
        raise ValueError("Investment amount must be greater than 0.")
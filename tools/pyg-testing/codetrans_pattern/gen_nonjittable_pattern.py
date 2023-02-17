from tools.codetrans import Pattern, TransferMethod, to_json

perf_patterns = []

pattern_args = Pattern(
    name = "args_pattern",
    method = TransferMethod.Begin,
    old_str = "for epoch in range",
    new_str = '''
import argparse
parser = argparse.ArgumentParser(description=__file__ + " Testing")

parser.add_argument('--warmup', default=5, type=int, 
                    help='Number of warmup iteration')

parser.add_argument('--iter-time', default=1, type=int, 
                    help='Number of perf iteration')
                    
parser.add_argument('--gpu-profile', default=False, type=bool, 
                    help='Using cuda profile')
args = parser.parse_args()

'''
)
perf_patterns.append(pattern_args)

pattern_warm_up = Pattern(
    name = "pattern_warm_up",
    method = TransferMethod.Rewrite,
    old_str = "for epoch in range",
    new_str = '''
warm_up_epoch = args.warmup
perf_epoch = args.iter_time
for epoch in range(1, warm_up_epoch + perf_epoch):    
'''
)
perf_patterns.append(pattern_warm_up)

pattern_profile_mid = Pattern(
    name = "pattern_profile_mid",
    method = TransferMethod.After,
    old_str = "for epoch in range",
    new_str = '''
    if(epoch == warm_up_epoch):
        start = time.time()
'''
)
perf_patterns.append(pattern_profile_mid)

pattern_profile_end = Pattern(
    name = "pattern_profile_end",
    method = TransferMethod.AfterBlock,
    old_str = "for epoch in range",
    new_str = '''
torch.cuda.synchronize()
end = time.time()
print((end - start) * 1000 / 1)
exit()
'''
)
perf_patterns.append(pattern_profile_end)


to_json(perf_patterns, "./nojittable.json")
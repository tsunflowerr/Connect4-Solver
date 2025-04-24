import sys, os, pickle, io, torch
import torchmetrics
from src.c4a0.nn import ConnectFourNet

# 1) Make sure Python can see your src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# 2) Monkey-patch storage loader so pickle.load never tries CUDA
_orig_load = torch.storage._load_from_bytes
def _cpu_load_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b),
                      map_location=torch.device('cpu'),
                      weights_only=False)
torch.storage._load_from_bytes = _cpu_load_from_bytes

# 3) Unpickle your model (won’t error on CUDA)
model_path = '/home/dnanper/UNI/Connect4-Solver/training/2025-04-23T16:42:39.048358/model.pkl'
with open(model_path, 'rb') as f:
    model: ConnectFourNet = pickle.load(f)

# 4) Restore the original loader (optional)
torch.storage._load_from_bytes = _orig_load

# # 5) Replace metrics with fresh CPU ones
model.policy_kl_div = torchmetrics.KLDivergence(log_prob=True).to('cpu')
model.value_mse    = torchmetrics.MeanSquaredError().to('cpu')

# 6) Finally move the rest of the model to CPU & eval
model = model.to(torch.device('cpu'))
model.eval()

print(model)

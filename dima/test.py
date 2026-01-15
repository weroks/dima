import torch
from dima.diffusion.dima import DiMAModel
from dima import get_config_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiMAModel(config_path=get_config_path(), device=device)
model.load_pretrained()

sequences = model.generate_samples(num_texts=10)

print(sequences)
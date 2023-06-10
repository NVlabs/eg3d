import torch

def sample_cross_section(G, ws, resolution=256, w=1.2):
    axis=0
    A, B = torch.meshgrid(torch.linspace(w/2, -w/2, resolution, device=ws.device), torch.linspace(-w/2, w/2, resolution, device=ws.device), indexing='ij')
    A, B = A.reshape(-1, 1), B.reshape(-1, 1)
    C = torch.zeros_like(A)
    coordinates = [A, B]
    coordinates.insert(axis, C)
    coordinates = torch.cat(coordinates, dim=-1).expand(ws.shape[0], -1, -1)

    sigma = G.sample_mixed(coordinates, torch.randn_like(coordinates), ws)['sigma']
    return sigma.reshape(-1, 1, resolution, resolution)

# if __name__ == '__main__':
#     sample_crossection(None)
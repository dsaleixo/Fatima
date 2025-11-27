from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import os
from pathlib import Path
import pandas as pd
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser, MMCIFParser


ATOM_CHANNELS = ["C", "N", "O", "S", "H", "OTHER"]

def _is_pdb_id(s: str) -> bool:
    # simples heurística: 4 caracteres alfanuméricos sem caminho -> assume PDB id
    return len(s) == 4 and os.path.sep not in s

def _load_atoms_from_pdb(input_pdb: str):
    """
    Retorna lista de (atom_name, element, x,y,z)
    aceita PDB ID (ex: '1abc') ou caminho para .pdb/.cif
    """
    if _is_pdb_id(input_pdb):
        ppdb = PandasPdb().fetch_pdb(input_pdb)
        df = ppdb.df['ATOM']
        atoms = []
        for _, r in df.iterrows():
            elem = str(r['element_symbol']).strip() if 'element_symbol' in df.columns else r['atom_name'][0]
            atoms.append((r['atom_name'].strip(), elem.strip().upper(), float(r['x_coord']), float(r['y_coord']), float(r['z_coord'])))
        return atoms
    else:
        # tenta PDBParser, se falhar tenta MMCIFParser
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('prot', input_pdb)
        except Exception:
            parser2 = MMCIFParser(QUIET=True)
            structure = parser2.get_structure('prot', input_pdb)
        atoms = []
        for atom in structure.get_atoms():
            name = atom.get_name()
            # elemento, fallback para primeira letra do nome
            element = (atom.element.strip().upper() if hasattr(atom, "element") and atom.element else name[0]).upper()
            coord = atom.get_coord()
            atoms.append((name, element, float(coord[0]), float(coord[1]), float(coord[2])))
        return atoms

def voxelize_pdb(
    input_pdb: str,
    grid_size: int = 64,
    voxel_size: float = 1.0,
    channels: Optional[List[str]] = None,
    sigma: float = 0.8,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str,int]]:
    """
    Voxeliza uma proteína em uma grade 3D.

    Args:
        input_pdb: PDB id (ex: "1abc") ou caminho para arquivo PDB/CIF.
        grid_size: número de voxels por dimensão (grid cúbico: grid_size^3).
        voxel_size: tamanho do voxel em Å (resolução).
        channels: lista de símbolos de elemento p/ canais (ex: ["C","N","O","S","H","OTHER"]) ou None usa padrão.
        sigma: desvio padrão (em Å) da gaussiana para espalhar cada átomo; se 0 -> marca 1 no voxel central.
        normalize: se True, divide cada canal pelo seu máximo (0-1).
    
    Returns:
        (grid, channel_map)
        - grid: np.ndarray shape (C, grid_size, grid_size, grid_size)
        - channel_map: dict que mapeia elemento -> canal index
    """
    if channels is None:
        channels = ATOM_CHANNELS.copy()
    atoms = _load_atoms_from_pdb(input_pdb)
    if len(atoms) == 0:
        raise ValueError("Nenhum átomo lido do PDB.")

    coords = np.array([[x,y,z] for (_,_,x,y,z) in atoms], dtype=np.float32)
    elems = [elem for (_,elem,_,_,_) in atoms]

    # centraliza na média dos átomos (pode trocar para centro de massa/resíduo)
    center = coords.mean(axis=0)
    coords_centered = coords - center

    half_grid_angstrom = (grid_size * voxel_size) / 2.0
    # origem (canto mínimo) em coordenadas centradas
    origin = -half_grid_angstrom

    # mapeamento de elementos para canais
    ch_map = {}
    for i, ch in enumerate(channels):
        ch_map[ch] = i
    # elementos desconhecidos vão para "OTHER" se existir, senão último canal
    other_idx = ch_map.get("OTHER", max(ch_map.values()))

    grid = np.zeros((len(channels), grid_size, grid_size, grid_size), dtype=np.float32)

    # Pré-calcula janela em voxels para 3*sigma
    if sigma > 0:
        radius_vox = int(np.ceil((3.0 * sigma) / voxel_size))
        # criar kernel gaussiano em voxels
        # coordenadas de kernel
        rng = np.arange(-radius_vox, radius_vox+1)
        xx, yy, zz = np.meshgrid(rng, rng, rng, indexing='ij')
        d2 = (xx*voxel_size)**2 + (yy*voxel_size)**2 + (zz*voxel_size)**2
        kernel = np.exp(-0.5 * d2 / (sigma**2))
    else:
        radius_vox = 0
        kernel = None

    # processa cada átomo
    for (elem, c) in zip(elems, coords_centered):
        # escolhe canal
        elem_key = elem.upper()
        if elem_key not in ch_map:
            chan = other_idx
        else:
            chan = ch_map[elem_key]

        # converte coordenada (Å) -> índice de voxel
        idx_f = (c - origin) / voxel_size  # posição em voxels (float)
        ix, iy, iz = np.round(idx_f).astype(int)

        # se sigma == 0, apenas incrementa voxel central (se dentro do grid)
        if sigma == 0:
            if 0 <= ix < grid_size and 0 <= iy < grid_size and 0 <= iz < grid_size:
                grid[chan, ix, iy, iz] += 1.0
            continue

        # aplica kernel: adicionar em vizinhança
        # limites na grade
        x0 = ix - radius_vox
        x1 = ix + radius_vox
        y0 = iy - radius_vox
        y1 = iy + radius_vox
        z0 = iz - radius_vox
        z1 = iz + radius_vox

        kx0 = 0
        ky0 = 0
        kz0 = 0
        kx1 = kernel.shape[0]
        ky1 = kernel.shape[1]
        kz1 = kernel.shape[2]

        # ajustar se cruzar borda
        if x0 < 0:
            kx0 = -x0
            x0 = 0
        if y0 < 0:
            ky0 = -y0
            y0 = 0
        if z0 < 0:
            kz0 = -z0
            z0 = 0
        if x1 >= grid_size:
            kx1 = kernel.shape[0] - (x1 - (grid_size-1))
            x1 = grid_size-1
        if y1 >= grid_size:
            ky1 = kernel.shape[1] - (y1 - (grid_size-1))
            y1 = grid_size-1
        if z1 >= grid_size:
            kz1 = kernel.shape[2] - (z1 - (grid_size-1))
            z1 = grid_size-1

        # fatia kernel e grade e adiciona
        # +1 porque x1,y1,z1 são índices inclusive
        grid_slice = (slice(None), slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1+1))
        k_slice = (slice(kx0, kx1), slice(ky0, ky1), slice(kz0, kz1))
        # atenção: kernel shape é (K,K,K)
        grid[chan, x0:x1+1, y0:y1+1, z0:z1+1] += kernel[k_slice]

    # normalização opcional por canal
    if normalize:
        for ch in range(grid.shape[0]):
            mx = grid[ch].max()
            if mx > 0:
                grid[ch] /= mx

    return grid, ch_map



def process_pdb_files(
    input_dir: str,
    output_dir: str,
    grid_size: int = 64,
    voxel_size: float = 1.0,
    sigma: float = 0.8,
    channels: Optional[List[str]] = None,
    normalize: bool = True,
    overwrite: bool = False
) -> None:
    """
    Processes all .pdb/.cif files in a directory and saves their voxel grids as .pt files.

    Args:
        input_dir: folder containing PDB/CIF files.
        output_dir: folder where .pt files will be saved.
        grid_size: voxel grid resolution.
        voxel_size: voxel size in Å.
        sigma: Gaussian smoothing strength.
        channels: list of channels (None = default).
        normalize: normalize channels to [0, 1].
        overwrite: if False, skip files that were already generated.
    """
    df = pd.read_csv(input_dir, sep="\t") 
     
    cont = 0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for idx, row in df.iterrows():

        cont+=1
        pdb_id = row['SAMPLE_PDB_ID']
        if cont >10:
            break
       

        name = row['SAMPLE_ID']
        out_file = output_path / f"{name}.pt"

        if out_file.exists() and not overwrite:
            print(f"[SKIP] {name} already exists.")
            continue

        try:
            print(f"Voxelizing {name}...")
           
            # voxelize
            grid_np, channel_map = voxelize_pdb(
                input_pdb=str(pdb_id),
                grid_size=grid_size,
                voxel_size=voxel_size,
                sigma=sigma,
                channels=channels,
                normalize=normalize
            )

            # convert to Torch tensor
            grid_tensor = torch.from_numpy(grid_np).float()

            # pack metadata
            data = {
                "grid": grid_tensor,   # (C, D, H, W)
                "channel_map": channel_map,
                "source_file": str(pdb_id),
                "grid_size": grid_size,
                "voxel_size": voxel_size,
                "sigma": sigma
            }

            torch.save(data, out_file)
            print(f"[OK] Saved: {out_file}")

        except Exception as e:
            print(f"[ERROR] Failed {name}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    process_pdb_files(
        input_dir="MdrDB/data/MdrDB_CoreSet_release_v1.0.2022.tsv",
        output_dir="datas/MdrDB_voxel_grids",
        grid_size=64,
        voxel_size=1.0,
        sigma=0.8,
        overwrite=False
    )
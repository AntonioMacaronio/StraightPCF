import os
import argparse
import torch
import numpy as np
from utils.misc import seed_all
from utils.transforms import NormalizeUnitSphere
from utils.denoise import patch_based_denoise
from models.straightpcf import StraightPCF

def apply_straightpcf_filter(
    input_pcl_np: np.ndarray,
    ckpt_path: str = "pretrained_straightpcf/ckpt_straightpcf.pt",
    device: str = 'cuda',
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
    num_iterations: int = 1,
) -> np.ndarray:
    """
    Applies StraightPCF filtering to a point cloud.

    Args:
        input_pcl_np: Input point cloud as a NumPy array (N, 3).
        ckpt_path: Path to the StraightPCF model checkpoint (.pt file).
        device: PyTorch device ('cuda' or 'cpu').
        patch_size: Size of patches for patch-based denoising.
        seed_k: Parameter for patch seeding.
        seed_k_alpha: Parameter for patch seeding adjustment.
        num_iterations: Number of denoising iterations to apply.

    Returns:
        Filtered point cloud as a NumPy array (N, 3).
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Load Model
    ckpt = torch.load(ckpt_path, map_location=device)
    # Ensure the args from checkpoint are used for model initialization
    # We might need to merge/override some args like device
    model_args = ckpt['args']
    model_args.device = device # Override device

    # If the checkpoint args don't have necessary parameters defined in StraightPCF.__init__
    # provide defaults or load them from elsewhere (e.g., CVM ckpt if needed).
    # For simplicity, assume ckpt['args'] contains necessary info for now.
    # Example: ensure required args like 'frame_knn', 'tot_its', 'num_train_points', etc. exist
    # This might need adjustment based on actual checkpoint contents.
    required_args = ['frame_knn', 'tot_its', 'num_train_points', 'dsm_sigma',
                     'feat_embedding_dim', 'distance_estimation', 'decoder_hidden_dim']
    for arg_name in required_args:
        if not hasattr(model_args, arg_name):
            # Provide sensible defaults or raise error if critical
            # These defaults are placeholders and might need tuning
            defaults = {'frame_knn': 16, 'tot_its': 3, 'num_train_points': 512,
                        'dsm_sigma': 0.01, 'feat_embedding_dim': 128,
                        'distance_estimation': False, 'decoder_hidden_dim': 128}
            if arg_name in defaults:
                print(f"Warning: Argument '{arg_name}' not found in checkpoint args. Using default value: {defaults[arg_name]}")
                setattr(model_args, arg_name, defaults[arg_name])
            else:
                # If cvm_ckpt path is needed but not in args, try adding default path
                if arg_name == 'cvm_ckpt' and not hasattr(model_args, 'cvm_ckpt'):
                     setattr(model_args, 'cvm_ckpt', './pretrained_cvm/ckpt_cvm.pt')
                     print(f"Warning: Argument 'cvm_ckpt' not found. Using default: {model_args.cvm_ckpt}")
                     if not os.path.exists(model_args.cvm_ckpt):
                         raise FileNotFoundError(f"Default CVM checkpoint not found: {model_args.cvm_ckpt}")
                else:
                    raise ValueError(f"Critical argument '{arg_name}' missing from checkpoint args and no default provided.")


    model = StraightPCF(args=model_args).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Prepare Input
    pcl_noisy_tensor = torch.FloatTensor(input_pcl_np).to(device)

    # Normalize
    pcl_normalized, center, scale = NormalizeUnitSphere.normalize(pcl_noisy_tensor)

    # Denoise
    with torch.no_grad():
        pcl_denoised_normalized = pcl_normalized
        for _ in range(num_iterations):
            pcl_denoised_normalized = patch_based_denoise(
                model=model,
                pcl_noisy=pcl_denoised_normalized,
                seed_k=seed_k,
                seed_k_alpha=seed_k_alpha,
                patch_size=patch_size,
            )

    # Denormalize
    pcl_denoised = pcl_denoised_normalized * scale + center

    # Convert to NumPy
    output_pcl_np = pcl_denoised.cpu().numpy()

    return output_pcl_np

def get_straightpcf_model(
    ckpt_path: str = "pretrained_straightpcf/ckpt_straightpcf.pt",
    device: str = 'cuda',
) -> StraightPCF:
    """
    Loads the StraightPCF model from a checkpoint and sets it to evaluation mode.

    Args:
        ckpt_path: Path to the StraightPCF model checkpoint (.pt file).
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        Loaded StraightPCF model in evaluation mode.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Load Model
    ckpt = torch.load(ckpt_path, map_location=device)
    # Ensure the args from checkpoint are used for model initialization
    # We might need to merge/override some args like device
    model_args = ckpt['args']
    model_args.device = device # Override device

    # If the checkpoint args don't have necessary parameters defined in StraightPCF.__init__
    # provide defaults or load them from elsewhere (e.g., CVM ckpt if needed).
    # For simplicity, assume ckpt['args'] contains necessary info for now.
    # Example: ensure required args like 'frame_knn', 'tot_its', 'num_train_points', etc. exist
    # This might need adjustment based on actual checkpoint contents.
    required_args = ['frame_knn', 'tot_its', 'num_train_points', 'dsm_sigma',
                     'feat_embedding_dim', 'distance_estimation', 'decoder_hidden_dim']
    for arg_name in required_args:
        if not hasattr(model_args, arg_name):
            # Provide sensible defaults or raise error if critical
            # These defaults are placeholders and might need tuning
            defaults = {'frame_knn': 16, 'tot_its': 3, 'num_train_points': 512,
                        'dsm_sigma': 0.01, 'feat_embedding_dim': 128,
                        'distance_estimation': False, 'decoder_hidden_dim': 128}
            if arg_name in defaults:
                print(f"Warning: Argument '{arg_name}' not found in checkpoint args. Using default value: {defaults[arg_name]}")
                setattr(model_args, arg_name, defaults[arg_name])
            else:
                # If cvm_ckpt path is needed but not in args, try adding default path
                if arg_name == 'cvm_ckpt' and not hasattr(model_args, 'cvm_ckpt'):
                     setattr(model_args, 'cvm_ckpt', './pretrained_cvm/ckpt_cvm.pt')
                     print(f"Warning: Argument 'cvm_ckpt' not found. Using default: {model_args.cvm_ckpt}")
                     if not os.path.exists(model_args.cvm_ckpt):
                         raise FileNotFoundError(f"Default CVM checkpoint not found: {model_args.cvm_ckpt}")
                else:
                    raise ValueError(f"Critical argument '{arg_name}' missing from checkpoint args and no default provided.")

    model = StraightPCF(args=model_args).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def apply_straightpcf_filter_with_model(
    model: StraightPCF,
    input_pcl_np: np.ndarray,
    patch_size: int = 1000,
    seed_k: int = 6,
    seed_k_alpha: int = 1,
    num_iterations: int = 1,
) -> np.ndarray:
    """
    Applies StraightPCF filtering to a point cloud using a pre-loaded StraightPCF model.

    Args:
        model: Pre-loaded StraightPCF model instance (with state_dict loaded and on correct device).
        input_pcl_np: Input point cloud as a NumPy array (N, 3).
        patch_size: Size of patches for patch-based denoising.
        seed_k: Parameter for patch seeding.
        seed_k_alpha: Parameter for patch seeding adjustment.
        num_iterations: Number of denoising iterations to apply.

    Returns:
        Filtered point cloud as a NumPy array (N, 3).
    """
    model.eval()
    device = next(model.parameters()).device

    # Prepare Input
    pcl_noisy_tensor = torch.FloatTensor(input_pcl_np).to(device)

    # Normalize
    pcl_normalized, center, scale = NormalizeUnitSphere.normalize(pcl_noisy_tensor)

    # Denoise
    with torch.no_grad():
        pcl_denoised_normalized = pcl_normalized
        for _ in range(num_iterations):
            pcl_denoised_normalized = patch_based_denoise(
                model=model,
                pcl_noisy=pcl_denoised_normalized,
                seed_k=seed_k,
                seed_k_alpha=seed_k_alpha,
                patch_size=patch_size,
            )

    # Denormalize
    pcl_denoised = pcl_denoised_normalized * scale + center

    output_pcl_np = pcl_denoised.cpu().numpy()
    return output_pcl_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply StraightPCF filtering to a point cloud file.")
    parser.add_argument("input_file", type=str, help="Path to the input point cloud file (.xyz or .npy)")
    parser.add_argument("output_file", type=str, help="Path to save the filtered point cloud file (.xyz or .npy)")
    parser.add_argument("--ckpt", type=str, default='./pretrained_straightpcf/ckpt_straightpcf.pt', help="Path to the model checkpoint file.")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--patch_size", type=int, default=1000, help="Patch size for denoising.")
    parser.add_argument("--seed_k", type=int, default=6, help="Seed k parameter.")
    parser.add_argument("--seed_k_alpha", type=int, default=1, help="Seed k alpha parameter.")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of denoising iterations.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")

    args = parser.parse_args()

    seed_all(args.seed)

    # Load input point cloud
    print(f"Loading point cloud from: {args.input_file}")
    if args.input_file.endswith('.xyz'):
        input_pcl = np.loadtxt(args.input_file)
    elif args.input_file.endswith('.npy'):
        input_pcl = np.load(args.input_file)
    else:
        raise ValueError("Input file must be .xyz or .npy")

    print(f"Input point cloud shape: {input_pcl.shape}")
    if input_pcl.shape[1] != 3:
        raise ValueError(f"Input point cloud must have 3 columns (X, Y, Z), but got {input_pcl.shape[1]}")

    # Apply filtering
    print("Applying StraightPCF filtering...")
    filtered_pcl = apply_straightpcf_filter(
        input_pcl_np=input_pcl,
        ckpt_path=args.ckpt,
        device=args.device,
        patch_size=args.patch_size,
        seed_k=args.seed_k,
        seed_k_alpha=args.seed_k_alpha,
        num_iterations=args.num_iterations
    )
    print(f"Filtered point cloud shape: {filtered_pcl.shape}")

    # Save output point cloud
    print(f"Saving filtered point cloud to: {args.output_file}")
    if args.output_file.endswith('.xyz'):
        np.savetxt(args.output_file, filtered_pcl, fmt='%.8f')
    elif args.output_file.endswith('.npy'):
        np.save(args.output_file, filtered_pcl)
    else:
        raise ValueError("Output file must be .xyz or .npy")

    print("Filtering complete.") 
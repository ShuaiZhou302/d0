# Dataset Factory
# Simple factory to create different types of datasets

import random
from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset


def create_dataset(config: OmegaConf, val: bool = False):
    """Create dataset based on config, including weighted multi-dataset training."""
    dataset_type = config.dataset.get('type', 'robotwin')  # Default to robotwin

    if dataset_type == 'multi':
        return _create_multi_dataset(config, val=val)

    return _create_single_dataset(config, val=val)


class MultiDataset(Dataset):
    """Weighted wrapper that pads heterogeneous action/state dimensions."""

    def __init__(
        self,
        datasets: List[Dataset],
        names: List[str],
        weights: List[float],
        target_action_dim: int,
        target_state_dim: int,
    ):
        if not datasets:
            raise ValueError("MultiDataset requires at least one child dataset")
        if len(datasets) != len(names) or len(datasets) != len(weights):
            raise ValueError("datasets, names, and weights must have the same length")
        if any(weight <= 0 for weight in weights):
            raise ValueError(f"All dataset weights must be > 0, got {weights}")

        self.datasets = datasets
        self.names = names
        self.weights = [float(weight) for weight in weights]
        self.target_action_dim = int(target_action_dim)
        self.target_state_dim = int(target_state_dim)
        self.lengths = [len(dataset) for dataset in datasets]
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"All child datasets must be non-empty, got lengths={self.lengths}")

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        # Child datasets already use stochastic episode/window sampling, so idx only drives epoch length.
        child_idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        child_dataset = self.datasets[child_idx]
        child_sample_idx = random.randrange(self.lengths[child_idx])
        sample = child_dataset[child_sample_idx]
        if sample is None:
            return None

        sample = dict(sample)
        sample['dataset_name'] = self.names[child_idx]

        if sample.get('action_sequence') is not None:
            actions = sample['action_sequence']
            sample['action_sequence'] = _pad_last_dim(actions, self.target_action_dim, 'action_sequence')
            action_mask = torch.zeros_like(sample['action_sequence'], dtype=torch.bool)
            action_mask[..., :actions.shape[-1]] = True
            sample['action_mask'] = action_mask

        if sample.get('initial_state') is not None:
            sample['initial_state'] = _pad_last_dim(sample['initial_state'], self.target_state_dim, 'initial_state')

        return sample


def _pad_last_dim(tensor: torch.Tensor, target_dim: int, name: str) -> torch.Tensor:
    """Pad a tensor's last dimension with zeros up to target_dim."""
    current_dim = tensor.shape[-1]
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        raise ValueError(f"{name} dim {current_dim} exceeds target dim {target_dim}")

    padded_shape = list(tensor.shape)
    padded_shape[-1] = target_dim
    padded = tensor.new_zeros(padded_shape)
    padded[..., :current_dim] = tensor
    return padded


def _create_multi_dataset(config: OmegaConf, val: bool = False) -> MultiDataset:
    if not hasattr(config.dataset, 'datasets') or len(config.dataset.datasets) == 0:
        raise ValueError("dataset.type='multi' requires a non-empty dataset.datasets list")

    target_action_dim = int(config.dataset.get('target_action_dim', config.common.action_dim))
    target_state_dim = int(config.dataset.get('target_state_dim', config.common.state_dim))

    datasets = []
    names = []
    weights = []
    for i, child_dataset_config in enumerate(config.dataset.datasets):
        child_config = _build_child_config(config, child_dataset_config)
        child_name = child_dataset_config.get('name', child_config.dataset.get('type', f'dataset_{i}'))
        child_weight = float(child_dataset_config.get('weight', 1.0))

        datasets.append(_create_single_dataset(child_config, val=val))
        names.append(str(child_name))
        weights.append(child_weight)

    return MultiDataset(
        datasets=datasets,
        names=names,
        weights=weights,
        target_action_dim=target_action_dim,
        target_state_dim=target_state_dim,
    )


def _build_child_config(config: OmegaConf, child_dataset_config: OmegaConf) -> OmegaConf:
    """Build a normal single-dataset config from one dataset.datasets entry."""
    child_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    child_dataset_dict = OmegaConf.to_container(child_dataset_config, resolve=True)
    child_common = child_dataset_dict.pop('common', None)
    child_dataset_dict.pop('name', None)
    child_dataset_dict.pop('weight', None)

    child_config.dataset = OmegaConf.create(child_dataset_dict)
    if child_common:
        child_config.common = OmegaConf.merge(child_config.common, child_common)
    return child_config


def _create_single_dataset(config: OmegaConf, val: bool = False):
    """
    Create dataset based on config.
    
    Args:
        config: Configuration object
        val: Whether to create validation dataset
        
    Returns:
        Dataset instance
    """
    dataset_type = config.dataset.get('type', 'robotwin')  # Default to robotwin
    
    if dataset_type == 'robotwin':
        from .robotwin2.robotwin_agilex_dataset import RobotWinTaskDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        if hasattr(config.dataset, 'randomized_limit_per_task'):
            params['randomized_limit_per_task'] = config.dataset.randomized_limit_per_task
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        if hasattr(config.dataset, 'use_language_action'):
            params['use_language_action'] = config.dataset.use_language_action
        params['val'] = val
        
        return RobotWinTaskDataset(**params)
    
    elif dataset_type == 'ac_one':
        from .ac_one.ac_one_dataset import ACOneDataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return ACOneDataset(**params)

    elif dataset_type == 'latent_action':
        from .latent_action.latent_action_dataset import LatentActionDataset

        params = {}

        # Common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        if hasattr(config.dataset, 'dataset_dir'):
            dataset_dir = list(config.dataset.dataset_dir)
            params['dataset_dir'] = [str(p) for p in dataset_dir]
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Optional VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Optional additional params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        params['val'] = val

        return LatentActionDataset(**params)

    elif dataset_type == 'egoverse_trimodal':
        from .egoverse.egoverse_vgm_dataset import EgoVerseTrimodalDataset

        params = {}
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'action_dim': config.common.action_dim,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        for key in ['train_manifest', 'val_manifest', 'manifest', 'max_samples', 'seed', 'action_mode']:
            if hasattr(config.dataset, key):
                params[key] = getattr(config.dataset, key)
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        if hasattr(config.dataset, 'params'):
            params.update(OmegaConf.to_object(config.dataset.params))
        params['val'] = val

        return EgoVerseTrimodalDataset(**params)

    elif dataset_type == 'aloha_agilex_2':
        from .aloha_agilex_2.aloha_agilex2_dataset import AlohaAgilex2Dataset
        
        # Get all parameters from config
        params = {}
        
        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })
        
        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'val_episodes'):
            params['val_episodes'] = config.dataset.val_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        
        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path
        
        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)
        
        # Set validation flag
        params['val'] = val
        
        return AlohaAgilex2Dataset(**params)

    elif dataset_type == 'lerobot':
        from .lerobot.lerobot_dataset import LeRobotMotusDataset

        # Get all parameters from config
        params = {}

        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val

        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        # Set validation flag
        params['val'] = val
        
        return LeRobotMotusDataset(**params)

    elif dataset_type == 'bridge':
        from .bridge.bridge_dataset import BridgeDataset

        params = {}

        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        if hasattr(config.dataset, 'normalize_actions'):
            params['normalize_actions'] = config.dataset.normalize_actions
        if hasattr(config.dataset, 'stats_path'):
            params['stats_path'] = config.dataset.stats_path
        if hasattr(config.dataset, 'stats_key'):
            params['stats_key'] = config.dataset.stats_key
        if hasattr(config.dataset, 'state_stats_signal'):
            params['state_stats_signal'] = config.dataset.state_stats_signal
        if hasattr(config.dataset, 'action_stats_signal'):
            params['action_stats_signal'] = config.dataset.action_stats_signal
        if hasattr(config.dataset, 'enable_setup_control_suffix'):
            params['enable_setup_control_suffix'] = config.dataset.enable_setup_control_suffix
        if hasattr(config.dataset, 'setup_text'):
            params['setup_text'] = config.dataset.setup_text

        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        # Set validation flag and optional language-action supervision
        if hasattr(config.dataset, 'use_language_action'):
            params['use_language_action'] = config.dataset.use_language_action
        params['val'] = val

        return BridgeDataset(**params)

    elif dataset_type == 'fractal_bridge':
        from .fractal.fractal_bridge_dataset import FractalDataset

        params = {}

        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        if hasattr(config.dataset, 'use_language_action'):
            params['use_language_action'] = config.dataset.use_language_action
        if hasattr(config.dataset, 'normalize_actions'):
            params['normalize_actions'] = config.dataset.normalize_actions
        if hasattr(config.dataset, 'stats_path'):
            params['stats_path'] = config.dataset.stats_path
        if hasattr(config.dataset, 'stats_key'):
            params['stats_key'] = config.dataset.stats_key
        if hasattr(config.dataset, 'state_stats_signal'):
            params['state_stats_signal'] = config.dataset.state_stats_signal
        if hasattr(config.dataset, 'action_stats_signal'):
            params['action_stats_signal'] = config.dataset.action_stats_signal
        if hasattr(config.dataset, 'enable_setup_control_suffix'):
            params['enable_setup_control_suffix'] = config.dataset.enable_setup_control_suffix
        if hasattr(config.dataset, 'setup_text'):
            params['setup_text'] = config.dataset.setup_text

        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        # Set validation flag
        params['val'] = val

        return FractalDataset(**params)

    elif dataset_type == 'droid' or dataset_type == 'droid_bridge':
        from .droid.droid_bridge_dataset import DroidDataset

        params = {}

        # Add common parameters
        if hasattr(config, 'common'):
            params.update({
                'global_downsample_rate': config.common.global_downsample_rate,
                'video_action_freq_ratio': config.common.video_action_freq_ratio,
                'num_video_frames': config.common.num_video_frames,
                'video_size': (config.common.video_height, config.common.video_width),
            })

        # Add dataset-specific parameters
        if hasattr(config.dataset, 'dataset_dir'):
            params['dataset_dir'] = config.dataset.dataset_dir
        if hasattr(config.dataset, 'data_mode'):
            params['data_mode'] = config.dataset.data_mode
        if hasattr(config.dataset, 'task_mode'):
            params['task_mode'] = config.dataset.task_mode
        if hasattr(config.dataset, 'task_name'):
            params['task_name'] = config.dataset.task_name
        if hasattr(config.dataset, 'max_episodes'):
            params['max_episodes'] = config.dataset.max_episodes
        if hasattr(config.dataset, 'image_aug'):
            params['image_aug'] = config.dataset.image_aug and not val  # No aug for validation
        if hasattr(config.dataset, 'use_language_action'):
            params['use_language_action'] = config.dataset.use_language_action
        if hasattr(config.dataset, 'normalize_actions'):
            params['normalize_actions'] = config.dataset.normalize_actions
        if hasattr(config.dataset, 'stats_path'):
            params['stats_path'] = config.dataset.stats_path
        if hasattr(config.dataset, 'stats_key'):
            params['stats_key'] = config.dataset.stats_key
        if hasattr(config.dataset, 'state_stats_signal'):
            params['state_stats_signal'] = config.dataset.state_stats_signal
        if hasattr(config.dataset, 'action_stats_signal'):
            params['action_stats_signal'] = config.dataset.action_stats_signal
        if hasattr(config.dataset, 'enable_setup_control_suffix'):
            params['enable_setup_control_suffix'] = config.dataset.enable_setup_control_suffix
        if hasattr(config.dataset, 'setup_text'):
            params['setup_text'] = config.dataset.setup_text

        # Add VLM checkpoint path
        if hasattr(config.model, 'vlm') and hasattr(config.model.vlm, 'checkpoint_path'):
            params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

        # Add any additional parameters from dataset.params
        if hasattr(config.dataset, 'params'):
            additional_params = OmegaConf.to_object(config.dataset.params)
            params.update(additional_params)

        # Set validation flag
        params['val'] = val

        return DroidDataset(**params)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available types: robotwin, bridge, fractal_bridge, droid, droid_bridge, ac_one, aloha_agilex_2, lerobot, latent_action")


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    # if any(vlm_input.get('labels') is not None for vlm_input in vlm_inputs):
    #     labels_list = [vlm_input.get('labels') for vlm_input in vlm_inputs]
    # else:
    #     labels_list = None

    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids_list, attention_mask_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
        else:
            padded_ids = ids
            padded_mask = mask
            # padded_labels = labels
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        # padded_labels.append(padded_labels)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
    }
def _process_vlm_inputs_batch_lap(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    # Extract components
    input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
    pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
    image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]
    attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
    if any(vlm_input.get('labels') is not None for vlm_input in vlm_inputs):
        labels_list = [vlm_input.get('labels') for vlm_input in vlm_inputs]
    else:
        labels_list = None
    if any(vlm_input.get('answer_start') is not None for vlm_input in vlm_inputs):
        answer_start_list = [vlm_input.get('answer_start') for vlm_input in vlm_inputs]
    else:
        answer_start_list = None
    # if any(vlm_input.get('language_action') is not None for vlm_input in vlm_inputs):
    #     language_action_list = [vlm_input.get('language_action') for vlm_input in vlm_inputs]
    # else:
    #     language_action_list = None
    
    # Pad input_ids to same length (simplified like model implementation)
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels_list = []

    for ids, mask, labels in zip(input_ids_list, attention_mask_list,labels_list):
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            # Pad input_ids
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            # Pad attention_mask
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None
            if labels is not None:
                labels_padding = torch.full((mask.shape[0], padding_size), -100, dtype=labels.dtype, device=labels.device)
                padded_labels = torch.cat([labels, labels_padding], dim=1)
            else:
                padded_labels = None
        else:
            padded_ids = ids
            padded_mask = mask
            padded_labels = labels
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        padded_labels_list.append(padded_labels)
    
    # Batch everything
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
        'labels': torch.cat(padded_labels_list, dim=0),
        'answer_start': torch.cat([answer_start for answer_start in answer_start_list if answer_start is not None], dim=0) if any(answer_start is not None for answer_start in answer_start_list) else None,
        # 'language_action': torch.cat([language_action for language_action in language_action_list if language_action is not None], dim=0) if any(language_action is not None for language_action in language_action_list) else None,
    }


def _process_language_embeddings_batch(language_embeddings: List[torch.Tensor], text_len: int = 512) -> torch.Tensor:
    """Process and batch language embeddings with padding."""
    padded_embeddings = []
    
    for emb in language_embeddings:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    
    # Stack to [B, seq_len, dim]
    return torch.stack(padded_embeddings, dim=0)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Universal collate function for all datasets.
    
    Args:
        batch: List of sample dictionaries (may contain None)
        
    Returns:
        Batched dictionary or None if all samples are None
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
    
    # Stack tensors（支持无 initial_state 的样本）
    first_frames = torch.stack([sample['first_frame'] for sample in batch])
    video_frames = torch.stack([sample['video_frames'] for sample in batch])
    has_actions = all(('action_sequence' in sample and sample['action_sequence'] is not None) for sample in batch)
    action_sequences = torch.stack([sample['action_sequence'] for sample in batch]) if has_actions else None
    has_action_mask = all(('action_mask' in sample and sample['action_mask'] is not None) for sample in batch)
    action_masks = torch.stack([sample['action_mask'] for sample in batch]) if has_action_mask else None
    has_initial_state = all(('initial_state' in sample and sample['initial_state'] is not None) for sample in batch)
    initial_states = torch.stack([sample['initial_state'] for sample in batch]) if has_initial_state else None
    dataset_names = [sample.get('dataset_name') for sample in batch]
    
    # Process VLM inputs with padding in collate_fn
    vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
    processed_vlm_inputs = None
    if vlm_inputs and all(vlm_input is not None for vlm_input in vlm_inputs):
        if any(vlm_input.get('labels') is not None for vlm_input in vlm_inputs):
            processed_vlm_inputs = _process_vlm_inputs_batch_lap(vlm_inputs)
        else:
            processed_vlm_inputs = _process_vlm_inputs_batch(vlm_inputs)
    
    # Process language embeddings with padding in collate_fn  
    language_embeddings = [sample.get('language_embedding') for sample in batch if 'language_embedding' in sample]
    processed_language_embeddings = None
    if language_embeddings and any(emb is not None for emb in language_embeddings):
        processed_language_embeddings = _process_language_embeddings_batch(language_embeddings)
    # print("labels:",processed_vlm_inputs['labels'].shape)
    # print("answer_start:",processed_vlm_inputs['answer_start'].shape)
    result = {
        'first_frame': first_frames,             # [B, C, H, W]
        'video_frames': video_frames,            # [B, F, C, H, W]
        'vlm_inputs': processed_vlm_inputs,
        'language_embedding': processed_language_embeddings,
    }

    if action_sequences is not None:
        result['action_sequence'] = action_sequences  # [B, F, D]
    if action_masks is not None:
        result['action_mask'] = action_masks
    if initial_states is not None:
        result['initial_state'] = initial_states
    if any(name is not None for name in dataset_names):
        result['dataset_name'] = dataset_names
    
    return result

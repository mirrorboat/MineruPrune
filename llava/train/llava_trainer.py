import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=['']):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def training_step(self, model, inputs):
        # if hasattr(self.data_collator, 'global_step'):
        #     # self.data_collator.global_step = self.state.global_step
        #     print(f"***old {self.data_collator.global_step}", flush=True)
        #     self.data_collator.set_global_step(self.state.global_step)
        #     print(f"***Setting global step {self.data_collator.global_step}", flush=True)
        # breakpoint()
        # inputs["pruned_steps"] = self.state.global_step
        # pruned_steps设置为0维tensor，其值是self.state.global_step
        inputs["pruned_steps"] = torch.tensor(self.state.global_step)
        # print(self.state.global_step)

        return super().training_step(model, inputs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        在评估前设置 model.set_eval(True)，在评估完成后恢复 model.set_eval(False)
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if eval_dataset is None:
            raise ValueError("Evaluation requires an `eval_dataset`.")

        model = self.model

        model.get_model().set_l0_module_eval(True)

        try:
            eval_results = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        finally:
            model.get_model().set_l0_module_eval(False)

        return eval_results


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     outputs = model(**inputs)
    #     loss = outputs.loss
    #     ce_loss = outputs.ce_loss
    #     lag_loss = outputs.lag_loss
    #     l0_output_1 = outputs.l0_output_1
    #     # l0_output_2 = outputs.l0_output_2
    #     l0_output_1_logs = {f'{name}': val.cpu().item() if torch.is_tensor(val) else val for name, val in l0_output_1.items()}
    #     # l0_output_2_logs = {f'{name}': val.cpu().item() if torch.is_tensor(val) else val for name, val in l0_output_2.items()}
    #     self.log({"TRAIN loss": loss.item(), "ce_loss": ce_loss.item(), "lag_loss": lag_loss.item(), **l0_output_1_logs})

    #     return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = None
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                if self.args.mm_vision_tower_lr is not None:
                    vision_tower_parameters = [
                        name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad and "l0_module" not in n and "model.layers.19" not in n and "model.layers.22" not in n )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        # {
                        #     "params": [
                        #         p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad and "l0_module" not in n)
                        #     ],
                        #     "weight_decay": self.args.weight_decay,
                        #     "lr": self.args.mm_vision_tower_lr,
                        # },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in vision_tower_parameters and p.requires_grad and "l0_module" not in n and "model.layers.19" not in n and "model.layers.22" not in n)
                            ],
                            "weight_decay": 0.0,
                        },
                        # {
                        #     "params": [
                        #         p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n in vision_tower_parameters and p.requires_grad and "l0_module" not in n)
                        #     ],
                        #     "weight_decay": 0.0,
                        #     "lr": self.args.mm_vision_tower_lr,
                        # },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad and "l0_module" not in n)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad and "l0_module" not in n)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad and "l0_module" not in n)
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad and "l0_module" not in n)
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad and "l0_module" not in n)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad and "l0_module" not in n)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "l0_module" not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "l0_module" not in n)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            if optimizer_grouped_parameters is None:
                optimizer_grouped_parameters = []
            l0_module_params = [p for n, p in opt_model.named_parameters() if "l0_module" in n and "lambda" not in n]
            lagrange_params = [p for n, p in opt_model.named_parameters() if "l0_module" in n and "lambda" in n]

            optimizer_grouped_parameters = optimizer_grouped_parameters + [
                {
                    "params": l0_module_params,
                    "weight_decay": 0.0,
                    "lr": self.args.lag_lr,
                },
                {
                    "params": lagrange_params,
                    "weight_decay": 0.0,
                    "lr": -(self.args.lag_lr),
                },
            ]
            
            # for n, p in opt_model.named_parameters():
            #     if p.requires_grad:
            #         print(n)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
        # super(LLaVATrainer, self)._save_checkpoint(model, trial)

        if getattr(self.args, 'save_vision_tower', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.model.get_vision_tower().image_processor.save_pretrained(
                os.path.join(output_dir, 'vision_tower'))
            self.model.get_vision_tower().config.save_pretrained(
                os.path.join(output_dir, 'vision_tower'))
            weight_to_save = get_vision_tower_state_maybe_zero_3(
                self.model.get_vision_tower().vision_tower.named_parameters())
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                torch.save(weight_to_save, os.path.join(
                    output_dir, 'vision_tower/pytorch_model.bin'))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(LLaVATrainer, self)._save(output_dir, state_dict)

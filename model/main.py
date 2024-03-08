from configs import get_config
from solver import Solver
from data_loader import get_loader

import wandb

if __name__ == '__main__':
    config = get_config(mode='train')
    test_config = get_config(mode='test')

    print(config)
    print(test_config)

    wandb.init(
        project="ac-sum-gan-0",
        
        config={
        "dataset": "TVSum",
        "epochs": config.n_epochs,
        "split_index": config.split_index,
        }
    )

    wandb.define_metric("custom_step")
    wandb.define_metric("original_prob", step_metric="custom_step")
    wandb.define_metric("sum_prob", step_metric="custom_step")
    wandb.define_metric("recon_loss_init_epoch", step_metric="custom_step")
    wandb.define_metric("recon_loss_epoch", step_metric="custom_step")
    wandb.define_metric("prior_loss_epoch", step_metric="custom_step")
    wandb.define_metric("g_loss_epoch", step_metric="custom_step")
    wandb.define_metric("e_loss_epoch", step_metric="custom_step")
    wandb.define_metric("d_loss_epoch", step_metric="custom_step")
    wandb.define_metric("c_original_loss_epoch", step_metric="custom_step")
    wandb.define_metric("c_summary_loss_epoch", step_metric="custom_step")
    wandb.define_metric("sparsity_loss_epoch", step_metric="custom_step")
    wandb.define_metric("actor_loss_epoch", step_metric="custom_step")
    wandb.define_metric("critic_loss_epoch", step_metric="custom_step")
    wandb.define_metric("reward_epoch", step_metric="custom_step")
    
    train_loader = get_loader(config.mode, config.split_index, config.action_state_size)
    test_loader = get_loader(test_config.mode, test_config.split_index, test_config.action_state_size)
    solver = Solver(config, train_loader, test_loader, config.ckpt)

    solver.build()
    solver.evaluate(-1)  # evaluates the summaries generated using the initial random weights of the network
    solver.train()

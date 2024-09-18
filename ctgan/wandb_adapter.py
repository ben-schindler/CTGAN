"""
This is an adapter for tensorboardX Framework
"""
import wandb


class WandBSummarizer():
    def __init__(self, run_name, dir):
        wandb.init(name=run_name,
                   project=dir)

    def finish(self):
        wandb.finish()

    def add_Gener_trainstep(self, current_iteration, lossG):
        wandb.log({"train/Generator_loss": lossG}, step=current_iteration)

    def add_Discr_trainstep(self, current_iteration, lossD, acc=None):
        wandb.log({"train/Discriminator_loss": lossD}, step=current_iteration)
        if acc is not None:
            wandb.log({"train/Discriminator_accuracy": acc}, step=current_iteration)

    def add_NNM_trainstep(self, current_iteration, lossNNM):
        wandb.log({"train/NNM_loss": lossNNM}, step=current_iteration)

    def add_Selector_trainstep(self, current_iteration, lossS):
        wandb.log({"train/Selector_loss": lossS}, step=current_iteration)

    def add_Gener_validation(self, current_iteration, lossG):
        wandb.log({"valid/Generator_loss": lossG}, step=current_iteration)

    def add_Discr_validation(self, current_iteration, lossD, acc):
        wandb.log({"valid/Discriminator_loss": lossD}, step=current_iteration)
        wandb.log({"valid/Discriminator_accuracy": acc}, step=current_iteration)

    def add_NNM_validation(
        self,
        current_iteration,
        sum_of_MSE,
        abs_SigmaDev=None,
        sqrd_SigmaDev=None,
        mean_Sigma_MAPE=None,
        mean_Sigma_MAE=None,
        mean_Sigma_RMSE=None,
    ):
        wandb.log({"valid/NNM_Sum_of_MSE": sum_of_MSE}, step=current_iteration)

        # fmt: off
        if abs_SigmaDev is not None:
            wandb.log({"valid/absolute_SigmaDeviation": abs_SigmaDev}, step=current_iteration)
        if sqrd_SigmaDev is not None:
            wandb.log({"valid/squared_SigmaDeviation": sqrd_SigmaDev}, step=current_iteration)
        if mean_Sigma_MAPE is not None:
            wandb.log({"valid/mean_Sigma_MAPE": mean_Sigma_MAPE}, step=current_iteration)
        if mean_Sigma_MAE is not None:
            wandb.log({"valid/Sigma_MAE": mean_Sigma_MAE}, step=current_iteration)
        if mean_Sigma_RMSE is not None:
            wandb.log({"valid/Sigma_RMSE": mean_Sigma_RMSE}, step=current_iteration)
        # fmt: on

    def add_validation(self, current_iteration, metrics):
        for key, value in metrics.items():
            wandb.log(
                {"valid/" + str(key): value}, step=current_iteration
            )



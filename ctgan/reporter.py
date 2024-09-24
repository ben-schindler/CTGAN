"""ctGAN Reporter"""

from ctgan.wandb_adapter import WandBSummarizer
import warnings
import itertools
import numpy as np
import torch
import os
import pandas as pd

from sdmetrics.reports.single_table.quality_report import QualityReport
from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity

class CTGANReporter():
    def __init__(self, run_name):
        self.run_name = run_name
        self.summarizer = WandBSummarizer(self.run_name, dir="CTGAN")
        self.best_score = 0.0 #score is between 0 and 1
        self.current_iteration = 0

    def reportTrainstep(self, loss_D, loss_G, current_iteration):
        self.summarizer.add_Gener_trainstep(current_iteration, loss_G)
        self.summarizer.add_Discr_trainstep(current_iteration, loss_D)
        self.current_iteration = current_iteration

    def reportValidation(
        self, fake_data, test_data, metadata, current_iteration):
        # ToDo - add arguments:
        # discr_predicts_real, discr_predicts_fake,

        ##############################
        # Compute Evaluation Metrics #
        ##############################
        metrics, variat_perf = self.__evaluation_metrics(fake_data, test_data, metadata)
        self.best_score = max(self.best_score,
                              variat_perf["variational_performance"])  # score is between 0 and 1
        ###############
        # SummaryWriter
        ###############
        self.summarizer.add_validation(current_iteration, metrics)

        return self.best_score == variat_perf["variational_performance"]

    def evaluateEnsemble(self, fake_data, netD, epoch=0):
        """
            Save Sample Gradients and individual Discriminator Outputs
        """

        if not isinstance(fake_data, torch.Tensor):
            fake_data = torch.tensor(fake_data, requires_grad=True)

        D_outs, fake_out_grads = [], []

        netD.zero_grad()
        fake_data.retain_grad()
        for n in range(netD.n_of_discr):
            D_out = netD.forward_single(fake_data, n)
            D_outs.append(D_out)
            loss_G = -D_out.mean()
            loss_G.backward(retain_graph=True)
            fake_out_grads.append(fake_data.grad)
            fake_data.grad = None
        # Batch x Discr
        D_outs = torch.concat(D_outs, dim=1)
        # Batch x Discr x Feature
        fake_out_grads = torch.stack(fake_out_grads, dim=1)

        with torch.no_grad():
            self.__save_samples_as_csv(D_outs.detach(), epoch=epoch, subfolder="ensemble",
                                        name_prefix="Discr_Outs", fmt='%.5f')
            self.__save_tensor_as_npz(fake_out_grads.detach(), epoch=epoch,
                                       subfolder="ensemble", name_prefix="fake_sample_grads")


    def __evaluation_metrics(self, fake_data, valid_data, metadata, check_amount_of_samples=True):
        if check_amount_of_samples:
            print("synthesized samples: \t{}".format(fake_data.shape[0]))
            print("validation samples: \t{}".format(valid_data.shape[0]))

            if fake_data.shape[0] < valid_data.shape[0]:
                valid_data = valid_data[-fake_data.shape[0]:]
                warnings.warn("pruned validation samples to: \t{}".format(valid_data.shape[0]))
            elif fake_data.shape[0] > valid_data.shape[0]:
                fake_data = fake_data[-valid_data.shape[0]:]
                warnings.warn("pruned synthetic samples during evaluation to: {}".format(fake_data.shape[0]))

        metrics = {}

        report = QualityReport()

        # this metadata conversion is necessary as CTGAN and sdmetrics use different structure for
        # their meta-data:
        metadata_converted = {'columns': {item['name']: {"sdtype": item['type']} for item in metadata['columns']}}
        # convert types: continuous -> numerical
        metadata_converted = {'columns': { col: {k: 'numerical' if v == 'continuous' else v
                                                 for k, v in attributes.items()}
                                           for col, attributes in metadata_converted['columns'].items()}}
        # convert types: ordinal -> categorical
        metadata_converted = {'columns': {col: {k: 'categorical' if v == 'ordinal' else v
                                                for k, v in attributes.items()}
                                          for col, attributes in
                                          metadata_converted['columns'].items()}}

        report.generate(valid_data, fake_data, metadata_converted)
        properties = report.get_properties().set_index('Property')

        #quality_report = {"quality_report": report.get_score()}
        variat_perf = {"variational_performance": report.get_score()}
        column_shape = properties.loc["Column Shapes"].values[0]
        column_pair_trends = properties.loc["Column Pair Trends"].values[0]

        metrics = {"Column_Shape": column_shape, "Column_Pair_Trend": column_pair_trends} | metrics
        metrics = variat_perf | metrics

        return metrics, variat_perf

    def __create_dirs(self, dirs):
        """
        dirs - a list of directories to create if these directories are not found
        :param dirs:
        :return:
        """
        if not isinstance(dirs, list):
            dirs = [dirs]
        try:
            for dir_ in dirs:
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
        except Exception as err:
            warnings.warn("Creating directories error: {0}".format(err))

    def __save_samples_as_csv(
            self,
            batch,
            epoch=0,
            name_prefix="",
            subfolder="",
            header=None,
            use_spectra_colnames=False,
            *args,
            **kwds
    ):
        """
        Saving a batch of tensors as csv
        :param batch: Tensor of shape (B,84)
        :param epoch: the number of current iteration
        :return: img_epoch: which will contain the image of this epoch
        """
        if subfolder != "":
            self.__create_dirs(["logs/" + self.run_name + '/' + subfolder])


        if epoch is not None:
            file_name = "{}_epoch_{:d}.csv".format(name_prefix, epoch)
        else:
            file_name = "{}.csv".format(name_prefix)
        csv_path = os.path.join("logs", self.run_name, subfolder, file_name)

        if isinstance(batch, torch.Tensor):
            out = batch.numpy()
        elif isinstance(batch, np.ndarray):
            out = batch
        elif isinstance(batch, pd.DataFrame):
            out = batch.values
            header = str(list(batch)).replace("'", "").replace("[", "").replace("]", "")
        else:
            raise TypeError("Batch must be numpy array, pandas Dataframe, or Torch Tensor")

        if header is None:
            header = ""

        np.savetxt(csv_path, out, delimiter=",", header=header, *args, **kwds)

    def __save_tensor_as_npz(
            self,
            tensor,
            epoch=0,
            name_prefix="",
            subfolder="",
            *args,
            **kwds
    ):
        """
        Saving a tensor as numpy npz-file
        :param tensor: input tensor to be saved
        :param epoch: the number of current iteration
        :return: img_epoch: which will contain the image of this epoch
        """
        if subfolder != "":
            self.__create_dirs(["logs/" + self.run_name + '/' + subfolder])

        if epoch is not None:
            file_name = "{}_epoch_{:d}".format(name_prefix, epoch)
        else:
            file_name = "{}".format(name_prefix)

        file_path = os.path.join("logs", self.run_name, subfolder, file_name)

        if isinstance(tensor, torch.Tensor):
            out = tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            out = tensor
        elif isinstance(tensor, pd.DataFrame):
            out = np.ndarray(tensor.values)
        else:
            raise TypeError("Batch must be numpy array, pandas Dataframe, or Torch Tensor")


        np.savez(file_path, out, *args, **kwds)



############################
# Deprecated Code Snippets #
############################
'''

# Column Shape metrics
        for column, info in metadata_converted['columns'].items():
            if info['sdtype'] == 'numerical':
                metrics[f"KS_{column}"] = KSComplement.compute(valid_data[column],
                                                                          fake_data[column])
            elif info['sdtype'] == 'categorical':
                metrics[f"TV_{column}"] = TVComplement.compute(valid_data[column],
                                                                          fake_data[column])

        # Column-pair-wise metrics
        columns = list(metadata_converted['columns'].keys())
        for col1, col2 in itertools.combinations(columns, 2):
            if metadata_converted['columns'][col1]['sdtype'] == 'numerical' and metadata_converted['columns'][col2][
                'sdtype'] == 'numerical':
                metrics[f"CorSim_{col1}_{col2}"] = CorrelationSimilarity.compute(
                    valid_data[[col1, col2]], fake_data[[col1, col2]]
                )
            elif metadata_converted['columns'][col1]['sdtype'] == 'categorical' and metadata_converted['columns'][col2][
                'sdtype'] == 'categorical':
            #else:
                metrics[f"ConSim_{col1}_{col2}"] = ContingencySimilarity.compute(
                    valid_data[[col1, col2]], fake_data[[col1, col2]]
                )

        # Aggregate Metrics:

        # Separate column-wise and column-pair-wise metrics
        column_shape = {k: v for k, v in metrics.items() if k.startswith(('KS_', 'TV_'))}
        column_pair_trends = {k: v for k, v in metrics.items() if
                            k.startswith(('ConSim_', 'CorSim_'))}

        # Calculate means
        column_shape = np.mean(list(column_shape.values()))
        column_pair_trends = np.mean(list(column_pair_trends.values()))

'''

from torch import from_numpy
import pandas as pd
import data_formatters.utils as utils
from data_formatters.base import InputTypes
from torch.utils.data import Dataset
import numpy as np
import click
from os import path

class TSDataset(Dataset):
    ## Mostly adapted from original TFT Github, data_formatters
    def __init__(self, cnf, data_formatter):

        self.params = cnf.all_params

        self.csv = utils.data_csv_path(cnf.ds_name)
        self.data = pd.read_csv(self.csv, index_col=0, na_filter=False)

        self.train_set, self.valid_set, self.test_set = data_formatter.split_data(self.data)
        self.params['column_definition'] = data_formatter.get_column_definition()

        self.inputs = None
        self.outputs = None
        self.time = None
        self.identifiers = None

    def train(self):
        max_samples = self.params['train_samples']
        if path.exists(utils.csv_path_to_folder(self.csv) + "processed_traindata.npz"):
            f = np.load(utils.csv_path_to_folder(self.csv) + "processed_traindata.npz", allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[
                f.files[3]]
        else:
            self.preprocess(self.train_set, max_samples)
            np.savez(utils.csv_path_to_folder(self.csv) + "processed_traindata.npz", self.inputs, self.outputs,
                     self.time,
                     self.identifiers)

    def test(self):
        max_samples = self.params['test_samples']
        if path.exists(utils.csv_path_to_folder(self.csv) + "processed_testdata.npz"):
            f = np.load(utils.csv_path_to_folder(self.csv) + "processed_testdata.npz", allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[
                f.files[3]]
        else:
            self.preprocess(self.test_set, max_samples)
            np.savez(utils.csv_path_to_folder(self.csv) + "processed_testdata.npz", self.inputs, self.outputs,
                     self.time,
                     self.identifiers)

    def val(self):
        max_samples = self.params['val_samples']
        if path.exists(utils.csv_path_to_folder(self.csv) + "processed_validdata.npz"):
            f = np.load(utils.csv_path_to_folder(self.csv) + "processed_validdata.npz", allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[
                f.files[3]]
        else:
            self.preprocess(self.valid_set, max_samples)
            np.savez(utils.csv_path_to_folder(self.csv) + "processed_validdata.npz", self.inputs, self.outputs,
                     self.time,
                     self.identifiers)

    def preprocess(self, data, max_samples):
        time_steps = int(self.params['total_time_steps'])
        input_size = int(self.params['input_size'])
        output_size = int(self.params['output_size'])
        column_definition = self.params['column_definition']

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)
        print('Getting valid sampling locations.')
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            # print('Getting locations for {}'.format(identifier))
            num_entries = len(df)
            if num_entries >= time_steps:
                valid_sampling_locations += [
                    (identifier, time_steps + i)
                    for i in range(num_entries - time_steps + 1)
                ]
            split_data_map[identifier] = df

        self.inputs = np.zeros((max_samples, time_steps, input_size))
        self.outputs = np.zeros((max_samples, time_steps, output_size))
        self.time = np.empty((max_samples, time_steps, 1), dtype=object)
        self.identifiers = np.empty((max_samples, time_steps, 1), dtype=object)
        print('# available segments={}'.format(len(valid_sampling_locations)))

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [
                valid_sampling_locations[i] for i in np.random.choice(
                    len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        for i, tup in enumerate(ranges):
            if ((i + 1) % 1000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - time_steps:start_idx]

            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i, :, 0] = sliced[id_col]

    def __getitem__(self, index):

        num_encoder_steps = int(self.params['num_encoder_steps'])
        s = {
            'inputs': self.inputs[index].astype(float),
            'outputs': self.outputs[index, num_encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[index, num_encoder_steps:, :]),
            'time': self.time[index].tolist(),
            'identifier': self.identifiers[index].tolist()
        }

        return s

    def __len__(self):
        return self.inputs.shape[0]

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""
        return utils.get_single_col_by_input_type(input_type, self.params['column_definition'])


@click.command()
@click.option('--conf_file_path', type=str, default="./conf/electricity.yaml")
def main(conf_file_path):
    import data_formatters.utils as utils
    from conf import Conf

    cnf = Conf(conf_file_path=conf_file_path, seed=15, exp_name="test", log=False)
    data_formatter = utils.make_data_formatter(cnf.ds_name)
    dataset_train = TSDataset(cnf, data_formatter)
    dataset_train.train()

    for i in range(10):
        # 192 x ['power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
        x = dataset_train[i]['inputs']
        # 24 x ['power_usage']
        y = dataset_train[i]['outputs']
        print(f'Example #{i}: x.shape={x.shape}, y.shape={y.shape}')


if __name__ == "__main__":
    main()
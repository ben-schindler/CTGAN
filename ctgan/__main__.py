"""CLI."""

import argparse
import datetime
import os
import os.path as path
import json

from ctgan.data import read_csv, read_tsv, write_tsv
from ctgan.synthesizers.ctgan import CTGAN


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument(
        '-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV'
    )
    parser.add_argument(
        '--no-header',
        dest='header',
        action='store_false',
        help='The CSV file has no header. Discrete columns will be indices.',
    )

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument(
        '-d', '--discrete', help='Comma separated list of discrete columns without whitespaces.'
    )
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        help='Number of rows to sample. Defaults to the training data size',
    )

    parser.add_argument(
        '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.'
    )
    parser.add_argument(
        '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.'
    )

    parser.add_argument(
        '--generator_decay', type=float, default=1e-6, help='Weight decay for the generator.'
    )
    parser.add_argument(
        '--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.'
    )

    parser.add_argument(
        '--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.'
    )
    parser.add_argument(
        '--generator_dim',
        type=str,
        default='256,256',
        help='Dimension of each generator layer. ' 'Comma separated integers with no whitespaces.',
    )
    parser.add_argument(
        '--discriminator_dim',
        type=str,
        default='256,256',
        help='Dimension of each discriminator layer. '
        'Comma separated integers with no whitespaces.',
    )

    parser.add_argument(
        '--batch_size', type=int, default=500, help='Batch size. Must be an even number.'
    )
    parser.add_argument(
        '--save', default=None, type=str, help='A filename to save the trained synthesizer.'
    )
    parser.add_argument(
        '--load', default=None, type=str, help='A filename to load a trained synthesizer.'
    )

    parser.add_argument(
        '--sample_condition_column', default=None, type=str, help='Select a discrete column name.'
    )
    parser.add_argument(
        '--sample_condition_column_value',
        default=None,
        type=str,
        help='Specify the value of the selected discrete column.',
    )

    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')

    ## Logger:
    parser.add_argument(
        "--add_time_to_exp",
        action="store_true",
        default=False,
        help="Add time to experiment name and output file",
    )

    ## Ensemble Arguments:

    parser.add_argument(
        '--ensemble',
        default=False,
        type=bool,
        help='Whether a discriminator ensemble should be used. Defaults to False.'
    )

    parser.add_argument(
        '--ens_multiplier',
        default=1,
        type=int,
        help='Number of Discriminators used for the Ensemble. Defaults to 1.'
    )

    parser.add_argument(
        '--ens_weighting',
        default='ew',
        type=str,
        help='Weighting Method among Discriminator used for Generator Training. Defaults to "ew".'
    )

    parser.add_argument(
        '--ens_fixed_weights',
        nargs='*',
        type=float,
        default=None,
        help='List of weights used for fixed weights as a list of floats. Only used when ens_weighting equals "fixed". Defaults to None.'
    )

    parser.add_argument(
        '--ens_split_batch',
        default=False,
        type=bool,
        help='Whether Batch is split among Discriminators. Note that when True, batchsize must be divisible by ens_multiplier. Defaults to False.'
    )

    parser.add_argument(
        '--ens_grad_norm',
        default=False,
        type=bool,
        help='Whether to use Gradient Normalization among Discriminators. Defaults to False.'
    )

    parser.add_argument(
        '--save_sample_gradients',
        default=False,
        type=bool,
        help='Whether to save discriminator gradients for fake data. Defaults to False.'
    )

    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
    else:
        data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)

    #load sdv-style metadata
    with open(args.metadata) as f:
        metadata = json.load(f)


    ## train-test-split

    data = data.sample(frac=1, random_state=161).reset_index(drop=True)

    n_train_samples=round(data.shape[0]*0.8)
    test_data = data[n_train_samples:]
    data = data[:n_train_samples]

    if args.add_time_to_exp:
        file, suffix = path.splitext(args.output)
        timestamp = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        args.output = file + "_" + timestamp + suffix
    exp_name = path.basename(args.output)

    if args.load:
        model = CTGAN.load(args.load)
    else:
        generator_dim = [int(x) for x in args.generator_dim.split(',')]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
        model = CTGAN(
            embedding_dim=args.embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=args.generator_lr,
            generator_decay=args.generator_decay,
            discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            ensemble=args.ensemble,
            ens_multiplier=args.ens_multiplier,
            ens_weighting=args.ens_weighting,
            ens_fixed_weights=args.ens_fixed_weights,
            ens_split_batch=args.ens_split_batch,
            ens_grad_norm=args.ens_grad_norm,
            save_sample_gradients=args.save_sample_gradients,
            verbose=True,
            exp_name=exp_name
        )
    model.fit(data, discrete_columns, test_data=test_data, metadata=metadata)

    if args.save is not None:
        model.save(args.save)

    num_samples = args.num_samples or len(data)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    sampled = model.sample(
        num_samples, args.sample_condition_column, args.sample_condition_column_value
    )

    out_path = os.path.join("logs", args.output)

    if args.tsv:
        out_path = os.path.join(out_path, "data.tsv")
        write_tsv(sampled, args.metadata, out_path)
    else:
        out_path = os.path.join(out_path, "data.csv")
        sampled.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()

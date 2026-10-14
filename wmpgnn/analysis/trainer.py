from optparse import OptionParser
import copy
import yaml

from wmpgnn.analysis.config_adjusting import *
from wmpgnn.analysis.load_module import *
from wmpgnn.data_loader.get_data_loader import load_trn_val_loader, load_tst_loader
from wmpgnn.data_loader.weights_calculator import transform_pos_weight
from wmpgnn.lightning_module.exec_lightning import training, evaluate
from wmpgnn.performance.plot_results import metrics_eval

if __name__ == "__main__":

    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))
    print("=" * 45)
    print("Starting the training script")


    with open(option.CONFIG, "r") as file:
        configs = yaml.safe_load(file)
        save_config = copy.deepcopy(configs)
    configs = adjust_config_training(configs)

    configs, weights, trn_loader, val_loader, chunkloader = load_trn_val_loader(configs)
    pos_weights = transform_pos_weight(weights, configs["inference"])


    if configs["model"] == "DFEI":

        module = load_module(configs, pos_weights)
    elif configs["model"] == "IFT":


        configs, dfei_model = load_dfei_for_ift(configs)
        module = load_module(configs, pos_weights, dfei_model=dfei_model)
    else:
        raise RuntimeError("No configuration file specified")

    """Start the training"""
    trainer = training(module, configs, trn_loader=trn_loader, val_loader=val_loader, chunkloader=chunkloader)
    version = trainer.logger.version


    with open(f"{configs['log_dir']}/{configs['model']}/version_{version}/input_config.yaml", "w") as file:
        yaml.dump(save_config, file, default_flow_style=False)

    """Start the testing"""
    configs, tst_loader, chunkloader = load_tst_loader(configs)
    evaluate(trainer, module, tst_loader=tst_loader, chunkloader=chunkloader)

    metric_path = f"{configs['log_dir']}/{configs['model']}/version_{version}/metrics.csv"
    metrics_eval(metric_path, configs, version)
    print("Done")
    print("=" * 45)

#!/usr/bin/env python
import os
import argparse
import pprint
from tqdm import tqdm
from jarvis.db.jsonutils import loadjson
from chipstb.config import CHIPSTBConfig
from chipstb.dftb import main as main_dftb


def main():
    parser = argparse.ArgumentParser(
        description="Run CHIPSTB Tight-Binding Analyzer"
    )
    parser.add_argument(
        "--input_file",
        default="tb_input.json",
        type=str,
        help="Path to the input configuration JSON file",
    )
    args = parser.parse_args()

    input_dict = loadjson(args.input_file)
    config = CHIPSTBConfig(**input_dict)
    pprint.pprint(config.dict())

    # Determine calculator types to run
    calculators = config.calculator_types or [config.calculator_type]
    if not calculators:
        raise ValueError("No calculator_type or calculator_types specified.")

    # Determine structure sources
    jids = config.jid_list or ([config.jid] if config.jid else [])

    if config.structure_path:
        for calc in calculators:
            print(
                f"Running TB analysis on structure {config.structure_path} using {calc}..."
            )
            tb = TBAnalyzer(config=config, calculator_type=calc)
            tb.run_local()
        return

    if jids:
        for j, jid in enumerate(jids):

            for c, calc in enumerate(calculators):
                if calc == "dftb+":
                    main_dftb(
                        jid=jids[j],
                        dftb_executable=config.calculator_executables[c],
                        k_mesh=config.k_mesh[j],
                        sk_dir=config.sk_or_model_dir[j],
                        config=config,
                    )
                print(f"Running TB analysis on {jid} using {calc}...")
    else:
        raise ValueError("No valid structure_path or JID(s) provided.")


if __name__ == "__main__":
    main()

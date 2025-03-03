<a id="readme-top"></a>
# Uncertainty Estimation for proteochemometric bioactivity prediction models using Deep Learning 


[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![wandb](https://img.shields.io/badge/Weights_%26_Biases-black?logo=weightsandbiases&logoColor=yellow)](https://wandb.ai/site)
[![arxiv](https://img.shields.io/badge/Preprint-arXiv:123123)](https://arxiv.org/) 


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
            <li><a href="#dataprocessing">Data Processing</a></li>
            <li><a href="#models">Models</a></li>
        </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#citation">Citation</a></li>

  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This repo represents a comparative study, applying and comparing various Uncertainty Estimation methods


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Project Folder Structure

```
uqdd/
│── README.md
│── LICENSE
│── environment.yml
│── .gitignore
│── uqdd/  # Source Code Directory
│   ├── data/
│   │   ├── data_papyrus.py
│   │   ├── data_chembl.py
│   │   ├── data_max_curated.py
│   │   ├── utils_data.py
│   │   ├── utils_chem.py
│   │   ├── utils_assay.py
│   │   ├── utils_prot.py
│   ├── models/
│   │   ├── pnn.py
│   │   ├── evidential.py
│   │   ├── ensemble.py
│   │   ├── mcdropout.py
│   │   ├── utils_metrics.py
│   │   ├── utils_models.py
│   │   ├── utils_train.py
│   ├── config/
│   │   ├── pnn.json
│   │   ├── evidential.json
│   │   ├── ensemble.json
│   │   ├── mcdropout.json
│   │   ├── eoe.json
│   │   ├── emc.json
│   │   ├── papyrus.json
│   ├── experiments/ 
│   │   ├── run_experiments.py
│   │   ├── benchmark_results.py
│   ├── logs/
│   ├── figures/
│   ├── tests/
│   │   ├── test_data.py
│   │   ├── test_models.py
│   │   ├── test_features.py
│   │   ├── test_experiments.py
│── notebooks/  # Jupyter Notebooks for Exploratory Analysis
│   ├── data_analysis.ipynb
│   ├── model_performance.ipynb
│── docs/  # Documentation
│── results/  # Processed Results and Outputs
```

### Prerequisites




### Installation


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


### Data Processing
Run the following scripts to preprocess the datasets:

```sh
python uqdd/data/data_papyrus.py
```
This will generate the preprocessed datasets in the `data/` directory.

As an example to use `data_papyrus.py`:
```shell
python uqdd/data/data_papyrus.py --activity xc50 --descriptor-protein ankh-large --descriptor-chemical ecfp2048 --split-type time --n-targets -1 --file-ext pkl --sanitize --verbose
```

### Models
The package provides several scripts to train and test different models. The main entry point is the `model_parser.py` script, which allows you to specify various options for data, model, and training configuration.

#### Model Configuration Options

The `model_parser.py` script allows you to specify various options for data, model, and training configuration. The following options are available:

- `--model`: Model type (baseline, ensemble, mcdropout, evidential)
- `--data_name`: Dataset name (papyrus, chembl, max_curated)
- `--n_targets`: Number of targets to train on (-1 for all targets)
- `--activity_type`: Activity type (xc50, kx) 
- `--descriptor_protein`: Protein descriptor type (ankh-large, ankh-small, unirep, ) 
- `--descriptor_chemical`: Chemical descriptor type (ecfp2048, ecfp1024, , )
- `--split_type`: Split type (random, scaffold, time)
- `--ext`: File extension (pkl, parquet, csv)
- `--task_type`: Task type (regression, classification)
- `--wandb_project_name`: Weights and Biases project name for logging
- `--ensemble_size`: Ensemble size for ensemble models
- `--num_mc_samples`: Number of MC samples for MC-Dropout models
- `--seed`: Random seed for reproducibility
- `--epochs`: Number of epochs for training
- `--batch_size`: Batch size for training
- `--lr`: Learning rate for training
- `--seed`: Random seed for reproducibility
- `--device`: Device for training (cpu, cuda)
#### **PNN Models**
To train and test the baseline model, use the following command:

```shell
python uqdd/models/model_parser.py --model pnn --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name pnn-test
```

#### **Ensemble Models**

To train and test the ensemble model, use the following command:

```shell
python uqdd/models/model_parser.py --model ensemble --ensemble_size 100 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name ensemble-test
  
```

#### **MC-Dropout Models**

To train and test the MC-Dropout model, use the following command:

```shell
python uqdd/models/model_parser.py --model mcdropout --num_mc_samples 100 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name mcdp-test
```

#### **Evidential Models**

To train and test the Evidential model, use the following command:

```shell
python uqdd/models/model_parser.py --model evidential --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name evidential-test
```

#### **Ensemble of Evidential (EOE) Models**

To train and test the Ensemble of Evidential model, use the following command:

```shell
python uqdd/models/model_parser.py --model eoe --ensemble_size 100 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name eoe-test
```

#### **Evidential MC-Dropout (EMC) Models**

To train and test the Evidential MC-Dropout model, use the following command:

```shell
python uqdd/models/model_parser.py --model emc --num_mc_samples 100 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name emc-test
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Bola Khalil [@bola-khalil](https://www.linkedin.com/in/bola-khalil/) - b.a.a.khalil@lacdr.leidenuniv.nl


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

B.K. acknowledges funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 955879. [DRUGTrain](www.drugtrain.eu/). B.K., N.D., and H.V.V. are affiliated with Janssen Pharmaceutica, a Johnson and Johnson company.

K.S. acknowledges funding from the ELLIS Unit Linz. The ELLIS Unit Linz, the LIT AI Lab, and the Institute for Machine Learning are supported by the Federal State of Upper Austria

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [X] Start README content
- [X] Add back to top links
- [ ] Add Env setup
- [ ] Add 


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CITATION -->
## Citation

If you find this work useful, please cite the following paper:

```bibtex
@article{Khalil2025,
  title={Uncertainty Quantification in Bioactivity Assessment Using Ensemble, MC Dropout, Evidential, and Hybrid Models},
  author={Khalil, Bola and Schweighofer, Kajetan and Dyubankova, Natalia and Klambauer, Guenter and van Westen, Gerard J.P. and Van Vlijmen, Herman},
  journal={arXiv},
  year={2025}
}
```
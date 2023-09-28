# GRANDE: Gradient-Based Decision Tree Ensembles

This repository provides the code for the paper ***"GRANDE: Gradient-Based Decision Tree Ensembles"***. 

The required versions of all packages are listed in "*relevant_dependencies.txt*" and can be installed running "*install_requirements.sh*". We further provide the following files to reproduce the results from the paper:
* *GRANDE-HPO-BS-BIN.ipynb*: Load the optimized parameters and generate results for all binary classification datasets from the paper.
* *GRANDE-DEFAULT-BS-BIN.ipynb*: Use defualt parameters for all approaches and generate results for all binary classification datasets from the paper.
* *GRANDE-caseStudy-PhishingWebsites.ipynb*: Reproduce the case study for the PhishingWebsites dataset including explanations from Anchors.
* *HPO_GRANDE_UPDATE_BS-BIN-250trials.ipynb*: Code used for HPO containing the search space for all approaches

We want to note that it is unfortunately not possible to generate reproducible results on a GPU with TensorFlow, even when setting the seed. However this only results in minor deviations of the performance and does not change the results shown in the paper.

By default, the GPU is used for the calculations of all approaches. This can be changed in the config ("*use_gpu*") but has a significant impact on the runtime. If you receive a memory error for GPU computations, this is most likely due to the parallelization of the experiments on a single GPU. Please try to reduce the number of parallel computations ("*jobs_per_gpu*") or run on CPU (set "*use_gpu*" to "*False*"). Please note that the single-GPU parallelization reduces the overall runtime of the experiments, but negatively influences the individual runtimes. Therefore the runtimes noted in the paper are generated in a separate setting without single-GPU parallelization of multiple folds/datasets.

The Python version used in the experiments was Python 3.11.4. The code was tested with a Linux distribution. For other distributions it might not work as intended. If there occor errors when running the "*install_requirements.sh*", please consider installing the packages manually. Thereby, it is important to follow the sequence of "*relevant_dependencies.txt*".

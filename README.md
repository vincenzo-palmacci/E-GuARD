### Fake The False: a simulated human in the loop (HITL) framework for improving quantitative to structure interference relation (QSIR) Machine Learning models

#### ABSTRACT

Selecting the optimal molecule for further optimization is challenging due to the prevalence of false positive results from High-Throughput Screening (HTS) experiments. To identify these nuisance compounds, various experimental protocols and in-silico methods have been developed, with the latter being favored for their speed and cost-effectiveness. However, in-silico approaches, particularly those based on machine learning, require extensive data for training and practical application. This data is typically derived from counter-screening assays that experimentally identify compounds causing false positive readouts. Despite this, both public and private sectors face a shortage of such data, limiting the applicability of models trained on experimental interfering compounds.
In this work, we aim to enhance the performance and extend the applicability domain of these models by increasing the training data through the generation of new molecules and retraining machine learning models on the augmented dataset. We utilize REINVENT 3.2, scored with machine learning models for interference prediction (Lies and Liabilities, Tropsha et al.), as the teacher model. Subsequently, we train a student model with the generated interfering compounds, selecting molecules for the training set via a simulated human-in-the-loop (HITL) sampling method (Nahal et al.).

#### GRAPHICAL ABSTRACT

[Overview of the FTF method.](figures/ftf_illustration.pdf) ah I need to upgrade my drawing software subscription to be able to export my figures again 😅

#### PREPROCESSING: Collect training data

The training data for our initial models were sourced from the previous work of Alves et al. [doi/10.1021/acs.jmedchem.3c00482]. In their study, the authors curated and assembled small datasets for detecting assay interference based on experimental evidence. The datasets encompass four types of interference mechanisms: (a) firefly luciferase inhibition, (b) nano luciferase interference, (c) thiol reactivity, and (d) redox activity. The compounds were already filtered and curated by the authors, so we simply converted the SDF files available in the supplementary materials into dataframes containing the SMILES notation and the experimental outcomes.

#### Note: add this to run export {PYTHONPATH="${PYTHONPATH}:/home2/vpalmacci/Projects/FTF4/REINVENT4/"}
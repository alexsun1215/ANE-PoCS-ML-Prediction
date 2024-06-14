# ANE-PoCS-ML-Prediction
# Machine Learning Prediction of Adverse Neurological Events in the Perioperative Period of Cardiac Surgery (ANE-PoCS)

## Overview
This project aims to predict major adverse neurological events (MANE) including postoperative stroke, postoperative atrial fibrillation, and postoperative death through machine learning model stacking for patients undergoing cardiac surgery (CABG and valve surgery). The models utilize data from four major databases: MIMIC-4, eICU, INSPIRE, and the data from Fuwai Hospital, Chinese Academy of Medical Sciences.

## Database
The dataset for this study is derived from a combination of the MIMIC-4, eICU, INSPIRE databases, and patient data from Fuwai Hospital, Chinese Academy of Medical Sciences. These datasets provide a comprehensive overview of patient care in intensive care units, making them invaluable for predictive modeling in the medical field.

## Installation

This project is developed using Python. To run the code, you will need Python installed on your machine along with several libraries.

### Prerequisites

- Python 3.6 or later
- pip (Python package manager)

### Libraries to Install

You will need to install several Python libraries including pandas, numpy, scikit-learn, and matplotlib. You can install these by running the following command in your terminal:

```bash
pip install pandas numpy scikit-learn matplotlib

## Getting Started

To get started with the ANE-PoCS-ML-Prediction project, follow these steps to clone the repository to your local machine and set up your environment.

### Clone the repository

1. Open your terminal.
2. Run the following command to clone the repository:

```bash
git clone https://github.com/alexsun1215/ANE-PoCS-ML-Prediction.git


### Navigate to the project directory
After cloning, move into the project directory:
cd ANE-PoCS-ML-Prediction

### Usage
The project contains six files in total; three of them are complete data preprocessing, feature selection, model development, evaluation, and prediction code for the ANE-PoCS model. The other three are for the simplified feature version of the model, ACCE-PoCS-20, for three outcomes.

###  Acknowledgments
Special thanks to the data providers of the MIMIC-4, eICU, and INSPIRE databases, and Fuwai Hospital for making their data available for this research.
MIMIC 4: https://physionet.org/content/mimiciv/2.2/
eICU：https://physionet.org/content/eicu-crd/2.0/
INSPIRE：https://physionet.org/content/inspire/1.2/
